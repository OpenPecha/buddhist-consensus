import os
import json
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from google import genai
from google.genai.types import EmbedContentConfig

# Load environment variables
load_dotenv()

# Setup API Keys
if "GOOGLE_API_KEY" not in os.environ:
    if os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "test_kangyur_tengyur")

if not all([MILVUS_URI, MILVUS_TOKEN, GEMINI_API_KEY]):
    print("Warning: Missing environment variables for Milvus or Gemini.")

# Initialize Clients
milvus_client = MilvusClient(
    uri=MILVUS_URI,
    token=MILVUS_TOKEN,
    collection_name=MILVUS_COLLECTION_NAME # Ensure consistency with notebook
)

# We need raw access to Gemini for embeddings to match the notebook's behavior exactly
genai_client = genai.Client(api_key=GEMINI_API_KEY)
doc_cfg = EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768)

def get_embedding(text: str) -> List[float]:
    """Generate embedding for the given text using Gemini."""
    resp = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=doc_cfg
    )
    return [i.values for i in resp.embeddings][0]

# --- Custom Hybrid Search Function ---
@tool
def hybrid_search_tool(query: str):
    """Searches the Tibetan knowledge base using hybrid search (semantic + keyword). 
    Returns relevant text segments with metadata for citation."""
    
    limit = 20
    
    # 1. Generate Embedding
    try:
        query_embedding = get_embedding(query)
    except Exception as e:
        return json.dumps({"error": f"Error generating embedding: {e}"})
    
    # 2. Prepare Search Requests
    # BM25 (Sparse) Request
    # expr = "parent_id == """ # User's commented out or broken filter attempt - removing for now to fix recursion
    req_bm25 = AnnSearchRequest(
        data=[query],
        anns_field="sparce_vector",
        param={},
        limit=limit
        # filter=expr # Removing broken filter
    )
    
    # Semantic (Dense) Request
    req_dense = AnnSearchRequest(
        data=[query_embedding],
        anns_field="dense_vector",
        param={"drop_ratio_search": 0.2},
        limit=limit
        # filter=expr # Removing broken filter
    )
    
    # 3. Perform Hybrid Search with RRF Reranking
    
    try:
        results = milvus_client.hybrid_search(
            collection_name=MILVUS_COLLECTION_NAME,
            reqs=[req_dense], 
            ranker=RRFRanker(),
            limit=limit,
            output_fields=["text", "title", "id"]
            # filter=expr # Removing broken filter
        )
    except Exception as e:
        return json.dumps({"error": f"Error searching Milvus: {e}"})
    
    # 4. Format Results (Return JSON string for raw structure)
    items = []
    for hits in results:
        for hit in hits:
            entity = hit.get("entity", {})
            item = {
                "id": entity.get("id") or hit.get("id"),
                "title": entity.get("title", "Unknown"),
                "text": entity.get("text", ""),
                "score": hit.get("score"),
                "distance": hit.get("distance") 
            }
            items.append(item)
            
    return json.dumps(items, ensure_ascii=False)

# --- Nodes and State ---

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Initialize Model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def generate_query_or_respond(state: State):
    """Decide whether to retrieve or respond directly."""
    # Force a system message to ensure tool usage is encouraged
    sys_msg = SystemMessage(content="You are a helpful assistant. You have access to a 'hybrid_search_tool' that searches a Tibetan Buddhist knowledge base. You MUST use this tool to answer questions about Buddhism, happiness, or life advice based on the texts. Do not answer from your own knowledge if it can be found in the texts.")
    
    # We need to prepend the system message if it's not already there, or just include it in the invocation
    # Since state["messages"] might already have history, we just prepend it for this call.
    messages = [sys_msg] + state["messages"]
    
    model_with_tools = llm.bind_tools([hybrid_search_tool])
    response = model_with_tools.invoke(messages)
    print(f"DEBUG: Model response: {response}")
    return {"messages": [response]}

def generate_answer(state: State):
    """Generate answer using retrieved context with strict citations."""
    messages = state["messages"]
    
    system_prompt = SystemMessage(content="""
    You are a helpful, friendly academic assistant for Buddhist studies, acting as a supportive friend.
    Use the provided retrieved context to answer the user's question in multiple languages if needed (English, Tibetan).
    
    CRITICAL CITATION RULES:
    1. Every single sentence or claim you make based on the text must be immediately followed by a citation.
    2. Use the EXACT format [ID] for citations. Do NOT use the title in the citation bracket, ONLY the ID.
       Example: "Emptiness is form [2pIapXDirmQdLVFLptm5r]."
    3. If the retrieved text is in Tibetan, quote the relevant Tibetan phrase in the answer where appropriate.
    4. If you cannot find the answer in the context, state that you don't know.
    5. Mention the book title too 

    FRIENDLY PERSONA:
    - Be warm, encouraging, and supportive.
    - Use a conversational tone while maintaining academic rigor with citations.

    MENTAL HEALTH DISCLAIMER:
    If the query relates to mental health (e.g., anxiety, depression, worry, suicide), you MUST:
    1. Start with a compassionate, friendly disclaimer: "I'm here for you, but I'm an AI. If you're feeling overwhelmed, please reach out to a professional."
    2. THEN proceed to answer the question using Buddhist perspectives found in the retrieved text. 
    Do NOT refuse to answer; provide the Buddhist context while maintaining the safety disclaimer.
    """)
    
    response = llm.invoke([system_prompt] + messages)
    return {"messages": [response]}

def rewrite_question(state: State):
    """Transform the query to produce a better question."""
    messages = state["messages"]
    # Find the last user message to rewrite
    question = messages[0].content # This assumes the first message is the query, but in a conversation flow, we might need the last one.
                                   # Notebook: question = messages[0].content
                                   # I will stick to notebook logic but for conversation we might want to be smarter.
                                   # However, strict adherence to notebook suggests messages[0].
                                   # But in a conversation, messages[0] is the OLDEST message.
                                   # Let's use the last HumanMessage.
    
    # Helper to find last human message
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if last_human:
        question = last_human.content
    else:
        question = messages[-1].content # Fallback
    
    msg = [
        HumanMessage(
            content=f"""Look at the input and try to reason about the underlying semantic intent / meaning. 
            Here is the initial question:
            \n{question}\n
            Formulate an improved question for a search engine to find Tibetan Buddhist texts:"""
        )
    ]
    response = llm.invoke(msg)
    # Note: Notebook returns HumanMessage here?
    # return {"messages": [HumanMessage(content=response.content)]}
    # This effectively replaces or adds a "refined query" as if the user said it.
    return {"messages": [HumanMessage(content=response.content)]}

class Grade(BaseModel):
    """Binary score for relevance check."""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(Grade)

def grade_documents(state: State):
    """Determines whether the retrieved documents are relevant."""
    messages = state["messages"]
    
    # Check recursion depth or loop count to prevent infinite loops
    # We can count how many times we've been here by counting ToolMessages or rewrites
    rewrite_count = len([m for m in messages if isinstance(m, HumanMessage) and "Look at the input" in str(m.content)])
    if rewrite_count > 3: # Force answer after 3 retries
         return "generate_answer"

    # Again, getting the question.
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    question = last_human.content if last_human else ""
    
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    if not tool_messages:
        return "generate_answer"
        
    # Get only the LATEST tool output to grade (not all history)
    latest_tool_msg = tool_messages[-1]
    docs_text = str(latest_tool_msg.content)
    
    # Check if empty result
    if not docs_text or docs_text == "[]" or "Error" in docs_text:
         # If search failed or returned nothing, maybe try one rewrite, but if we are already in a loop, stop.
         if rewrite_count > 1:
             return "generate_answer"
         return "rewrite_question"

    prompt = f"""You are a grader assessing relevance of retrieved Tibetan texts to a user question. \n 
    Here is the retrieved document content: \n\n {docs_text} \n\n
    Here is the user question: {question} \n
    If the document content seems even remotely related or helpful, grade it as 'yes'.
    Give a binary score 'yes' or 'no'."""
    
    scored_result = structured_llm_grader.invoke(prompt)
    
    if scored_result.binary_score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

# --- Build Graph ---
workflow = StateGraph(State)

workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([hybrid_search_tool]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END
    }
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate_answer": "generate_answer",
        "rewrite_question": "rewrite_question"
    }
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

app_graph = workflow.compile()

# --- FastAPI Application ---

app = FastAPI(title="Agentic RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str
    retrieved_items: List[Dict[str, Any]]

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the Chat UI at the root."""
    with open("chat_ui.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/chat", response_model=ChatResponse)
async def chat_main(request: ChatRequest):
    lc_messages = []
    for msg in request.messages:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            lc_messages.append(SystemMessage(content=msg.content))
            
    inputs = {"messages": lc_messages}
    
    # Use ainvoke to get final state
    final_state = await app_graph.ainvoke(inputs)
    messages = final_state["messages"]
    
    retrieved_items = []
    final_response = ""
    
    # Extract information
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # The content is the JSON string returned by hybrid_search_tool
            try:
                # Attempt to parse JSON
                content = msg.content
                if isinstance(content, str):
                    data = json.loads(content)
                    if isinstance(data, list):
                        retrieved_items.extend(data)
                    elif isinstance(data, dict):
                         retrieved_items.append(data)
            except json.JSONDecodeError:
                # Fallback if something returned text
                pass
                
        if isinstance(msg, AIMessage):
            # Update final response to the latest AI message (ignoring tool_calls if they exist and we want the final answer)
            if not msg.tool_calls:
                final_response = str(msg.content)
    
    return ChatResponse(
        response=final_response,
        retrieved_items=retrieved_items
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
