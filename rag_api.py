from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import json
import asyncio

# LangChain / LangGraph Imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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
loaded = load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)
if not loaded:
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

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
    collection_name=MILVUS_COLLECTION_NAME
)

# Raw Gemini client for embeddings and expansion
genai_client = genai.Client(api_key=GEMINI_API_KEY)
doc_cfg = EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768)

# Helper: Get Embedding
def get_embedding(text: str) -> List[float]:
    """Generate embedding for the given text using Gemini."""
    resp = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=doc_cfg
    )
    return [i.values for i in resp.embeddings][0]

# Helper: Expand Query
def generate_expanded_queries(query: str) -> Dict[str, str]:
    """
    Generates 4 expanded queries: 
    2 for BM25 (Tibetan/English keywords), 2 for Semantic (Tibetan/English questions).
    """
    prompt = f"""
    You are an expert search query optimizer for a Tibetan Buddhist database.
    The user query is: "{query}"
    
    Generate 4 varied search queries to maximize retrieval recall:
    1. "tibetan_bm25": Key Tibetan terms/phrases (for keyword match).
    2. "english_bm25": Key English terms/phrases (for keyword match).
    3. "tibetan_semantic": A natural language question/statement in Tibetan capturing the meaning.
    4. "english_semantic": A natural language question/statement in English capturing the meaning.
    
    Return ONLY a JSON object with these 4 keys.
    """
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={'response_mime_type': 'application/json'}
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error expanding query: {e}")
        # Fallback: just use original query for all if expansion fails
        return {
            "tibetan_bm25": query,
            "english_bm25": query,
            "tibetan_semantic": query,
            "english_semantic": query
        }

# --- Custom Hybrid Search Tool with Expansion ---
@tool
def hybrid_search_tool(query: str):
    """
    Searches the Tibetan knowledge base. 
    Automatically expands the query into Tibetan and English variations 
    (keywords and semantic) to improve coverage.
    Returns relevant text segments with metadata.
    """
    limit = 15 # Higher limit since we dedup later
    
    # 1. Generate Expanded Queries
    expanded = generate_expanded_queries(query)
    print(f"DEBUG: Expanded queries: {expanded}")
    
    queries_to_run = list(expanded.values())
    # Ensure original query is included if not covered? 
    # The expanded ones should cover it.
    
    all_results = []
    
    # 2. Run searches for each variation
    for q_text in queries_to_run:
        if not q_text or not q_text.strip():
            continue
            
        try:
            # Generate embedding
            q_emb = get_embedding(q_text)
            
            # BM25 Req
            req_bm25 = AnnSearchRequest(
                data=[q_text],
                anns_field="sparce_vector",
                param={},
                limit=limit
            )
            
            # Dense Req
            req_dense = AnnSearchRequest(
                data=[q_emb],
                anns_field="dense_vector",
                param={"drop_ratio_search": 0.2},
                limit=limit
            )
            
            # Hybrid Search
            res = milvus_client.hybrid_search(
                collection_name=MILVUS_COLLECTION_NAME,
                reqs=[req_bm25, req_dense],
                ranker=RRFRanker(),
                limit=limit,
                output_fields=["text", "title", "id"]
            )
            
            # Collect hits
            for hits in res:
                for hit in hits:
                    all_results.append(hit)
                    
        except Exception as e:
            print(f"Error searching for '{q_text}': {e}")
            continue

    # 3. Deduplicate and Format
    unique_items = {}
    for hit in all_results:
        entity = hit.get("entity", {})
        # ID might be in 'id' or 'entity.id'
        item_id = str(hit.get("id") or entity.get("id") or "")
        
        if item_id and item_id not in unique_items:
            unique_items[item_id] = {
                "id": item_id,
                "title": entity.get("title", "Unknown"),
                "text": entity.get("text", ""),
                "score": hit.get("score", 0),
                "distance": hit.get("distance", 0)
            }
            
    # Sort by score/distance? RRF scores are relative.
    # Let's just return them list values.
    final_items = list(unique_items.values())
    
    # Limit total return size
    final_items = final_items[:20] 
    
    # Return results + queries
    output = {
        "results": final_items,
        "queries": expanded
    }
    
    return json.dumps(output, ensure_ascii=False)


# --- Graph Definitions ---

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Initialize Model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)

def generate_query_or_respond(state: State):
    """Decide whether to retrieve or respond directly."""
    # Force system prompt
    sys_msg = SystemMessage(content="""You are a helpful assistant. You have access to a 'hybrid_search_tool' that searches a Tibetan Buddhist knowledge base. 
    
    You MUST use this tool to answer questions about Buddhism, happiness, or life advice based on the texts. 
    - Do not answer from your own knowledge if it can be found in the texts.
    - ACCEPT queries in ANY language (especially Tibetan).
    - NEVER ask the user to translate their query. If the query is in Tibetan, simply use the tool with the Tibetan query.
    """)
    messages = [sys_msg] + state["messages"]
    
    model_with_tools = llm.bind_tools([hybrid_search_tool])
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

def generate_answer(state: State):
    """Generate answer using retrieved context."""
    messages = state["messages"]
    
    system_prompt = SystemMessage(content="""
    You are a helpful, friendly academic assistant for Buddhist studies, acting as a supportive friend.
    
    LANGUAGE INSTRUCTION:
    1. Detect the language of the user's last message (Tibetan or English).
    2. You MUST answer in the SAME language as the user's query.
    3. If the user explicitly requests a specific language, honor that request.
    4. Do NOT switch languages unless asked. If the query is in Tibetan, the answer MUST be in Tibetan. If the query is in English, the answer MUST be in English.
 
    CRITICAL CITATION RULES:
    1. Every single sentence or claim you make based on the text must be immediately followed by a citation.
    2. Use the EXACT format [ID] for citations. Do NOT use the title in the citation bracket, ONLY the ID.
       Example: "Emptiness is form [2pIapXDirmQdLVFLptm5r]."
    3. If the retrieved text is in Tibetan, quote the relevant Tibetan phrase in the answer where appropriate.
    4. If you cannot find the answer in the context, state that you don't know.
    5. Mention the book title too where relevant.
    6. TIBETAN CITATION PLACEMENT: If the sentence ends with a shad (།), place the citation AFTER the shad.
       Example: ...བཞུགས་སོ། [ID] (Correct)
       Example: ...བཞུགས་སོ [ID]། (Incorrect)
    FRIENDLY PERSONA:
    - Be ༷warm, encouraging, and supportive.
    - Use a conversational tone while maintaining academic rigor with citations.
    """)
    
    response_content = ""
    for chunk in llm.stream([system_prompt] + messages):
        response_content += chunk.content
        
    return {"messages": [AIMessage(content=response_content)]}

def rewrite_question(state: State):
    """Transform the query to produce a better question."""
    messages = state["messages"]
    # Find last human message
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    question = last_human.content if last_human else messages[-1].content
    
    msg = [
        HumanMessage(
            content=f"""Look at the input and try to reason about the underlying semantic intent / meaning. 
            Here is the initial question:
            \n{question}\n
            Formulate an improved question for a search engine to find Tibetan Buddhist texts:"""
        )
    ]
    response = llm.invoke(msg)
    return {"messages": [HumanMessage(content=response.content)]}

class Grade(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(Grade)

def grade_documents(state: State):
    """Determines whether the retrieved documents are relevant."""
    messages = state["messages"]
    
    rewrite_count = len([m for m in messages if isinstance(m, HumanMessage) and "Look at the input" in str(m.content)])
    if rewrite_count > 2:
         return "generate_answer"

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    if not tool_messages:
        return "generate_answer"
        
    latest_tool_msg = tool_messages[-1]
    docs_text = str(latest_tool_msg.content)
    
    if not docs_text or docs_text == "[]" or "Error" in docs_text:
         if rewrite_count > 0:
             return "generate_answer"
         return "rewrite_question"

    prompt = f"""You are a grader assessing relevance of retrieved Tibetan texts to a user question. \n 
    Here is the retrieved document content (JSON structure): \n\n {docs_text}... \n\n
    
    If the document content seems even remotely related or helpful, grade it as 'yes'.
    Give a binary score༷ 'yes' or 'no'."""
    
    try:
        scored_result = structured_llm_grader.invoke(prompt)
        if scored_result.binary_score == "yes":
            return "generate_answer"
    except:
        pass
        
    return "rewrite_question"

# Build Graph
workflow = StateGraph(State)
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([hybrid_search_tool]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges("generate_query_or_respond", tools_condition, {"tools": "retrieve", END: END})
workflow.add_conditional_edges("retrieve", grade_documents, {"generate_answer": "generate_answer", "rewrite_question": "rewrite_question"})
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
    try:
        with open("chat_ui.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "<h1>Chat UI not found</h1>"

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE endpoint"""
    async def event_generator() -> AsyncGenerator[str, None]:
        lc_messages = []
        for msg in request.messages:
            if msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
        
        inputs = {"messages": lc_messages}
        
        async for event in app_graph.astream_events(inputs, version="v1"):
            kind = event["event"]
            
            if kind == "on_tool_end" and event["name"] == "hybrid_search_tool":
                try:
                    content = event["data"].get("output")
                    if content:
                        if hasattr(content, "content"):
                             content = content.content
                        
                        if isinstance(content, str):
                            parsed_output = json.loads(content)
                            if isinstance(parsed_output, dict) and "results" in parsed_output:
                                data = parsed_output["results"]
                                queries = parsed_output.get("queries", {})
                                if isinstance(data, list):
                                    event_data = {
                                        "type": "search_results", 
                                        "data": data,
                                        "queries": queries
                                    }
                                    yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                except Exception:
                    pass
            
            elif kind == "on_chat_model_stream":
                node_name = event.get("metadata", {}).get("langgraph_node")
                if node_name in ["generate_answer", "generate_query_or_respond"]:
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        event_data = {"type": "token", "data": chunk.content}
                        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
        
        event_data = {"type": "done", "data": {}}
        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

