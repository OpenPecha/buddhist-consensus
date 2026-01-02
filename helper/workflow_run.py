from helper.config import LOW_PRIORITY_MODEL
from helper.tool import hybrid_search_tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from type.chat import Grade, State
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langgraph.prebuilt import ToolNode, tools_condition


llm = ChatGoogleGenerativeAI(model=LOW_PRIORITY_MODEL, temperature=0)
structured_llm_grader = llm.with_structured_output(Grade)


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