from fastapi import FastAPI, HTTPException,Depends,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from typing import  AsyncGenerator
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import json

# LangChain / LangGraph Imports
from helper.workflow_run import app_graph
from langchain_core.messages import  HumanMessage, AIMessage, SystemMessage
from route.search import search_router
from type.chat import ChatRequest
from fastapi_throttle import RateLimiter


# Load environment variables
loaded = load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)
if not loaded:
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)



LIMIT_TIMES = 10
LIMIT_SECONDS = 60
# --- FastAPI Application ---
api_limit = RateLimiter(times=LIMIT_TIMES, seconds=LIMIT_SECONDS)
app = FastAPI(title="Agentic RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse,dependencies=[Depends(api_limit)])
async def read_root():
    try:
        with open("chat_ui.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "<h1>Chat UI not found</h1>"
    
    
    
app.include_router(search_router,prefix='/search',tags=['search'],dependencies=[Depends(api_limit)])



@app.post("/api/chat/stream", dependencies=[Depends(api_limit)])
async def chat_stream(request: ChatRequest):
    """SSE endpoint"""
    MAX_MESSAGES = 50
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Limit request.messages to the last 4000 elements if exceeded
            messages = request.messages
            if len(messages) > MAX_MESSAGES:
                messages = messages[-MAX_MESSAGES:]

            lc_messages = []
            for msg in messages:
                if msg.role == "user":
                    lc_messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    lc_messages.append(AIMessage(content=msg.content))
                elif msg.role == "system":
                    lc_messages.append(SystemMessage(content=msg.content))

            inputs = {"messages": lc_messages}

            tokens_yielded = False
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
                            tokens_yielded = True
                            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"

            if not tokens_yielded:
                fallback_data = {"type": "token", "data": 'I cannot answer this question. My knowledge base is specific to Tibetan Buddhism and does not contain information about a concept of "God" in the way it might be understood in other religions.'}
                yield f"data: {json.dumps(fallback_data, ensure_ascii=False)}\n\n"

            event_data = {"type": "done", "data": {}}
            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_data = {"type": "error", "data": {"message": str(e)}}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@app.get("/health",dependencies=[Depends(api_limit)])
async def health():
    
    # Setup API Keys
    if "GOOGLE_API_KEY" not in os.environ:
        if os.getenv("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    MILVUS_URI = os.getenv("MILVUS_URI")
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
    MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "test_kangyur_tengyur")
    if not all([MILVUS_URI, MILVUS_TOKEN,MILVUS_COLLECTION_NAME, GEMINI_API_KEY]):
        raise HTTPException(status_code=500, detail="Missing environment variables for Milvus or Gemini.")
    return {"status": "healthy"}


if __name__ == "__main__":
    is_dev = os.getenv('ENV') == 'development'
    port=int(os.getenv('PORT',8000))
    import uvicorn
    uvicorn.run("main:app", port=port,reload=is_dev)

