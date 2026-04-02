import uvicorn
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
from langchain_core.messages import  HumanMessage, AIMessage, SystemMessage, ToolMessage
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
async def chat_stream(request: Request, chat_request: ChatRequest):
    """SSE endpoint with proper isolation and completion signaling"""
    import uuid
    
    MAX_MESSAGES = 30
    
    async def event_generator() -> AsyncGenerator[str, None]:
        # Generate unique run_id for request isolation
        run_id = str(uuid.uuid4())
        has_content = False
        
        # Repetition detection state
        REPETITION_THRESHOLD = 4
        MIN_CHUNK_LENGTH = 10  # Only track chunks with meaningful content
        recent_chunks: list[str] = []
        repetition_detected = False
        
        def check_repetition(new_chunk: str) -> bool:
            """Detect if the stream is stuck in a repetition loop."""
            if len(new_chunk.strip()) < MIN_CHUNK_LENGTH:
                return False
            
            recent_chunks.append(new_chunk)
            # Keep only the last N chunks for comparison
            if len(recent_chunks) > REPETITION_THRESHOLD * 2:
                recent_chunks.pop(0)
            
            if len(recent_chunks) < REPETITION_THRESHOLD:
                return False
            
            # Check if last N chunks are identical
            last_chunks = recent_chunks[-REPETITION_THRESHOLD:]
            if all(chunk == last_chunks[0] for chunk in last_chunks):
                return True
            
            # Check for pattern repetition in accumulated text
            accumulated = "".join(recent_chunks[-REPETITION_THRESHOLD:])
            chunk_len = len(new_chunk.strip())
            if chunk_len > 0 and len(accumulated) >= chunk_len * REPETITION_THRESHOLD:
                pattern = new_chunk.strip()
                if accumulated.count(pattern) >= REPETITION_THRESHOLD:
                    return True
            
            return False
        
        try:
            messages = chat_request.messages
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
            config = {"configurable": {"thread_id": run_id}}
            
            async for event in app_graph.astream_events(inputs, config=config, version="v2"):
                # Check if client disconnected or repetition detected
                if await request.is_disconnected() or repetition_detected:
                    break
                    
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
                                        has_content = True
                                        event_data = {
                                            "type": "search_results",
                                            "data": data,
                                            "queries": queries
                                        }
                                        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                    except json.JSONDecodeError as e:
                        print(f"[{run_id}] JSON decode error in tool output: {e}")
                    except Exception as e:
                        print(f"[{run_id}] Error processing tool output: {e}")

                if kind == "on_chat_model_stream":
                    node_name = event.get("metadata", {}).get("langgraph_node")
                    if node_name in ["generate_answer", "generate_query_or_respond"]:
                        chunk = event["data"].get("chunk")
                        if chunk and chunk.content:
                            # Check for repetition before yielding
                            if check_repetition(chunk.content):
                                print(f"[{run_id}] Repetition detected, closing stream")
                                repetition_detected = True
                                break
                            
                            has_content = True
                            event_data = {"type": "token", "data": chunk.content}
                            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            
            # Send completion signal
            if not has_content:
                yield f"data: {json.dumps({'type': 'token', 'data': 'No results found.'}, ensure_ascii=False)}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            print(f"[{run_id}] Stream error: {e}")
            error_data = {"type": "error", "data": {"message": str(e)}}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream"
        }
    )


@app.post("/api/chat", dependencies=[Depends(api_limit)])
async def chat(request: ChatRequest):
    """Non-streaming endpoint that returns complete response"""
    MAX_MESSAGES = 50
    try:
        # Limit request.messages to the last 50 elements if exceeded
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

        # Invoke the graph to get final state
        final_state = app_graph.invoke(inputs)
        final_messages = final_state.get("messages", [])

        # Extract answer from last AIMessage
        answer = ""
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage):
                answer = msg.content
                break

        # Extract search results from ToolMessage(s) with hybrid_search_tool
        search_results = []
        queries = {}
        for msg in final_messages:
            if isinstance(msg, ToolMessage):
                try:
                    content = msg.content
                    if isinstance(content, str):
                        parsed_output = json.loads(content)
                        if isinstance(parsed_output, dict) and "results" in parsed_output:
                            data = parsed_output["results"]
                            queries = parsed_output.get("queries", {})
                            if isinstance(data, list):
                                search_results = data
                                # Use the last tool message result
                                break
                except Exception:
                    pass
        
        response_data = {
            "type": "done",
            "data": {
                "answer": answer,
                "search_results": search_results,
                "queries": queries
            }
        }
        
        return response_data
        
    except Exception as e:
        error_data = {
            "type": "error",
            "data": {
                "message": str(e),
                "answer": "",
                "search_results": [],
                "queries": {}
            }
        }
        return error_data


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
    port=int(os.getenv('PORT',10000))
    uvicorn.run("main:app", port=port,reload=True)

