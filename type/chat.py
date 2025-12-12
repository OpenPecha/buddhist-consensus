from pydantic import BaseModel, Field
from typing import Annotated, List, Dict, Any, Optional, TypedDict, Union, AsyncGenerator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str
    retrieved_items: List[Dict[str, Any]]
    
class Grade(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")
    
    
# --- Graph Definitions ---

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]