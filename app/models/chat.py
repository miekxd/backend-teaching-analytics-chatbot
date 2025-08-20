from pydantic import BaseModel
from typing import List, Optional, Dict, Union

class GraphInfo(BaseModel):
    type: str
    reason: str

class ChatRequest(BaseModel):
    file_id: List[int]
    question: str
    top_k: Optional[int] = 3

class UnifiedChatRequest(BaseModel):
    message: str
    file_ids: List[int]  # Required for lesson context
    conversation_history: List[Dict[str, str]] = []  # Optional chat history

class UnifiedResponse(BaseModel):
    response: str
    needs_graph: bool = False
    # Single graph fields (for backward compatibility)
    graph_type: Optional[str] = None
    graph_reason: Optional[str] = None
    # Multiple graphs field (new)
    graph_types: Optional[List[GraphInfo]] = None
    lesson_filter: List[str] = []  # Array of file_ids for lesson filtering
    area_filter: List[str] = []    # Array of teaching area codes for area filtering
