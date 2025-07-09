from pydantic import BaseModel
from typing import List, Optional, Dict

class ChatRequest(BaseModel):
    file_id: List[int]
    question: str
    top_k: Optional[int] = 3

# Enhanced request model for general chat with file context
class EnhancedChatRequest(BaseModel):
    message: str
    file_ids: List[int]  # Required for lesson context
    conversation_history: List[Dict[str, str]] = []  # Optional chat history

class EnhancedChatResponse(BaseModel):
    response: str
    method: str
    flag_specific: float  # Can be 0, 0.5, or 1
    confidence: float
    routing_reason: str
    files_analyzed: int
    has_lesson_context: bool
    suggested_detailed_queries: List[str]

class EnhancedChatErrorResponse(BaseModel):
    response: str
    method: str
    flag_specific: int
    confidence: float
    routing_reason: str
    error: str