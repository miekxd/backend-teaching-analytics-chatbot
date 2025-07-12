from pydantic import BaseModel
from typing import List, Optional, Dict

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
