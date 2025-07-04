from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    file_id: List[int]
    question: str
    top_k: Optional[int] = 3
