from fastapi import FastAPI
from app.api.v1.endpoints import router as v1_router

app = FastAPI()

@app.get("/")
def root():
    return {"message": "RAG Chatbot API is running"}

app.include_router(v1_router, prefix="/api/v1")
