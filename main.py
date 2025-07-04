from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import router as v1_router

app = FastAPI()

# CORS setup
origins = [
    "http://localhost:3000",  # Local frontend
    # "https://your-frontend-domain.com",  # Add your production frontend URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "RAG Chatbot API is running"}

app.include_router(v1_router, prefix="/api/v1")
