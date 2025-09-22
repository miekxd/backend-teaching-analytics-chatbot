import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str = "2024-12-01-preview"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str
    AZURE_OPENAI_DEPLOYMENT_RAG: str
    AZURE_OPENAI_DEPLOYMENT_GENERAL: str

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra environment variables

settings = Settings()
