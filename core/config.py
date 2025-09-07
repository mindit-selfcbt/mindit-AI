import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API 설정
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Mindit AI Chatbot API"
    
    # LLM 설정
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
    
    # FAISS 설정
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    
    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
