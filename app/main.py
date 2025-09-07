from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from core.logging import setup_logging
from app.obsession_router import router as obsession_router

# 로깅 설정
setup_logging()

# FastAPI 앱 생성
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="AI 기반 상담 챗봇 API"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(obsession_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "message": "Mindit AI Chatbot API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 