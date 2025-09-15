from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    follow_up_question: Optional[str] = None
    choices: Optional[List[str]] = None

class ObsessionAnalysisRequest(BaseModel):
    user_text: str
    session_id: Optional[str] = None

class ObsessionAnalysisResponse(BaseModel):
    question: str
    choices: List[str]
    session_id: str

class ObsessionAnalysis2Request(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class ObsessionAnalysis2Response(BaseModel):
    session_id: str
    response: str

class ObsessionAnalysis3Request(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class ObsessionAnalysis3Response(BaseModel):
    session_id: str
    gratitude_message: str  #고정형 감사 인사
    user_pattern_summary: str  #사용자 패턴 요약
    question: str  #고정형 질문
    thought_examples: List[str]  #생각 예시 3개

class ObsessionAnalysis4Request(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class ObsessionAnalysis4Response(BaseModel):
    session_id: str
    user_pattern_summary: str
    category_message: str
    encouragement: str

class ObsessionAnalysis5Request(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class ObsessionAnalysis5Response(BaseModel):
    session_id: str
    response: str

class ObsessionAnalysis6Request(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class ObsessionAnalysis6Response(BaseModel):
    session_id: str
    response: str