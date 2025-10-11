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


class ObsessionAnalysis7Request(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class ObsessionAnalysis7Response(BaseModel):
    session_id: str
    intro_message: str
    question: str
    situations: List[str]

class ObsessionAnalysis8Request(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class ObsessionAnalysis8Response(BaseModel):
    session_id: str
    response: str

class ObsessionAnalysis9Request(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class AnxietyHierarchyItem(BaseModel):
    order: int
    situation: str
    score: int

class ObsessionAnalysis9Response(BaseModel):
    session_id: str
    intro_message: str
    anxiety_hierarchy: List[AnxietyHierarchyItem]
    practice_message: str
    example_message: str
    support_message: str

class ObsessionReportRequest(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class AnxietyHierarchyItem(BaseModel):
    situation: str
    score: int

class ObsessionReportResponse(BaseModel):
    session_id: str
    obsession_type: str
    anxiety_thoughts: str
    compulsive_behaviors: str
    anxiety_hierarchy: List[AnxietyHierarchyItem]

class WeekInfo(BaseModel):
    week: int
    start_date: str
    end_date: str

class TrainingItem(BaseModel):
    week: int
    training_name: str
    goal: str
    content: str

class TrainingExplanation(BaseModel):
    title: str
    description: str
    when: str
    instruction: str

class ObsessionReport2Request(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class ObsessionReport2Response(BaseModel):
    session_id: str
    obsession_type: str
    training_explanation: Dict[str, TrainingExplanation]
    weeks: List[WeekInfo]
    training_plan: List[TrainingItem]

class ResponsePreventionRequest(BaseModel):
    obsession_situation: str
    anxiety_level_start: int
    anxiety_level_end: int
    obsession_thought: str
    session_id: str

class ResponsePreventionContent(BaseModel):
    encouragement_message: str
    cognitive_error_explanation: str
    personalized_situation: str
    emotion_question: str  # emotion_question과 emotion_examples 통합

class ResponsePreventionResponse(BaseModel):
    session_id: str
    obsession_type: str
    content: ResponsePreventionContent

class CognitiveError(BaseModel):
    error_type: str
    description: str

class ResponsePreventionAnalysis2Request(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class ResponsePreventionAnalysis2Response(BaseModel):
    session_id: str
    cognitive_errors: List[CognitiveError]
    reflection_question: str
    reality_check_question: str

class ResponsePreventionAnalysis3Request(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class ResponsePreventionAnalysis3Response(BaseModel):
    session_id: str
    empathy_message: str
    insight_message: str
    encouragement_message: str
    choice_question: str

class ResponsePreventionAnalysis3_2Request(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class ResponsePreventionAnalysis3_2Response(BaseModel):
    session_id: str
    response: str

class ResponsePreventionAnalysis3_3Request(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class CognitiveRestructuring(BaseModel):
    old_thought: str
    new_perspective: str

class ResponsePreventionAnalysis3_3Response(BaseModel):
    session_id: str
    intro_message: str
    cognitive_restructuring: CognitiveRestructuring
    encouragement_message: str
    insight_message: str
    final_encouragement: str
    choice_question: str

class ResponsePreventionAnalysis4_1Request(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class ResponsePreventionAnalysis4_1Response(BaseModel):
    session_id: str
    response: str

class ResponsePreventionAnalysis4_2Request(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class ResponsePreventionAnalysis4_2Response(BaseModel):
    session_id: str
    completion_message: str

class ResponsePreventionReportRequest(BaseModel):
    conversation_history: List[Dict[str, Any]]
    session_id: str

class CognitiveErrorReport(BaseModel):
    error_type: str
    description: str

class ResponsePreventionReportResponse(BaseModel):
    session_id: str
    consultation_date: str  # 00월 00일 0요일
    consultation_time: str  # 오전 00시 00분
    obsession_type: str
    obsession_situation: str
    intrusive_thoughts: str
    false_beliefs: str
    cognitive_errors: List[CognitiveErrorReport]
    realistic_thinking: str