from fastapi import APIRouter, HTTPException
from models.request import ObsessionAnalysisRequest, ObsessionAnalysisResponse, ObsessionAnalysis2Request, ObsessionAnalysis2Response, ObsessionAnalysis3Request, ObsessionAnalysis3Response, ObsessionAnalysis4Request, ObsessionAnalysis4Response, ObsessionAnalysis5Request, ObsessionAnalysis5Response, ObsessionAnalysis6Request, ObsessionAnalysis6Response
from services.chatbot_service import ChatbotService
from formatters.obsession_formatter import format_obsession_question
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/obsession", tags=["obsession-analysis"])

# 서비스 인스턴스 생성
chatbot_service = ChatbotService()

@router.post("/analyze", response_model=ObsessionAnalysisResponse)
async def analyze_obsession(request: ObsessionAnalysisRequest):
    """
    사용자의 텍스트를 분석하여 강박 관련 질문과 선택지를 생성합니다.
    """
    try:
        logger.info(f"강박 분석 요청: {request.user_text[:50]}...")
        
        # LLM을 통해 질문과 선택지 생성
        raw_response = chatbot_service.generate_obsession_question(request.user_text)
        
        # 응답 형식 가공
        formatted_response = format_obsession_question(raw_response)
        
        logger.info(f"강박 분석 완료: 질문 생성됨")
        
        return ObsessionAnalysisResponse(
            question=formatted_response["question"],
            choices=formatted_response["choices"],
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"강박 분석 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@router.post("/analyze2", response_model=ObsessionAnalysis2Response)
async def analyze_obsession2(request: ObsessionAnalysis2Request):
    """
    대화 히스토리를 분석하여 강박 행동에 대한 공감적 질문을 생성합니다.
    """
    try:
        logger.info(f"강박 분석2 요청: session_id={request.session_id}")
        
        # LLM을 통해 공감적 질문 생성
        response = chatbot_service.generate_obsession_analysis2_response(request.conversation_history)
        
        logger.info(f"강박 분석2 완료: 응답 생성됨")
        
        return ObsessionAnalysis2Response(
            session_id=request.session_id,
            response=response
        )
        
    except Exception as e:
        logger.error(f"강박 분석2 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@router.post("/analyze3", response_model=ObsessionAnalysis3Response)
async def analyze_obsession3(request: ObsessionAnalysis3Request):
    try:
        logger.info(f"강박 분석3 요청: session_id={request.session_id}")
        
        # LLM을 통해 패턴 요약과 생각 예시 생성
        analysis_result = chatbot_service.generate_obsession_analysis3_response(request.conversation_history)
        
        logger.info(f"강박 분석3 완료: 응답 생성됨")
        
        return ObsessionAnalysis3Response(
            session_id=request.session_id,
            gratitude_message="자세히 말씀해주셔서 고마워요.",
            user_pattern_summary=analysis_result["user_pattern_summary"],
            question="혹시 이런 생각이 자주 떠오르진 않으시나요?",
            thought_examples=analysis_result["thought_examples"]
        )
    except Exception as e:
        logger.error(f"강박 분석3 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@router.post("/analyze4", response_model=ObsessionAnalysis4Response)
async def analyze_obsession4(request: ObsessionAnalysis4Request):
    try:
        logger.info(f"강박 분석4 요청: session_id={request.session_id}")
        
        # LLM을 통해 강박 유형별 맞춤 응답 생성
        analysis_result = chatbot_service.generate_obsession_analysis4_response(request.conversation_history)
        
        logger.info(f"강박 분석4 완료: 응답 생성됨 (카테고리: {analysis_result.get('obsession_type', 'unknown')})")
        
        return ObsessionAnalysis4Response(
            session_id=request.session_id,
            user_pattern_summary=analysis_result["user_pattern_summary"],
            category_message=analysis_result["category_message"],
            encouragement=analysis_result["encouragement"]
        )
    except Exception as e:
        logger.error(f"강박 분석4 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")


@router.post("/analyze5", response_model=ObsessionAnalysis5Response)
async def analyze_obsession5(request: ObsessionAnalysis5Request):
    """
    대화 히스토리를 바탕으로 280자 이내의 자각을 돕는 질문을 생성합니다.
    """
    try:
        logger.info(f"강박 분석5 요청: session_id={request.session_id}")
        response = chatbot_service.generate_obsession_analysis5_response(request.conversation_history)
        logger.info("강박 분석5 완료: 응답 생성됨")
        return ObsessionAnalysis5Response(
            session_id=request.session_id,
            response=response
        )
    except Exception as e:
        logger.error(f"강박 분석5 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@router.get("/health")
async def health_check():
    """
    서비스 상태 확인
    """
    return {"status": "healthy", "service": "obsession-analysis"} 

@router.post("/analyze6", response_model=ObsessionAnalysis6Response)
async def analyze_obsession6(request: ObsessionAnalysis6Request):
    """
    LLM으로 공감적 도입부를 생성하고, 불안 위계로 전환하게끔 함.
    """
    try:
        logger.info(f"강박 분석6 요청: session_id={request.session_id}")
        response = chatbot_service.generate_obsession_analysis6_response(request.conversation_history)
        logger.info("강박 분석6 완료: 응답 생성됨")
        return ObsessionAnalysis6Response(
            session_id=request.session_id,
            response=response
        )
    except Exception as e:
        logger.error(f"강박 분석6 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")