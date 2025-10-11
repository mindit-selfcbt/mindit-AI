from fastapi import APIRouter, HTTPException
from models.request import (
    ResponsePreventionRequest, ResponsePreventionResponse, 
    ResponsePreventionAnalysis2Request, ResponsePreventionAnalysis2Response, 
    ResponsePreventionAnalysis3Request, ResponsePreventionAnalysis3Response,
    ResponsePreventionAnalysis3_2Request, ResponsePreventionAnalysis3_2Response,
    ResponsePreventionAnalysis3_3Request, ResponsePreventionAnalysis3_3Response,
    ResponsePreventionAnalysis4_1Request, ResponsePreventionAnalysis4_1Response,
    ResponsePreventionAnalysis4_2Request, ResponsePreventionAnalysis4_2Response,
    ResponsePreventionReportRequest, ResponsePreventionReportResponse,
    CognitiveError
)
from services.response_prevention_service import ResponsePreventionService
from formatters.response_prevention_formatter import format_response_prevention_analysis
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/response-prevention", tags=["response-prevention"])

response_prevention_service = ResponsePreventionService()

@router.post("/analyze", response_model=ResponsePreventionResponse)
async def analyze_response_prevention(request: ResponsePreventionRequest):
    """
    반응방지 완료 후 분석 및 맞춤형 응답을 생성합니다.
    """
    try:
        logger.info(f"반응방지 분석 요청: session_id={request.session_id}")
        
        raw_response = response_prevention_service.generate_response_prevention_analysis(
            obsession_situation=request.obsession_situation,
            anxiety_level_start=request.anxiety_level_start,
            anxiety_level_end=request.anxiety_level_end,
            obsession_thought=request.obsession_thought
        )
        
        formatted_response = format_response_prevention_analysis(raw_response)
        
        logger.info(f"반응방지 분석 완료: session_id={request.session_id}")
        
        return ResponsePreventionResponse(
            session_id=request.session_id,
            obsession_type=formatted_response["obsession_type"],
            content=formatted_response["content"]
        )
        
    except Exception as e:
        logger.error(f"반응방지 분석 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@router.post("/analyze2", response_model=ResponsePreventionAnalysis2Response)
async def analyze_response_prevention2(request: ResponsePreventionAnalysis2Request):
    """
    대화 히스토리를 기반으로 RAG를 사용하여 인지적 오류를 분석합니다.
    """
    try:
        logger.info(f"반응방지 분석2 요청: session_id={request.session_id}")
        
        # RAG를 사용하여 인지적 오류 분석
        cognitive_errors_data = response_prevention_service.generate_response_prevention_analysis2(
            session_id=request.session_id,
            conversation_history=request.conversation_history
        )
        
        # CognitiveError 객체로 변환
        cognitive_errors = []
        for error_data in cognitive_errors_data:
            cognitive_errors.append(CognitiveError(
                error_type=error_data.get("error_type", ""),
                description=error_data.get("description", "")
            ))
        
        logger.info(f"반응방지 분석2 완료: session_id={request.session_id}, 발견된 인지적 오류: {len(cognitive_errors)}개")
        
        return ResponsePreventionAnalysis2Response(
            session_id=request.session_id,
            cognitive_errors=cognitive_errors,
            reflection_question="이제 이 '그릇된 믿음'을 현실적으로 다시 생각해 볼까요?",
            reality_check_question="이 생각이 얼마나 자주, 얼마나 강하게 나타났고 실제로 어떤 일이 벌어졌나요? 비슷한 불안을 느낀 적이 자주 있었는데, 실제로 큰일이 났던 적이 있었을까요?"
        )
        
    except Exception as e:
        logger.error(f"반응방지 분석2 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@router.post("/analyze3", response_model=ResponsePreventionAnalysis3Response)
async def analyze_response_prevention3(request: ResponsePreventionAnalysis3Request):
    """
    대화 히스토리를 기반으로 공감 메시지와 통찰 메시지를 생성합니다.
    """
    try:
        logger.info(f"반응방지 분석3 요청: session_id={request.session_id}")
        
        # 공감 메시지와 통찰 메시지 생성
        analysis_result = response_prevention_service.generate_response_prevention_analysis3(
            conversation_history=request.conversation_history
        )
        
        logger.info(f"반응방지 분석3 완료: session_id={request.session_id}")
        
        return ResponsePreventionAnalysis3Response(
            session_id=request.session_id,
            empathy_message=analysis_result["empathy_message"],
            insight_message=analysis_result["insight_message"],
            encouragement_message="불안을 없애는 게 아니라, 다르게 반응하는 연습을 계속하고 있다는 걸 기억해주세요. 그 자체로 지금 충분히 잘하고 있어요💪",
            choice_question="지금 나눈 대화가 조금이라도 마음을 정리하는 데 도움이 되었기를 바라요. 오늘 대화는 여기서 마무리해볼까요, 아니면 조금 더 이야기 나눌까요?"
        )
        
    except Exception as e:
        logger.error(f"반응방지 분석3 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@router.post("/analyze3-2", response_model=ResponsePreventionAnalysis3_2Response)
async def analyze_response_prevention3_2(request: ResponsePreventionAnalysis3_2Request):
    """
    오염강박용 analyze3-2: 구체적인 근거 질문을 생성합니다.
    """
    try:
        logger.info(f"반응방지 분석3-2 요청: session_id={request.session_id}")
        
        # 구체적인 근거 질문 생성
        response = response_prevention_service.generate_response_prevention_analysis3_2(
            conversation_history=request.conversation_history
        )
        
        logger.info(f"반응방지 분석3-2 완료: session_id={request.session_id}")
        
        return ResponsePreventionAnalysis3_2Response(
            session_id=request.session_id,
            response=response
        )
        
    except Exception as e:
        logger.error(f"반응방지 분석3-2 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@router.post("/analyze3-3", response_model=ResponsePreventionAnalysis3_3Response)
async def analyze_response_prevention3_3(request: ResponsePreventionAnalysis3_3Request):
    """
    오염강박용 analyze3-3: 인지적 재구성과 격려 메시지를 생성합니다.
    """
    try:
        logger.info(f"반응방지 분석3-3 요청: session_id={request.session_id}")
        
        # 인지적 재구성과 격려 메시지 생성
        response_data = response_prevention_service.generate_response_prevention_analysis3_3(
            session_id=request.session_id,
            conversation_history=request.conversation_history
        )
        
        logger.info(f"반응방지 분석3-3 완료: session_id={request.session_id}")
        
        return ResponsePreventionAnalysis3_3Response(
            session_id=request.session_id,
            intro_message=response_data["intro_message"],
            cognitive_restructuring=response_data["cognitive_restructuring"],
            encouragement_message=response_data["encouragement_message"],
            insight_message=response_data["insight_message"],
            final_encouragement=response_data["final_encouragement"],
            choice_question=response_data["choice_question"]
        )
        
    except Exception as e:
        logger.error(f"반응방지 분석3-3 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@router.post("/analyze4-1", response_model=ResponsePreventionAnalysis4_1Response)
async def analyze_response_prevention4_1(request: ResponsePreventionAnalysis4_1Request):
    """
    더 이야기 나누기를 선택했을 때의 응답을 생성합니다.
    """
    try:
        logger.info(f"반응방지 분석4-1 요청: session_id={request.session_id}")
        
        # 더 이야기 나누기 응답 생성
        response = response_prevention_service.generate_response_prevention_analysis4_1(
            conversation_history=request.conversation_history
        )
        
        logger.info(f"반응방지 분석4-1 완료: session_id={request.session_id}")
        
        return ResponsePreventionAnalysis4_1Response(
            session_id=request.session_id,
            response=response
        )
        
    except Exception as e:
        logger.error(f"반응방지 분석4-1 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@router.post("/analyze4-2", response_model=ResponsePreventionAnalysis4_2Response)
async def analyze_response_prevention4_2(request: ResponsePreventionAnalysis4_2Request):
    """
    마무리를 선택했을 때의 고정 메시지를 반환합니다.
    """
    try:
        logger.info(f"반응방지 분석4-2 요청: session_id={request.session_id}")
        
        # 고정 마무리 메시지
        completion_message = response_prevention_service.generate_response_prevention_analysis4_2()
        
        logger.info(f"반응방지 분석4-2 완료: session_id={request.session_id}")
        
        return ResponsePreventionAnalysis4_2Response(
            session_id=request.session_id,
            completion_message=completion_message
        )
        
    except Exception as e:
        logger.error(f"반응방지 분석4-2 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@router.post("/report", response_model=ResponsePreventionReportResponse)
async def generate_response_prevention_report(request: ResponsePreventionReportRequest):
    """
    전체 대화 내역을 기반으로 종합 리포트를 생성합니다.
    """
    try:
        logger.info(f"반응방지 리포트 요청: session_id={request.session_id}")
        
        # 종합 리포트 생성
        report_data = response_prevention_service.generate_response_prevention_report(
            conversation_history=request.conversation_history
        )
        
        logger.info(f"반응방지 리포트 완료: session_id={request.session_id}")
        
        return ResponsePreventionReportResponse(
            session_id=request.session_id,
            consultation_date=report_data["consultation_date"],
            consultation_time=report_data["consultation_time"],
            obsession_type=report_data["obsession_type"],
            obsession_situation=report_data["obsession_situation"],
            intrusive_thoughts=report_data["intrusive_thoughts"],
            false_beliefs=report_data["false_beliefs"],
            cognitive_errors=report_data["cognitive_errors"],
            realistic_thinking=report_data["realistic_thinking"]
        )
        
    except Exception as e:
        logger.error(f"반응방지 리포트 생성 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@router.get("/health")
async def health_check():
    """
    서비스 상태 확인
    """
    return {"status": "healthy", "service": "response-prevention"}
