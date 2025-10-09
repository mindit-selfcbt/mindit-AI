from fastapi import APIRouter, HTTPException
from models.request import ObsessionAnalysisRequest, ObsessionAnalysisResponse, ObsessionAnalysis2Request, ObsessionAnalysis2Response, ObsessionAnalysis3Request, ObsessionAnalysis3Response, ObsessionAnalysis4Request, ObsessionAnalysis4Response, ObsessionAnalysis5Request, ObsessionAnalysis5Response, ObsessionAnalysis6Request, ObsessionAnalysis6Response, ObsessionAnalysis7Request, ObsessionAnalysis7Response, ObsessionAnalysis8Request, ObsessionAnalysis8Response, ObsessionAnalysis9Request, ObsessionAnalysis9Response, ObsessionReportRequest, ObsessionReportResponse, AnxietyHierarchyItem, ObsessionReport2Request, ObsessionReport2Response, WeekInfo, TrainingItem, TrainingExplanation
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
        analysis_result = chatbot_service.generate_obsession_analysis4_response(request.conversation_history, request.session_id)
        
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

@router.post("/analyze7", response_model=ObsessionAnalysis7Response)
async def analyze_obsession7(request: ObsessionAnalysis7Request):
    """
    이전 대화내역을 받아서 불안 상황과 손 씻기 행동을 구체적으로 정리하고 7가지 예시를 제공합니다.
    """
    try:
        logger.info(f"강박 분석7 요청: session_id={request.session_id}")
        analysis_result = chatbot_service.generate_obsession_analysis7_response(request.conversation_history)
        logger.info("강박 분석7 완료: 응답 생성됨")
        return ObsessionAnalysis7Response(
            session_id=request.session_id,
            intro_message=analysis_result["intro_message"],
            question=analysis_result["question"],
            situations=analysis_result["situations"]
        )
    except Exception as e:
        logger.error(f"강박 분석7 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@router.post("/analyze8", response_model=ObsessionAnalysis8Response)
async def analyze_obsession8(request: ObsessionAnalysis8Request):
    """
    사용자가 상황별 불안 점수를 0-100점으로 매기도록 하는 메시지를 생성합니다.
    """
    try:
        logger.info(f"강박 분석8 요청: session_id={request.session_id}")
        response = chatbot_service.generate_obsession_analysis8_response(request.conversation_history)
        logger.info("강박 분석8 완료: 응답 생성됨")
        return ObsessionAnalysis8Response(
            session_id=request.session_id,
            response=response
        )
    except Exception as e:
        logger.error(f"강박 분석8 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@router.post("/analyze9", response_model=ObsessionAnalysis9Response)
async def analyze_obsession9(request: ObsessionAnalysis9Request):
    """
    점수 데이터를 바탕으로 불안 위계표를 만들고 연습 계획을 제시하는 메시지를 생성합니다.
    """
    try:
        logger.info(f"강박 분석9 요청: session_id={request.session_id}")
        analysis_result = chatbot_service.generate_obsession_analysis9_response(request.conversation_history)
        logger.info("강박 분석9 완료: 응답 생성됨")
        
        # anxiety_hierarchy 데이터 변환 (딕셔너리로 직접 전달)
        hierarchy_items = []
        for item in analysis_result.get("anxiety_hierarchy", []):
            hierarchy_items.append({
                "order": item.get("order", 0),
                "situation": item.get("situation", ""),
                "score": item.get("score", 0)
            })
        
        return ObsessionAnalysis9Response(
            session_id=request.session_id,
            intro_message=analysis_result.get("intro_message", ""),
            anxiety_hierarchy=hierarchy_items,
            practice_message=analysis_result.get("practice_message", ""),
            example_message=analysis_result.get("example_message", ""),
            support_message=analysis_result.get("support_message", "")
        )
    except Exception as e:
        logger.error(f"강박 분석9 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

# analyze10은 analyze9와 동일한 기능이므로 제거됨

@router.post("/report", response_model=ObsessionReportResponse)
async def generate_obsession_report(request: ObsessionReportRequest):
    """
    전체 대화내역을 기반으로 강박 유형, 사고, 행동, 불안 위계를 정리한 리포트를 생성합니다.
    """
    try:
        logger.info(f"강박 리포트 요청: session_id={request.session_id}")
        report_data = chatbot_service.generate_obsession_report(request.conversation_history, request.session_id)
        logger.info("강박 리포트 완료: 리포트 생성됨")
        
        # anxiety_hierarchy 데이터 변환
        hierarchy_items = []
        for item in report_data.get("anxiety_hierarchy", []):
            hierarchy_items.append(AnxietyHierarchyItem(
                situation=item.get("situation", ""),
                score=item.get("score", 0)
            ))
        
        return ObsessionReportResponse(
            session_id=request.session_id,
            obsession_type=report_data.get("obsession_type", "기타"),
            anxiety_thoughts=report_data.get("anxiety_thoughts", ""),
            compulsive_behaviors=report_data.get("compulsive_behaviors", ""),
            anxiety_hierarchy=hierarchy_items
        )
    except Exception as e:
        logger.error(f"강박 리포트 생성 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@router.post("/report2", response_model=ObsessionReport2Response)
async def generate_obsession_report2(request: ObsessionReport2Request):
    """
    대화내역을 기반으로 맞춤형 5주차 강박개선 플랜을 생성합니다.
    """
    try:
        logger.info(f"강박 리포트2 요청: session_id={request.session_id}")
        report_data = chatbot_service.generate_obsession_report2(request.conversation_history)
        logger.info("강박 리포트2 완료: 리포트 생성됨")
        
        # weeks 데이터 변환
        weeks = []
        for week in report_data.get("weeks", []):
            weeks.append(WeekInfo(
                week=week.get("week", 0),
                start_date=week.get("start_date", ""),
                end_date=week.get("end_date", "")
            ))
        
        # training_plan 데이터 변환
        training_plan = []
        for training in report_data.get("training_plan", []):
            training_plan.append(TrainingItem(
                week=training.get("week", 0),
                training_name=training.get("training_name", ""),
                goal=training.get("goal", ""),
                content=training.get("content", "")
            ))
        
        # training_explanation 데이터 변환
        training_explanation = {}
        for key, explanation in report_data.get("training_explanation", {}).items():
            training_explanation[key] = TrainingExplanation(
                title=explanation.get("title", ""),
                description=explanation.get("description", ""),
                when=explanation.get("when", ""),
                instruction=explanation.get("instruction", "")
            )
        
        return ObsessionReport2Response(
            session_id=request.session_id,
            obsession_type=report_data.get("obsession_type", "기타"),
            training_explanation=training_explanation,
            weeks=weeks,
            training_plan=training_plan
        )
    except Exception as e:
        logger.error(f"강박 리포트2 생성 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")