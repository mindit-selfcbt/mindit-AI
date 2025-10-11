import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from core.config import settings
from core.logging import get_logger
from services.rag_service import RAGService
from services.chatbot_service import ChatbotService

logger = get_logger(__name__)

class ResponsePreventionService:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=0.7
        )
        self.rag_service = RAGService()
        self.chatbot_service = ChatbotService()

    def _extract_user_messages(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        대화 히스토리에서 사용자 메시지를 추출하여 하나의 문자열로 결합합니다.
        """
        user_messages = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_messages.append(content)
                elif isinstance(content, dict):
                    user_messages.append(str(content))
        return " ".join(user_messages)

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        LLM 응답에서 JSON을 파싱합니다.
        """
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        return None

    def generate_response_prevention_analysis(self, obsession_situation: str, anxiety_level_start: int, 
                                            anxiety_level_end: int, obsession_thought: str) -> Dict[str, Any]:
            system_prompt = """당신은 경험 많은 상담가입니다.
            사용자의 반응방지 완료 후 상황을 분석하여 맞춤형 응답을 생성해주세요.

            **응답 구성 요소:**
            1. 인지적 오류 설명: 왜곡된 사고를 현실적으로 바꾸는 과정 설명
            2. 맞춤형 상황: 사용자의 구체적인 강박 상황을 반영한 문장
            3. 감정 질문: 사용자가 느꼈을 감정에 대한 질문

            **응답 형식 (JSON):**
            {
                "content": {
                "cognitive_error_explanation": "'인지적 오류 파악하기'는 나의 왜곡된 사고를 유연하고 현실적인 사고로 바꾸는 과정을 말합니다.\\n\\n예를 들어 \"이걸 만지면 병에 걸릴 거야\"와 같은 비현실적인 사고를 \"실제로 병에 걸릴 가능성은 매우 낮아\"와 같은 인식으로 전환하는 것을 말해요.",
                "personalized_situation": "반응 방지를 하면서 \"[사용자 맞춤 상황]\" 라고 했었어요.",
                    "emotion_question": "그 생각이 들었을 때, 어떤 감정을 느꼈나요? (불안, 찝찝함, 공포, 분노, 초조함 등)"
                },
                "obsession_type": "확인 강박" 또는 "오염 강박" 또는 "기타"
            }

            **중요한 규칙:**
            1. personalized_situation에서 [사용자 맞춤 상황] 부분을 사용자의 실제 강박 상황과 생각으로 대체하세요.
            2. 사용자의 강박 유형에 맞게 상황을 구체적으로 작성하세요.
            3. emotion_question에는 질문과 함께 예시 감정들을 괄호 안에 포함하세요."""

            user_prompt = f"""강박 상황: {obsession_situation}
            불안 정도 (시작): {anxiety_level_start}/100
            불안 정도 (완료): {anxiety_level_end}/100
            강박 사고: {obsession_thought}"""

            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                response = self.llm.invoke(messages)
                response_text = response.content
                
                result = self._parse_json_response(response_text)
                if result:
                    result["obsession_type"] = self._classify_obsession_type(obsession_situation, obsession_thought)
                    # encouragement_message를 고정형으로 추가
                    if "content" in result:
                        result["content"]["encouragement_message"] = "반응 방지를 잘 마치셨네요. 고생 많았어요! ☺️"
                    return result
                
                default_response = self._get_default_response(obsession_situation, obsession_thought)
                default_response["obsession_type"] = self._classify_obsession_type(obsession_situation, obsession_thought)
                # encouragement_message를 고정형으로 추가
                default_response["content"]["encouragement_message"] = "반응 방지를 잘 마치셨네요. 고생 많았어요! ☺️"
                return default_response
            
            except Exception as e:
                logger.error(f"반응방지 분석 생성 중 오류: {e}")
                default_response = self._get_default_response(obsession_situation, obsession_thought)
                default_response["obsession_type"] = self._classify_obsession_type(obsession_situation, obsession_thought)
                # encouragement_message를 고정형으로 추가
                default_response["content"]["encouragement_message"] = "반응 방지를 잘 마치셨네요. 고생 많았어요! ☺️"
                return default_response

    def _get_default_response(self, obsession_situation: str, obsession_thought: str) -> Dict[str, Any]:
        return {
            "content": {
                "cognitive_error_explanation": "'인지적 오류 파악하기'는 나의 왜곡된 사고를 유연하고 현실적인 사고로 바꾸는 과정을 말합니다.\\n\\n예를 들어 \"이걸 만지면 병에 걸릴 거야\"와 같은 비현실적인 사고를 \"실제로 병에 걸릴 가능성은 매우 낮아\"와 같은 인식으로 전환하는 것을 말해요.",
                "personalized_situation": f"반응 방지를 하면서 \"{obsession_thought}\" 라고 했었어요.",
                "emotion_question": "그 생각이 들었을 때, 어떤 감정을 느꼈나요? (불안, 찝찝함, 공포, 분노, 초조함 등)"
            }
        }
        
    def generate_response_prevention_analysis2(self, session_id: str, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        대화 히스토리를 기반으로 RAG를 사용하여 인지적 오류를 분석합니다.
        """
        try:
            # 대화 히스토리에서 사용자 메시지 추출
            full_context = self._extract_user_messages(conversation_history)
            
            # RAG를 사용하여 관련 인지적 오류 검색
            cognitive_errors = self.rag_service.extract_cognitive_errors_from_context(full_context)
            
            # LLM을 사용하여 사용자 상황에 맞게 인지적 오류 설명 생성
            if cognitive_errors:
                cognitive_errors = self._customize_cognitive_errors_with_llm(cognitive_errors, full_context)
            
            return cognitive_errors
            
        except Exception as e:
            logger.error(f"반응방지 분석2 생성 중 오류: {e}")
            return []
    
    def _customize_cognitive_errors_with_llm(self, cognitive_errors: List[Dict[str, str]], context: str) -> List[Dict[str, str]]:
        """
        LLM을 사용하여 사용자 상황에 맞게 인지적 오류 설명을 커스터마이징합니다.
        """
        system_prompt = """당신은 경험 많은 상담가입니다.
        사용자의 대화 내용을 바탕으로 발견된 인지적 오류들을 사용자 상황에 맞게 설명해주세요.
        
        **응답 형식 (JSON):**
        {
            "cognitive_errors": [
                {
                    "error_type": "인지적 오류 유형",
                    "description": "사용자 상황에 맞는 구체적인 설명 (1-2줄로 간결하게, 사용자 말을 작은따옴표로 인용)"
                }
            ]
        }
        
        **중요한 규칙:**
        1. description은 사용자의 실제 상황과 감정을 반영하여 1-2줄로 간결하게 작성하세요.
        2. 사용자가 언급한 내용의 핵심을 요약해서 작은따옴표로 인용해, 해당 내용을 포함하세요요.
        3. 각 인지적 오류는 사용자가 경험한 상황과 연결지어 설명하세요.
        4. 따뜻하고 공감적인 톤으로 작성하세요."""
        
        user_prompt = f"""사용자 대화 내용: {context}

발견된 인지적 오류들:
{json.dumps(cognitive_errors, ensure_ascii=False, indent=2)}"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # JSON 파싱 시도
            result = self._parse_json_response(response_text)
            if result:
                return result.get("cognitive_errors", cognitive_errors)
            
            return cognitive_errors
            
        except Exception as e:
            logger.error(f"인지적 오류 커스터마이징 중 오류: {e}")
            return cognitive_errors

    def _classify_obsession_type(self, obsession_situation: str, obsession_thought: str) -> str:
        """
        강박 상황과 사고를 기반으로 강박 유형을 분류합니다.
        기존 chatbot_service의 LLM 기반 분류 로직을 활용합니다.
        """
        # 임시 대화 히스토리 생성 (분류 함수 사용을 위해)
        temp_conversation_history = [
            {"role": "user", "content": f"{obsession_situation} {obsession_thought}"}
        ]
        
        # 기존 chatbot_service의 LLM 기반 분류 함수 사용
        llm_result = self.chatbot_service.categorize_obsession_type(temp_conversation_history)
        logger.info(f"강박 유형 분류 결과: {llm_result}, 입력: {obsession_situation} | {obsession_thought}")
        
        # LLM 결과를 한국어로 변환
        if llm_result == "contamination":
            return "오염강박"
        elif llm_result == "checking":
            return "확인강박"
        elif llm_result == "other":
            return "기타"
        else:
            logger.warning(f"예상치 못한 강박 유형: {llm_result}, 기본값 '기타' 사용")
            return "기타"

    def generate_response_prevention_analysis3(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        대화 히스토리를 기반으로 공감 메시지와 통찰 메시지를 생성합니다.
        """
        try:
            # 대화 히스토리에서 사용자 메시지 추출
            full_context = self._extract_user_messages(conversation_history)
            
            system_prompt = """당신은 경험 많은 상담가입니다.
            사용자의 대화 내용을 바탕으로 공감과 통찰을 담은 메시지를 생성해주세요.
            
            **응답 형식 (JSON):**
            {
                "empathy_message": "불안이 당신을 지켜주려는 마음에서 비롯된 건 맞지만, 결국은 많이 지치게 만들고 있다는 걸 깨닫는 것도 정말 중요한 통찰이에요.",
                "insight_message": "이렇게 솔직하게 자신을 돌아본 건, 회복을 향한 아주 큰 한 걸음이에요. 지금 느낀 걸 잘 기억해두면, 다음에 불안이 찾아왔을 때 '지키려는 마음'과 '지치게 만드는 영향'을 더 잘 구별할 수 있을 거예요."
            }
            
            **중요한 규칙:**
            1. empathy_message는 사용자의 구체적인 상황과 감정을 반영하여 공감적으로 작성하세요.
            2. insight_message는 사용자가 깨달은 점을 인정하고 회복에 대한 희망을 주는 메시지로 작성하세요.
            3. 사용자가 언급한 구체적인 상황이나 감정을 포함하세요.
            4. 따뜻하고 격려하는 톤으로 작성하세요."""
            
            user_prompt = f"사용자 대화 내용: {full_context}"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # JSON 파싱 시도
            result = self._parse_json_response(response_text)
            if result:
                return result
            
            # 기본값 반환
            return {
                "empathy_message": "불안이 당신을 지켜주려는 마음에서 비롯된 건 맞지만, 결국은 많이 지치게 만들고 있다는 걸 깨닫는 것도 정말 중요한 통찰이에요.",
                "insight_message": "이렇게 솔직하게 자신을 돌아본 건, 회복을 향한 아주 큰 한 걸음이에요. 지금 느낀 걸 잘 기억해두면, 다음에 불안이 찾아왔을 때 '지키려는 마음'과 '지치게 만드는 영향'을 더 잘 구별할 수 있을 거예요."
            }
            
        except Exception as e:
            logger.error(f"반응방지 분석3 생성 중 오류: {e}")
            return {
                "empathy_message": "불안이 당신을 지켜주려는 마음에서 비롯된 건 맞지만, 결국은 많이 지치게 만들고 있다는 걸 깨닫는 것도 정말 중요한 통찰이에요.",
                "insight_message": "이렇게 솔직하게 자신을 돌아본 건, 회복을 향한 아주 큰 한 걸음이에요. 지금 느낀 걸 잘 기억해두면, 다음에 불안이 찾아왔을 때 '지키려는 마음'과 '지치게 만드는 영향'을 더 잘 구별할 수 있을 거예요."
            }

    def generate_response_prevention_analysis3_2(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        오염강박용 analyze3-2: 구체적인 근거 질문을 생성합니다.
        """
        try:
            # 대화 히스토리에서 사용자 메시지 추출
            full_context = self._extract_user_messages(conversation_history)
            
            system_prompt = """당신은 경험 많은 상담가입니다.
            오염강박을 가진 사용자에게 구체적인 근거를 묻는 질문을 생성해주세요.
            
            **응답 형식:**
            "좋아요. 그런 경험이 실제로 없었다는 건 중요한 단서예요. 지금 떠오른 생각은 사실이 아니라 가능성에 대한 두려움일 수 있다는 거죠.
            
            그럼, 조금 더 구체적으로 살펴볼게요.
            
            [사용자 상황에 맞는 구체적인 질문]"
            
            **중요한 규칙:**
            1. 사용자가 언급한 구체적인 오염 상황을 반영하여 질문을 작성하세요.
            2. "손잡이를 잡는다고 해서 반드시 병에 걸릴 거라고 믿는 근거는 무엇인가요?"와 같은 구체적인 질문을 포함하세요.
            3. "그 생각이 실제 현실에서도 증명되었던 적이 있나요?"와 같은 현실 검증 질문을 포함하세요.
            4. 따뜻하고 공감적인 톤을 유지하세요."""
            
            user_prompt = f"사용자 대화 내용: {full_context}"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"반응방지 분석3-2 생성 중 오류: {e}")
            return "좋아요. 그런 경험이 실제로 없었다는 건 중요한 단서예요. 지금 떠오른 생각은 사실이 아니라 가능성에 대한 두려움일 수 있다는 거죠.\n\n그럼, 조금 더 구체적으로 살펴볼게요.\n\n손잡이를 잡는다고 해서 반드시 병에 걸릴 거라고 믿는 근거는 무엇인가요? 그리고 그 생각이 실제 현실에서도 증명되었던 적이 있나요?"

    def generate_response_prevention_analysis3_3(self, session_id: str, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        오염강박용 analyze3-3: 인지적 재구성과 격려 메시지를 생성합니다.
        """
        try:
            # 대화 히스토리에서 사용자 메시지 추출
            full_context = self._extract_user_messages(conversation_history)
            
            system_prompt = """당신은 경험 많은 상담가입니다.
            오염강박을 가진 사용자에게 인지적 재구성과 격려 메시지를 생성해주세요.
            
            **응답 형식 (JSON):**
            {
                "intro_message": "지금 떠오르는 생각은 '혹시나'에 대한 거잖아요. 하지만 '혹시나'가 매번 현실이 되지는 않아요. 한번 이렇게 바꿔서 생각해보면 어떨까요?",
                "cognitive_restructuring": {
                    "old_thought": "사용자 상황에 맞는 기존 생각",
                    "new_perspective": "사용자 상황에 맞는 새로운 관점"
                },
                "encouragement_message": "불안은 우리가 통제할 수 없는 상황을 대비하기 위해 생기는 감정이에요. 하지만 지금처럼 현실을 기준으로 사고를 다시 구성하는 연습을 계속하다 보면, 그 불안도 점점 작아질 수 있어요.",
                "insight_message": "지금 이 순간, 그런 불안을 견디며 대화를 이어가고 있는 당신은 정말 대단해요. 그 자체가 이미 변화의 시작이에요.",
                "final_encouragement": "불안을 없애는 게 아니라, 다르게 반응하는 연습을 계속하고 있다는 걸 기억해주세요. 그 자체로 지금 충분히 잘하고 있어요.💪",
                "choice_question": "오늘 대화는 여기서 마무리해볼까요, 아니면 조금 더 이야기 나눌까요?"
            }
            
            **중요한 규칙:**
            1. 사용자가 언급한 구체적인 오염 상황을 반영하여 기존생각과 새로운 관점을 작성하세요.
            2. '혹시나'에 대한 두려움을 현실적 관점으로 전환하는 내용을 포함하세요.
            3. 격려와 공감의 메시지를 포함하세요.
            4. 각 메시지는 간결하고 따뜻하게 작성하세요."""
            
            user_prompt = f"사용자 대화 내용: {full_context}"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # JSON 파싱 시도
            result = self._parse_json_response(response_text)
            if result:
                result["session_id"] = session_id
                return result
            
            # 기본값 반환
            return {
                "session_id": session_id,
                "intro_message": "지금 떠오르는 생각은 '혹시나'에 대한 거잖아요. 하지만 '혹시나'가 매번 현실이 되지는 않아요. 한번 이렇게 바꿔서 생각해보면 어떨까요?",
                "cognitive_restructuring": {
                    "old_thought": "손잡이를 잠깐 만져도 병이 옮을 수 있어. 그러니까 절대 만지면 안 돼.",
                    "new_perspective": "손잡이를 만지면 불편한 느낌이 들 수는 있지만, 실제로 병에 걸린 적은 없어. 그 불편함은 시간이 지나면서 사라지고, 나는 그걸 감당할 수 있어."
                },
                "encouragement_message": "불안은 우리가 통제할 수 없는 상황을 대비하기 위해 생기는 감정이에요. 하지만 지금처럼 현실을 기준으로 사고를 다시 구성하는 연습을 계속하다 보면, 그 불안도 점점 작아질 수 있어요.",
                "insight_message": "지금 이 순간, 그런 불안을 견디며 대화를 이어가고 있는 당신은 정말 대단해요. 그 자체가 이미 변화의 시작이에요.",
                "final_encouragement": "불안을 없애는 게 아니라, 다르게 반응하는 연습을 계속하고 있다는 걸 기억해주세요. 그 자체로 지금 충분히 잘하고 있어요.💪",
                "choice_question": "오늘 대화는 여기서 마무리해볼까요, 아니면 조금 더 이야기 나눌까요?"
            }
            
        except Exception as e:
            logger.error(f"반응방지 분석3-3 생성 중 오류: {e}")
            return {
                "session_id": session_id,
                "intro_message": "지금 떠오르는 생각은 '혹시나'에 대한 거잖아요. 하지만 '혹시나'가 매번 현실이 되지는 않아요. 한번 이렇게 바꿔서 생각해보면 어떨까요?",
                "cognitive_restructuring": {
                    "old_thought": "손잡이를 잠깐 만져도 병이 옮을 수 있어. 그러니까 절대 만지면 안 돼.",
                    "new_perspective": "손잡이를 만지면 불편한 느낌이 들 수는 있지만, 실제로 병에 걸린 적은 없어. 그 불편함은 시간이 지나면서 사라지고, 나는 그걸 감당할 수 있어."
                },
                "encouragement_message": "불안은 우리가 통제할 수 없는 상황을 대비하기 위해 생기는 감정이에요. 하지만 지금처럼 현실을 기준으로 사고를 다시 구성하는 연습을 계속하다 보면, 그 불안도 점점 작아질 수 있어요.",
                "insight_message": "지금 이 순간, 그런 불안을 견디며 대화를 이어가고 있는 당신은 정말 대단해요. 그 자체가 이미 변화의 시작이에요.",
                "final_encouragement": "불안을 없애는 게 아니라, 다르게 반응하는 연습을 계속하고 있다는 걸 기억해주세요. 그 자체로 지금 충분히 잘하고 있어요.💪",
                "choice_question": "오늘 대화는 여기서 마무리해볼까요, 아니면 조금 더 이야기 나눌까요?"
            }

    def generate_response_prevention_analysis4_1(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        더 이야기 나누기를 선택했을 때의 응답을 생성합니다.
        RAG를 활용하여 전문적인 응답을 제공합니다.
        """
        try:
            # 대화 히스토리에서 사용자 메시지 추출
            full_context = self._extract_user_messages(conversation_history)
            
            # RAG를 사용하여 관련 인지적 오류 검색
            cognitive_errors = self.rag_service.extract_cognitive_errors_from_context(full_context)
            
            system_prompt = """당신은 경험 많은 상담가입니다.
            사용자가 더 이야기 나누기를 선택했을 때, RAG로 검색된 인지적 오류 정보를 활용하여 전문적이고 공감적인 응답을 제공해주세요.
            
            **중요한 규칙:**
            1. 사용자의 대화 내용을 바탕으로 공감적인 응답을 작성하세요.
            2. RAG로 검색된 인지적 오류 정보를 활용하여 전문적인 조언을 제공하세요.
            3. 사용자가 계속 대화할 수 있도록 자연스러운 질문을 포함하세요.
            4. 따뜻하고 전문적인 톤을 유지하세요."""
            
            user_prompt = f"""사용자 대화 내용: {full_context}

관련 인지적 오류 정보:
{json.dumps(cognitive_errors, ensure_ascii=False, indent=2) if cognitive_errors else "관련 정보 없음"}"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"반응방지 분석4-1 생성 중 오류: {e}")
            return "더 이야기 나눌 수 있어서 좋아요. 어떤 부분이 가장 어려웠는지, 또는 지금 어떤 생각이 드시나요?"

    def generate_response_prevention_analysis4_2(self) -> str:
        """
        마무리를 선택했을 때의 고정 메시지를 반환합니다.
        """
        return "좋아요😄 수고하셨습니다! 아래 버튼을 눌러 인지적 오류리포트를 확인하세요."

    def generate_response_prevention_report(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        전체 대화 내역을 기반으로 종합 리포트를 생성합니다.
        """
        try:
            # 대화 히스토리에서 사용자 메시지 추출
            full_context = self._extract_user_messages(conversation_history)
            
            # 현재 시간을 한국 시간으로 포맷팅
            from datetime import datetime
            import locale
            
            # 한국 시간대 설정
            kst = datetime.now().astimezone()
            
            # 요일 한글 변환
            weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
            weekday = weekdays[kst.weekday()]
            
            # 날짜 포맷팅 (00월 00일 0요일)
            consultation_date = f"{kst.month:02d}월 {kst.day:02d}일 {weekday}"
            
            # 시간 포맷팅 (오전/오후 00시 00분)
            if kst.hour < 12:
                time_period = "오전"
                hour = kst.hour if kst.hour != 0 else 12
            else:
                time_period = "오후"
                hour = kst.hour - 12 if kst.hour != 12 else 12
            
            consultation_time = f"{time_period} {hour:02d}시 {kst.minute:02d}분"
            
            system_prompt = """당신은 경험 많은 상담가입니다.
            사용자의 전체 대화 내용을 바탕으로 종합적인 강박 분석 리포트를 생성해주세요.
            
            **응답 형식 (JSON):**
            {
                "obsession_type": "확인 강박" 또는 "오염 강박" 또는 "기타",
                "obsession_situation": "사용자가 언급한 구체적인 강박 상황",
                "intrusive_thoughts": "당장 손을 씻지 않으면 병에 걸릴 것 같다는 생각이 든다",
                "false_beliefs": "손을 씻지 않으면 걱정하는 일이 벌어질 것이라는 믿음",
                "cognitive_errors": [
                    {
                        "error_type": "과도한 일반화",
                        "description": "'한 번 병에 걸릴 수도 있다'는 가능성을 '항상 그럴 것이다'로 과도하게 일반화한다"
                    },
                    {
                        "error_type": "예언자적 오류",
                        "description": "손잡이를 잡는 행동이 바로 병에 걸리는 심각한 결과로 이어질 것이라는 생각은 실제 증거나 정보 없이 미래를 단정하는 경향이다"
                    },
                    {
                        "error_type": "의미 확대",
                        "description": "병에 걸릴 가능성을 실제보다 높게 평가한다"
                    }
                ],
                "realistic_thinking": "손잡이를 만지면 불편한 느낌이 들 수는 있지만, 실제로 병에 걸린 적은 없다. 또한 그 불편함은 시간이 지나면서 사라지고, 나는 그걸 감당할 수 있다."
            }
            
            **중요한 규칙:**
            1. intrusive_thoughts는 "당장 ~하지 않으면 ~할 것 같다는 생각이 든다" 형태로 작성하세요.
            2. false_beliefs는 "~하지 않으면 ~일 것이라는 믿음" 형태로 작성하세요.
            3. cognitive_errors는 분석된 인지적 오류 3가지를 배열로 작성하되, 각각 error_type과 description을 포함하세요.
            4. realistic_thinking은 사용자가 깨달은 현실적인 관점을 구체적으로 작성하세요.
            5. 사용자가 실제로 언급한 내용을 바탕으로 작성하세요."""
            
            user_prompt = f"사용자 전체 대화 내용: {full_context}"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # JSON 파싱 시도
            result = self._parse_json_response(response_text)
            if result:
                result["consultation_date"] = consultation_date
                result["consultation_time"] = consultation_time
                return result
            
                # 기본값 반환
                return {
                    "consultation_date": consultation_date,
                    "consultation_time": consultation_time,
                    "obsession_type": "기타",
                    "obsession_situation": "사용자가 언급한 강박 상황",
                    "intrusive_thoughts": "당장 ~하지 않으면 ~할 것 같다는 생각이 든다",
                    "false_beliefs": "~하지 않으면 ~일 것이라는 믿음",
                    "cognitive_errors": [
                        {
                            "error_type": "과도한 일반화",
                            "description": "관련된 인지적 오류 분석"
                        }
                    ],
                    "realistic_thinking": "사용자가 깨달은 현실적인 생각"
                }
            
        except Exception as e:
            logger.error(f"반응방지 리포트 생성 중 오류: {e}")
            # 현재 시간을 한국 시간으로 포맷팅 (예외 처리용)
            from datetime import datetime
            kst = datetime.now().astimezone()
            weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
            weekday = weekdays[kst.weekday()]
            consultation_date = f"{kst.month:02d}월 {kst.day:02d}일 {weekday}"
            
            if kst.hour < 12:
                time_period = "오전"
                hour = kst.hour if kst.hour != 0 else 12
            else:
                time_period = "오후"
                hour = kst.hour - 12 if kst.hour != 12 else 12
            consultation_time = f"{time_period} {hour:02d}시 {kst.minute:02d}분"
            
            return {
                "consultation_date": consultation_date,
                "consultation_time": consultation_time,
                "obsession_type": "기타",
                "obsession_situation": "사용자가 언급한 강박 상황",
                "intrusive_thoughts": "당장 ~하지 않으면 ~할 것 같다는 생각이 든다",
                "false_beliefs": "~하지 않으면 ~일 것이라는 믿음",
                "cognitive_errors": [
                    {
                        "error_type": "과도한 일반화",
                        "description": "관련된 인지적 오류 분석"
                    }
                ],
                "realistic_thinking": "사용자가 깨달은 현실적인 생각"
            }
