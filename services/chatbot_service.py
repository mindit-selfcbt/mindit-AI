import uuid
import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

class ChatbotService:
    #공통 시스템 프롬프트
    COMMON_SYSTEM_PROMPT = """당신은 경험 많은 상담가입니다. 
    사용자의 고민을 듣고 공감하며, 전문적이고 따뜻한 조언을 제공해주세요.
    강박증, 불안, 우울 등 정신건강 관련 문제에 대해 전문적인 관점에서 답변하되,
    항상 전문의 상담을 권장하는 것을 잊지 마세요."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=0.7
        )
        # 세션별 강박 유형 저장
        self.session_obsession_types = {}

    #이 함수 삭제할 수도 있음.    
    def _stringify_message_content(self, content: Any) -> str:
        """
        Normalize arbitrary message content into a readable string.
        - Lists are joined with ", ".
        - Dicts are JSON-encoded with ensure_ascii=False.
        - Others are coerced via str().
        """
        try:
            if isinstance(content, list):
                return ", ".join(str(item) for item in content)
            if isinstance(content, dict):
                return json.dumps(content, ensure_ascii=False)
            return str(content)
        except Exception:
            return str(content)
    
    def generate_obsession_question(self, user_text: str) -> Dict[str, Any]:
        """
        사용자의 텍스트를 바탕으로 강박 관련 질문과 선택지를 생성합니다.
        """
        system_prompt = """당신은 경험 많은 상담가입니다. 
        사용자의 텍스트를 분석하여 강박적 사고나 행동 패턴을 파악하고, 
        더 깊이 있는 상담을 위한 자연스러운 질문과 선택지를 생성해주세요.
        
        **응답 형식 (JSON):**
        {
            "question": "[자연스러운 대화형 질문]",
            "choices": ["선택지1", "선택지2", "선택지3"]
        }
        
        **중요한 규칙:**
        1. question은 **자연스럽고 대화형**으로 만들어주세요. "~알아보고 싶습니다" 같은 형식적 표현을 피하세요.
        2. 사용자의 상황을 이해하고 공감하는 듯한 질문을 만들어주세요.
        3. choices는 "~할 때" 형태로 3개를 만들어주세요. **중복이나 어색한 표현을 피하세요**.
        4. 선택지에서 "~전에할 때", "~후에할 때" 같은 중복 표현을 피하세요.
        5. 원형 표현을 피하세요 (예: 완벽주의에 "완벽함을 추구할 때"는 중복).
        
        **좋은 예시:**
        - 입력: "손을 계속 씻어야 한다는 생각이 들어요"
        - 응답: {
            "question": "손이 더럽다고 느낄 때, 보통 어떤 상황에서 그런 생각이 드나요?",
            "choices": ["스트레스가 있을 때", "특정 장소에 있을 때", "불안감이 높을 때"]
        }
        
        - 입력: "문을 잠갔는지 계속 확인해야 해요"
        - 응답: {
            "question": "문을 잠갔는지 확인하고 싶을 때, 주로 언제 그런 생각이 드나요?",
            "choices": ["외출할 때", "잠자리에 들 때", "불안감이 높을 때"]
        }
        
        - 입력: "모든 것이 완벽해야 한다는 생각이 들어요"
        - 응답: {
            "question": "완벽해야 한다고 느낄 때, 어떤 상황에서 그런 압박감을 받으시나요?",
            "choices": ["새로운 일을 시작할 때", "작업을 마무리할 때", "타인의 평가를 받을 때"]
        }"""
        
        user_prompt = f"사용자 텍스트: {user_text}"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content
            
            #JSON 파싱 시도
            import json
            try:
                #JSON 부분만 추출
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    return result
            except:
                pass
            
            #JSON 파싱 실패 시 기본값
            return {
                "question": f"{user_text}에 대해 더 자세히 알아보고 싶습니다.",
                "choices": [
                    "스트레스가 있을 때",
                    "특정 상황에서", 
                    "불안감이 높을 때"
                ]
            }
            
        except Exception as e:
            logger.error(f"LLM 호출 중 오류 발생: {e}")
            return {
                "question": f"{user_text}에 대해 더 자세히 알아보고 싶습니다.",
                "choices": [
                    "스트레스가 있을 때",
                    "특정 상황에서", 
                    "불안감이 높을 때"
                ]
            }
    
    def generate_obsession_analysis2_response(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        대화 히스토리를 분석하여 강박 행동에 대한 공감적 질문을 생성합니다.
        """
        system_prompt = """당신은 경험 많은 상담가입니다. 
        사용자의 대화 히스토리를 분석하여 강박적 행동이나 사고 패턴을 파악하고,
        공감적이고 따뜻한 질문을 생성해주세요.
        
        **응답 형식:**
        "말씀해주셔서 감사해요.
        혹시 [사용자의 강박 행동을 구체적으로 언급]하면 불편했던 마음이
        좀 나아지나요?"
        
        **중요한 규칙:**
        1. 사용자가 언급한 구체적인 강박 행동을 파악하여 [ ] 부분에 넣어주세요.
        2. 예시:
           - 사용자가 "손이 더럽다고 계속 느낀다"고 말했다면 → "손을 씻으면"
           - 사용자가 "문을 잠갔는지 확인한다"고 말했다면 → "문을 확인하면"
           - 사용자가 "완벽해야 한다고 생각한다"고 말했다면 → "완벽하게 하면"
        3. 공감적이고 따뜻한 톤을 유지하세요.
        4. 강박 행동을 부정적으로 표현하지 말고, 중립적으로 표현하세요."""
        
        #대화 히스토리에서 사용자 메시지 추출
        user_messages = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_messages.append(self._stringify_message_content(content))
        
        #최근 사용자 메시지들을 하나의 텍스트로 결합
        recent_context = " ".join(user_messages[-3:])  #최근 3개 메시지만 사용
        
        user_prompt = f"대화 히스토리: {recent_context}"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"강박 분석2 응답 생성 중 오류: {e}")
            return "말씀해주셔서 감사해요.\n혹시 그런 행동을 하면 불편했던 마음이\n좀 나아지나요?"
    
    def generate_obsession_analysis3_response(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        대화 히스토리를 분석하여 강박 패턴 요약과 생각 예시를 생성합니다.
        """
        system_prompt = """당신은 경험 많은 상담가입니다. 
        사용자의 대화 히스토리를 분석하여 강박적 사고나 행동 패턴을 파악하고,
        사용자의 패턴을 요약하고 관련된 생각 예시를 생성해주세요.
        
        **응답 형식 (JSON):**
        {
            "user_pattern_summary": "당신은 [구체적인 강박 패턴]하는 경향이 있는 것 같아요.",
            "thought_examples": [
                "생각 예시 1",
                "생각 예시 2", 
                "생각 예시 3"
            ]
        }
        
        **중요한 규칙:**
        1. user_pattern_summary는 사용자가 언급한 구체적인 강박 행동을 바탕으로 작성하세요.
        2. 예시:
           - 손 씻기 강박: "당신은 손이 오염됐을 것 같다는 불안이 자주 들고, 그 불안을 줄이기 위해 손 씻기를 반복하는 경향이 있는 것 같아요."
           - 확인 강박: "당신은 문을 제대로 잠갔는지, 가스를 끄지 않았는지 걱정이 되어 반복적으로 확인하는 경향이 있는 것 같아요."
           - 완벽주의: "당신은 모든 것을 완벽하게 해야 한다는 압박감을 느끼고, 실수를 방지하기 위해 반복적으로 확인하는 경향이 있는 것 같아요."
        3. thought_examples는 해당 강박과 관련된 구체적인 생각 3개를 생성하세요.
        4. 생각 예시는 실제로 강박을 경험하는 사람이 가질 법한 현실적인 생각으로 작성하세요."""
        
        #대화 히스토리에서 사용자 메시지 추출
        user_messages = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_messages.append(self._stringify_message_content(content))
        
        #최근 사용자 메시지들을 하나의 텍스트로 결합
        recent_context = " ".join(user_messages[-5:])  #최근 5개 메시지 사용
        
        user_prompt = f"대화 히스토리: {recent_context}"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content
            
            #JSON 파싱 시도
            import json
            try:
                #JSON 부분만 추출
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    return result
            except:
                pass
            
            #JSON 파싱 실패 시 기본값
            return {
                "user_pattern_summary": "당신은 불안감을 줄이기 위해 반복적인 행동을 하는 경향이 있는 것 같아요.",
                "thought_examples": [
                    "이것을 하지 않으면 나쁜 일이 일어날 것 같아",
                    "확인하지 않으면 불안해져",
                    "완벽하지 않으면 실패할 것 같아"
                ]
            }
            
        except Exception as e:
            logger.error(f"강박 분석3 응답 생성 중 오류: {e}")
            return {
                "user_pattern_summary": "당신은 불안감을 줄이기 위해 반복적인 행동을 하는 경향이 있는 것 같아요.",
                "thought_examples": [
                    "이것을 하지 않으면 나쁜 일이 일어날 것 같아",
                    "확인하지 않으면 불안해져",
                    "완벽하지 않으면 실패할 것 같아"
                ]
            }

    def categorize_obsession_type(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        대화 히스토리를 분석하여 강박 유형을 카테고리화합니다.
        반환값: "contamination" (오염강박), "checking" (확인강박), "other" (그 외 강박)
        """
        system_prompt = """당신은 강박증 전문가입니다. 
        사용자의 대화 히스토리를 분석하여 강박 유형을 분류해주세요.
        
        **분류 기준:**
        1. 오염강박 (contamination): 
           - 손 씻기, 청소, 오염에 대한 불안
           - 세균, 바이러스, 더러움에 대한 두려움
           - 예: "손이 더럽다고 느껴서 계속 씻어야 해요", "세균이 무서워요"
        
        2. 확인강박 (checking):
           - 문 잠금, 가스 끄기, 전자제품 확인
           - 위치, 시간, 경로, 목적지 확인
           - 안전, 정확성, 완료 여부에 대한 반복적 확인
           - 예: "문을 잠갔는지 계속 확인해요", "가스를 끄지 않았나 걱정돼요", "내릴 역을 지나쳤는지 위치를 확인해야 해요", "늦을 것 같아서 시간을 계속 확인해요"
        
        3. 그 외 강박 (other):
           - 완벽주의, 순서/정리 강박, 수집 강박 등
           - 위 두 카테고리에 해당하지 않는 모든 강박
           - 예: "모든 것이 완벽해야 해요", "정해진 순서대로 해야 해요"
        
        **응답 형식:**
        반드시 다음 중 하나만 정확히 반환하세요:
        - "contamination"
        - "checking" 
        - "other"
        
        **중요한 규칙:**
        1. 사용자가 언급한 구체적인 강박 행동을 바탕으로 판단하세요.
        2. 애매한 경우에는 "other"로 분류하세요.
        3. 반드시 위 3개 값 중 하나만 반환하세요."""
        
        # 대화 히스토리에서 사용자 메시지 추출
        user_messages = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_messages.append(self._stringify_message_content(content))
        
        # 최근 사용자 메시지들을 하나의 텍스트로 결합
        recent_context = " ".join(user_messages[-5:])  # 최근 5개 메시지 사용
        
        user_prompt = f"대화 히스토리: {recent_context}"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            category = response.content.strip().lower()
            
            # 따옴표 제거
            category = category.replace('"', '').replace("'", "")
            
            # 유효한 카테고리인지 확인
            if category in ["contamination", "checking", "other"]:
                return category
            else:
                logger.warning(f"예상치 못한 카테고리 반환: {category}, 기본값 'other' 사용")
                return "other"
                
        except Exception as e:
            logger.error(f"강박 카테고리 분류 중 오류: {e}")
            return "other"

    def generate_obsession_analysis4_response(self, conversation_history: List[Dict[str, Any]], session_id: str = None) -> Dict[str, Any]:
        """
        대화 히스토리를 분석하여 강박 유형별 맞춤 응답을 생성합니다.
        """
        # 1단계: 강박 유형 카테고리화
        obsession_type = self.categorize_obsession_type(conversation_history)
        
        # 세션별 강박 유형 저장
        if session_id:
            self.session_obsession_types[session_id] = obsession_type
        
        # 2단계: 카테고리별 시나리오에 따른 응답 생성
        response_data = self._generate_category_specific_response(conversation_history, obsession_type)
        
        return response_data

    def _generate_category_specific_response(self, conversation_history: List[Dict[str, Any]], obsession_type: str) -> Dict[str, Any]:
        """
        강박 유형에 따른 맞춤 응답을 생성합니다.
        """
        # 대화 히스토리에서 사용자 메시지 추출
        user_messages = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_messages.append(self._stringify_message_content(content))
        
        recent_context = " ".join(user_messages[-5:])
        
        if obsession_type == "contamination":
            # 오염강박 시나리오
            system_prompt = """당신은 경험 많은 상담가입니다. 
            사용자의 오염강박 관련 대화를 분석하여 공감적이고 도움이 되는 응답을 생성해주세요.
            
            **응답 구성:**
            1. 공감과 이해 표현
            2. 오염에 대한 불안에 대한 설명
            3. 일상생활 방해 시 신호에 대한 언급
            4. 연습이 필요하다는 안내
            
            **응답 형식:**
            "맞아요. 누구나 그런 생각을 할 수 있어요.
            하지만 이런 생각이 너무 자주 떠오르거나,
            반복되는 행동(예: 손 씻기)이 일상생활을
            방해한다면 그건 '오염에 대한 불안'을 다루는
            연습이 필요하다는 신호일 수 있어요."
            
            **중요한 규칙:**
            1. 공감적이고 따뜻한 톤을 유지하세요.
            2. 사용자의 구체적인 상황을 반영하여 자연스럽게 작성하세요.
            3. 위 형식을 기본으로 하되, 사용자 상황에 맞게 조정하세요.
            4. 한글 기준 총 길이를 200자 이내로 작성하세요.
            5. '오염 강박', '확인 강박', '강박증', 'OCD' 같은 명칭/진단/유형 라벨은 언급하지 마세요. 행동과 경험만 자연스럽게 묘사하세요."""
            
        elif obsession_type == "checking":
            # 확인강박 시나리오
            system_prompt = """당신은 경험 많은 상담가입니다. 
            사용자의 확인강박 관련 대화를 분석하여 공감적이고 도움이 되는 응답을 생성해주세요.
            
            **응답 구성:**
            1. 공감과 이해 표현
            2. 확인에 대한 불안에 대한 설명
            3. 일상생활 방해 시 신호에 대한 언급
            4. 연습이 필요하다는 안내
            
            **응답 형식:**
            "맞아요. 누구나 그런 생각을 할 수 있어요.
            하지만 이런 생각이 너무 자주 떠오르거나,
            반복되는 행동(예: 확인하기)이 일상생활을
            방해한다면 그건 '확인에 대한 불안'을 다루는
            연습이 필요하다는 신호일 수 있어요."
            
            **중요한 규칙:**
            1. 공감적이고 따뜻한 톤을 유지하세요.
            2. 사용자의 구체적인 상황을 반영하여 자연스럽게 작성하세요.
            3. 위 형식을 기본으로 하되, 사용자 상황에 맞게 조정하세요.
            4. 한글 기준 총 길이를 200자 이내로 작성하세요.
            5. '오염 강박', '확인 강박', '강박증', 'OCD' 같은 명칭/진단/유형 라벨은 언급하지 마세요. 행동과 경험만 자연스럽게 묘사하세요."""
            
        else:  # other
            # 그 외 강박 시나리오
            system_prompt = """당신은 경험 많은 상담가입니다. 
            사용자의 강박 관련 대화를 분석하여 공감적이고 도움이 되는 응답을 생성해주세요.
            
            **응답 구성:**
            1. 공감과 이해 표현
            2. 강박에 대한 일반적인 설명
            3. 일상생활 방해 시 신호에 대한 언급
            4. 연습이 필요하다는 안내
            
            **응답 형식:**
            "맞아요. 누구나 그런 생각을 할 수 있어요.
            하지만 이런 생각이 너무 자주 떠오르거나,
            반복되는 행동이 일상생활을
            방해한다면 그건 '강박적 불안'을 다루는
            연습이 필요하다는 신호일 수 있어요."
            
            **중요한 규칙:**
            1. 공감적이고 따뜻한 톤을 유지하세요.
            2. 사용자의 구체적인 상황을 반영하여 자연스럽게 작성하세요.
            3. 위 형식을 기본으로 하되, 사용자 상황에 맞게 조정하세요.
            4. 한글 기준 총 길이를 200자 이내로 작성하세요.
            5. '오염 강박', '확인 강박', '강박증', 'OCD' 같은 명칭/진단/유형 라벨은 언급하지 마세요. 행동과 경험만 자연스럽게 묘사하세요."""
        
        user_prompt = f"대화 히스토리: {recent_context}"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            llm_response = response.content.strip()
            
            # 카테고리별 고정 메시지 생성
            if obsession_type == "contamination":
                category_message = "지금 당신이 이야기해주신 불편함은, '오염 강박'이라고 불리는 강박증의 한 유형과 비슷한 모습이에요."
                encouragement = "하지만 걱정하지 마세요. 이를 인식하고, 조금씩 다루는 연습을 해볼 수 있어요. 제가 그 과정을 도와드릴게요 😊"
            elif obsession_type == "checking":
                category_message = "지금 당신이 이야기해주신 불편함은, '확인 강박'이라고 불리는 강박증의 한 유형과 비슷한 모습이에요."
                encouragement = "하지만 걱정하지 마세요. 이를 인식하고, 조금씩 다루는 연습을 해볼 수 있어요. 제가 그 과정을 도와드릴게요 😊"
            else:  # other
                category_message = "아쉽게도 저희 Mindit 서비스에서는 언급해주신 강박에 도움을 드릴 수 있는 기능이 없어요."
                encouragement = "하지만 걱정하지 마세요. 전문기관에 방문하여 상담 받으신다면, 금방 해결해나가실 수 있을 거예요. 사용자분의 여정을 응원합니다."
            
            return {
                "user_pattern_summary": llm_response,
                "encouragement": encouragement,
                "category_message": category_message,
                "obsession_type": obsession_type
            }
            
        except Exception as e:
            logger.error(f"카테고리별 응답 생성 중 오류: {e}")
            # 기본값 반환
            if obsession_type == "contamination":
                category_message = "지금 당신이 이야기해주신 불편함은, '오염 강박'이라고 불리는 강박증의 한 유형과 비슷한 모습이에요."
            elif obsession_type == "checking":
                category_message = "지금 당신이 이야기해주신 불편함은, '확인 강박'이라고 불리는 강박증의 한 유형과 비슷한 모습이에요."
            else:
                category_message = "아쉽게도 저희 Mindit 서비스에서는 언급해주신 강박에 도움을 드릴 수 있는 기능이 없어요."
            
            return {
                "user_pattern_summary": "맞아요. 누구나 그런 생각을 할 수 있어요. 하지만 이런 생각이 너무 자주 떠오르거나, 반복되는 행동이 일상생활을 방해한다면 그건 강박적 불안을 다루는 연습이 필요하다는 신호일 수 있어요.",
                "encouragement": "하지만 걱정하지 마세요. 이를 인식하고, 조금씩 다루는 연습을 해볼 수 있어요. 제가 그 과정을 도와드릴게요 😊" if obsession_type != "other" else "하지만 걱정하지 마세요. 전문기관에 방문하여 상담 받으신다면, 금방 해결해나가실 수 있을 거예요. 사용자분의 여정을 응원합니다.",
                "category_message": category_message,
                "obsession_type": obsession_type
            }

    def generate_chat_response(self, message: str, conversation_history: List[Dict] = None) -> str:
        """
        일반적인 채팅 응답을 생성합니다.
        """
        messages = [SystemMessage(content=self.COMMON_SYSTEM_PROMPT)]
        
        #대화 히스토리가 있다면 추가
        if conversation_history:
            for hist in conversation_history[-5:]:  #최근 5개 메시지만 사용
                if hist.get("role") == "user":
                    messages.append(HumanMessage(content=hist.get("content", "")))
                elif hist.get("role") == "assistant":
                    messages.append(SystemMessage(content=hist.get("content", "")))
        
        messages.append(HumanMessage(content=message))
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"채팅 응답 생성 중 오류: {e}")
            return "죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

    def generate_obsession_analysis5_response(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        대화 히스토리를 바탕으로 사용자가 스스로 패턴을 자각하도록 돕는
        공감적 반추 질문을 한국어로 생성합니다.
        """
        system_prompt = """당신은 경험 많은 상담가입니다.
        사용자의 최근 대화를 바탕으로, 사용자가 스스로 패턴을 알아차리도록 돕는 문장을 만들어주세요.

        규칙: 예시의 분량과 형식에 맞춰 자연스럽게 질문형으로 마무리
        

        예시:
        "혹시, 방금 나눈 대화를 통해
        '내가 특정 상황에서 불안해지고,
        그것 때문에 손 씻기를 반복하는구나'
        하고 조금 더 자각이 생긴 부분이 있을까요?"
        """

        # 사용자 메시지 최근 문맥 추출
        user_messages: List[str] = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_messages.append(self._stringify_message_content(content))

        recent_context = " ".join(user_messages[-5:])
        user_prompt = f"최근 사용자 맥락: {recent_context}"

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            text = response.content.strip()
            return text
        except Exception as e:
            logger.error(f"강박 분석5 응답 생성 중 오류: {e}")
            fallback = (
                "혹시, 방금 나눈 대화를 돌아보면 특정 상황에서 불안이 올라오고, "
                "그 불안을 달래기 위해 어떤 행동을 반복하게 되는 흐름이 보일까요? "
                "조금 더 자각이 생긴 부분이 있을까요?"
            )
            return fallback

    def generate_obsession_analysis6_response(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        사용자의 최근 대화를 바탕으로 '알아가는 것이 중요하다, 함께 연습할 수 있다'는
        메시지를 LLM으로 자연스럽게 생성하고, 고정 문장을 후행으로 붙여 반환합니다.
        고정 문장: "먼저, 어떤 상황이 특히 불안했는지\n정리하며 시작해볼까요?"
        """
        system_prompt = (
            "당신은 경험 많은 상담가입니다.\n"
            "사용자의 최근 대화를 바탕으로, 사용자가 자신의 불안을 '인식하고 알아가는 것이 중요하다'는 메시지를 느낄 수 있도록 돕는 문장을 작성해주세요."
            "규칙: 예시의 분량과 형식에 맞춰 자연스럽게 작성하되 마지막에는 질문을 붙이지 말기"

            "예시:그 인식이 정말 중요해요 👏 이제 우리가 함께 그 불안을 조금씩 줄이는 연습을 시작해볼 수 있어요. " 
        )

        # 최근 사용자 맥락 생성
        user_messages: List[str] = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_messages.append(self._stringify_message_content(content))

        recent_context = " ".join(user_messages[-5:])
        user_prompt = f"최근 사용자 맥락: {recent_context}"

        closing_fixed = (
            "먼저, 어떤 상황이 특히 불안했는지 정리하며 시작해볼까요?"
        )

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = self.llm.invoke(messages)
            intro = response.content.strip()

            return f"{intro}\n\n{closing_fixed}"
        except Exception as e:
            logger.error(f"강박 분석6 응답 생성 중 오류: {e}")
            fallback_intro = (
                "지금 느끼는 불안을 알아차리고 말해주는 것 자체가 큰 시작이에요. "
                "우리는 그 과정을 함께 천천히 연습해볼 수 있어요."
            )
            return f"{fallback_intro}\n\n{closing_fixed}"


    def generate_obsession_analysis7_response(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        이전 대화내역을 받아서 불안 상황과 손 씻기 행동을 구체적으로 정리하고 7가지 예시를 제공합니다.
        """
        system_prompt = """당신은 경험 많은 상담가입니다.
        사용자의 이전 대화내역을 분석하여, 불안 상황과 손 씻기 행동을 구체적으로 정리하고 7가지 상황 예시를 생성해주세요.

        **응답 형식 (JSON):**
        {
            "intro_message": "이제, 어떤 상황에서 불안을 느끼고\n손 씻기를 하게 되는지 좀 더 구체적으로\n나눠서 정리해볼게요.",
            "question": "다음 중, 당신이 손을 씻게 되는 상황은\n어떤 게 있나요? 맞는 걸 고르거나,\n자유롭게 추가해도 좋아요.",
            "situations": [
                "~했을 때",
                "~했을 때",
                "~했을 때",
                "~했을 때",
                "~했을 때",
                "~했을 때",
                "~했을 때"
            ]
        }

        **중요한 규칙:**
        1. intro_message와 question은 고정 문장이 아닙니다. 사용자의 대화내역을 바탕으로 작성하세요.(e.g. 확인 강박일 경우, 확인과 관련된 문장이어야 함.)
        2. situations는 사용자의 대화내역을 바탕으로 7가지 구체적인 상황을 생성하세요.
        3. 각 상황은 "~했을 때" 형태로 끝나야 합니다.
        4. 사용자가 언급한 구체적인 상황들을 반영하세요.
        5. 일반적인 오염강박 상황들도 포함하세요.

        **예시 situations:**
        - "외출에서 돌아왔을 때"
        - "화장실을 다녀왔을 때"
        - "공공장소에서 물건을 만졌을 때"
        - "음식을 만지기 전에"
        - "침대에 누우기 전에"
        - "다른 사람과 악수를 한 후에"
        - "돈을 만진 후에"

        **사용자 맞춤화:**
        - 사용자가 언급한 특정 상황이 있다면 그것을 우선적으로 포함하세요.
        - 사용자의 생활 패턴이나 언급한 장소를 반영하세요."""

        # 대화 히스토리에서 사용자 메시지 추출
        user_messages = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_messages.append(self._stringify_message_content(content))

        # 전체 사용자 메시지들을 하나의 텍스트로 결합
        full_context = " ".join(user_messages)

        user_prompt = f"대화 히스토리: {full_context}"

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # JSON 파싱 시도
            import json
            try:
                # JSON 부분만 추출
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    return result
            except:
                pass
            
            # JSON 파싱 실패 시 기본값
            return {
                "intro_message": "이제, 어떤 상황에서 불안을 느끼고\n손 씻기를 하게 되는지 좀 더 구체적으로\n나눠서 정리해볼게요.",
                "question": "다음 중, 당신이 손을 씻게 되는 상황은\n어떤 게 있나요? 맞는 걸 고르거나,\n자유롭게 추가해도 좋아요.",
                "situations": [
                    "외출에서 돌아왔을 때",
                    "화장실을 다녀왔을 때",
                    "공공장소에서 물건을 만졌을 때",
                    "음식을 만지기 전에",
                    "침대에 누우기 전에",
                    "다른 사람과 악수를 한 후에",
                    "돈을 만진 후에"
                ]
            }
            
        except Exception as e:
            logger.error(f"강박 분석7 응답 생성 중 오류: {e}")
            return {
                "intro_message": "이제, 어떤 상황에서 불안을 느끼고\n손 씻기를 하게 되는지 좀 더 구체적으로\n나눠서 정리해볼게요.",
                "question": "다음 중, 당신이 손을 씻게 되는 상황은\n어떤 게 있나요? 맞는 걸 고르거나,\n자유롭게 추가해도 좋아요.",
                "situations": [
                    "외출에서 돌아왔을 때",
                    "화장실을 다녀왔을 때",
                    "공공장소에서 물건을 만졌을 때",
                    "음식을 만지기 전에",
                    "침대에 누우기 전에",
                    "다른 사람과 악수를 한 후에",
                    "돈을 만진 후에"
                ]
            }

    def generate_obsession_analysis8_response(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        사용자가 상황별 불안 점수를 0-100점으로 매기도록 하는 메시지를 생성합니다.
        """
        system_prompt = """당신은 경험 많은 상담가입니다.
        사용자의 이전 대화내역을 분석하여, 상황별 불안 점수를 매기도록 하는 메시지를 생성해주세요.

        **응답 형식:**
        "좋아요. 그럼 지금 말씀해주신 각각의 상황에 대해,
        불안하거나 찝찝한 느낌이 얼마나 강한지
        0~100점으로 표현해볼 수 있을까요?

        예를 들어:
        [사용자가 언급한 구체적인 상황 3개를 예시로 제시]
        이런 식으로요."

        **중요한 규칙:**
        1. 고정 문장을 사용하되, 예시 부분만 사용자의 대화내역을 반영하세요.
        2. 사용자가 언급한 구체적인 상황 3개를 예시로 제시하세요.
        3. 각 예시는 "상황명 → 점수" 형태로 작성하세요.
        4. 점수는 0~100점 범위 내에서 적절하게 설정하세요.
        5. 사용자가 언급하지 않은 상황이 있다면 일반적인 오염강박 상황으로 보완하세요.

        **예시:**
        - "지하철 손잡이 → 90점"
        - "화장실 문 → 100점" 
        - "외출 후 귀가 → 70점"

        **사용자 맞춤화:**
        - 사용자가 언급한 구체적인 상황을 우선적으로 포함하세요.
        - 사용자의 생활 패턴이나 언급한 장소를 반영하세요."""

        # 대화 히스토리에서 사용자 메시지 추출
        user_messages = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_messages.append(self._stringify_message_content(content))

        # 전체 사용자 메시지들을 하나의 텍스트로 결합
        full_context = " ".join(user_messages)

        user_prompt = f"대화 히스토리: {full_context}"

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"강박 분석10 응답 생성 중 오류: {e}")
            return ("좋아요. 그럼 지금 말씀해주신 각각의 상황에 대해,\n"
                   "불안하거나 찝찝한 느낌이 얼마나 강한지\n"
                   "0~100점으로 표현해볼 수 있을까요?\n\n"
                   "예를 들어:\n"
                   "지하철 손잡이 → 90점\n"
                   "화장실 문 → 100점\n"
                   "외출 후 귀가 → 70점\n"
                   "이런 식으로요.")

    def generate_obsession_analysis9_response(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        점수 데이터를 바탕으로 불안 위계표를 만들고 연습 계획을 제시하는 메시지를 생성합니다.
        """
        # 대화 히스토리에서 불안 점수 데이터 추출
        anxiety_scores = self._parse_anxiety_scores(conversation_history)
        
        if not anxiety_scores:
            # 점수 데이터가 없으면 기본 응답
            return {
                "intro_message": "정말 잘하셨어요! 👏 이제, 이 점수를 기준으로 '덜 불안한 상황 → 가장 불안한 상황' 순서대로 불안 위계표를 만들어볼게요.",
                "anxiety_hierarchy": [],
                "practice_message": "이제 이 표를 바탕으로, 불안도가 낮은 위계부터 차근차근 연습해볼 거예요.",
                "example_message": "예를 들어, 가장 불안도가 낮은 상황을 잡은 뒤, 확인하지 않고 5분 버티기 같은 작은 실천부터요.",
                "support_message": "버티는 동안 불안이 생기면 제가 함께 그 불안을 다룰 수 있도록 도와드릴 거예요."
            }
        
        # 점수 순서대로 정렬 (낮은 점수부터)
        sorted_scores = sorted(anxiety_scores, key=lambda x: x['score'])
        
        # 위계표 생성
        hierarchy = []
        for i, item in enumerate(sorted_scores, 1):
            hierarchy.append({
                "order": i,
                "situation": item['situation'],
                "score": item['score']
            })
        
        # 가장 낮은 점수 상황을 반영한 자연스러운 예시 메시지 생성
        if sorted_scores:
            lowest_situation = sorted_scores[0]['situation']
            example_message = self._generate_example_message(lowest_situation, conversation_history)
        else:
            example_message = "예를 들어, 확인하지 않고 5분 버티기 같은 작은 실천부터요."
        
        return {
            "intro_message": "정말 잘하셨어요! 👏 이제, 이 점수를 기준으로 '덜 불안한 상황 → 가장 불안한 상황' 순서대로 불안 위계표를 만들어볼게요.",
            "anxiety_hierarchy": hierarchy,
            "practice_message": "이제 이 표를 바탕으로, 불안도가 낮은 위계부터 차근차근 연습해볼 거예요.",
            "example_message": example_message,
            "support_message": "버티는 동안 불안이 생기면 제가 함께 그 불안을 다룰 수 있도록 도와드릴 거예요."
        }
    
    def _generate_example_message(self, lowest_situation: str, conversation_history: List[Dict[str, Any]]) -> str:
        """가장 낮은 점수 상황을 반영한 자연스러운 예시 메시지를 생성합니다."""
        system_prompt = """당신은 경험 많은 상담가입니다.
        사용자의 가장 낮은 불안 점수 상황을 바탕으로, 자연스럽고 구체적인 연습 예시 메시지를 생성해주세요.

        **응답 형식:**
        "예를 들어, [구체적인 연습 방법] 같은 작은 실천부터요."

        **중요한 규칙:**
        1. 주어진 상황에 맞는 구체적인 연습 방법을 제시하세요.
        2. 상황에 따라 적절한 행동을 선택하세요:
           - 확인 강박: "확인하지 않고 X분 버티기"
           - 손 씻기 강박: "손을 씻지 않고 X분 버티기"
           - 기타 강박: "그 행동을 하지 않고 X분 버티기"
        3. 시간은 3-10분 사이의 적절한 시간으로 설정하세요.
        4. 자연스럽고 격려하는 톤으로 작성하세요.
        5. 전체 메시지는 50자 이내로 작성하세요.

        **예시:**
        - "가스 밸브를 잠갔는지 확인했을 때" → "예를 들어, 가스 밸브를 확인하지 않고 5분 버티기 같은 작은 실천부터요."
        - "손을 씻었을 때" → "예를 들어, 손을 씻지 않고 3분 버티기 같은 작은 실천부터요."
        - "문을 잠갔을 때" → "예를 들어, 문을 확인하지 않고 5분 버티기 같은 작은 실천부터요."
        """
        
        user_prompt = f"가장 낮은 점수 상황: {lowest_situation}"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"예시 메시지 생성 중 오류: {e}")
            # 기본값 반환
            if "확인" in lowest_situation:
                return "예를 들어, 확인하지 않고 5분 버티기 같은 작은 실천부터요."
            elif "씻" in lowest_situation or "손" in lowest_situation:
                return "예를 들어, 손을 씻지 않고 3분 버티기 같은 작은 실천부터요."
            else:
                return "예를 들어, 그 행동을 하지 않고 5분 버티기 같은 작은 실천부터요."
    
    def _parse_anxiety_scores(self, conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """대화내역에서 불안 점수 데이터를 추출합니다."""
        scores = []
        
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                
                # JSON 형태의 점수 데이터인지 확인
                if isinstance(content, dict) and content.get("type") == "anxiety_scores":
                    data = content.get("data", {})
                    situations = data.get("situations", [])
                    for item in situations:
                        scores.append({
                            "situation": item.get("situation", ""),
                            "score": item.get("score", 0)
                        })
                
                # 텍스트 형태의 점수 데이터인지 확인 (예: "상황: 90점" 형태)
                elif isinstance(content, str) and "점" in content:
                    # 정규식으로 점수 추출
                    import re
                    pattern = r'([^:]+):\s*(\d+)점'
                    matches = re.findall(pattern, content)
                    for situation, score in matches:
                        scores.append({
                            "situation": situation.strip(),
                            "score": int(score)
                        })
        
        return scores

    def generate_obsession_report(self, conversation_history: List[Dict[str, Any]], session_id: str = None) -> Dict[str, Any]:
        """
        전체 대화내역을 기반으로 강박 유형, 사고, 행동, 불안 위계를 정리한 리포트를 생성합니다.
        """
        # 실제 점수 데이터 파싱
        anxiety_scores = self._parse_anxiety_scores(conversation_history)
        
        # 점수 순서대로 정렬 (내림차순 - 높은 점수부터)
        sorted_scores = sorted(anxiety_scores, key=lambda x: x['score'], reverse=True)
        
        # 위계표 생성 (최대 4개)
        hierarchy = []
        for i, item in enumerate(sorted_scores[:4], 1):
            hierarchy.append({
                "situation": item['situation'],
                "score": item['score']
            })
        
        # 저장된 강박 유형 사용 (없으면 새로 분류)
        if session_id and session_id in self.session_obsession_types:
            obsession_type = self.session_obsession_types[session_id]
        else:
            obsession_type = self.categorize_obsession_type(conversation_history)
        
        # 강박 유형을 한국어로 변환
        korean_obsession_type = self._convert_obsession_type_to_korean(obsession_type)
        
        # LLM으로 사고와 행동 패턴 추출
        anxiety_thoughts, compulsive_behaviors = self._extract_patterns_with_llm(conversation_history, obsession_type)
        
        return {
            "obsession_type": korean_obsession_type,
            "anxiety_thoughts": anxiety_thoughts,
            "compulsive_behaviors": compulsive_behaviors,
            "anxiety_hierarchy": hierarchy
        }
    
    def _convert_obsession_type_to_korean(self, obsession_type: str) -> str:
        type_mapping = {
            "contamination": "오염강박",
            "checking": "확인강박", 
            "other": "기타"
        }
        return type_mapping.get(obsession_type, "기타")
    
    def _extract_obsession_type_from_history(self, conversation_history: List[Dict[str, Any]]) -> str:
        """대화내역에서 이미 분류된 강박 유형을 추출합니다."""
        # analyze4에서 이미 분류된 결과가 있는지 확인
        for msg in conversation_history:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, dict):
                    # analyze4 응답에서 강박 유형 추출
                    if "obsession_type" in content:
                        return content["obsession_type"]
                    elif "category_message" in content:
                        category_msg = content["category_message"]
                        if "오염 강박" in category_msg:
                            return "오염강박"
                        elif "확인 강박" in category_msg:
                            return "확인강박"
                        else:
                            return "기타"
        
        # analyze4 결과가 없으면 키워드 기반으로 분류
        return self._classify_obsession_type_by_keywords(conversation_history)
    
    def _classify_obsession_type_by_keywords(self, conversation_history: List[Dict[str, Any]]) -> str:
        """키워드 기반으로 강박 유형을 분류합니다."""
        # 대화 히스토리에서 사용자 메시지 추출
        user_messages = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_messages.append(self._stringify_message_content(content))
        
        full_context = " ".join(user_messages).lower()
        
        # 오염강박 키워드
        contamination_keywords = ["손 씻기", "손씻기", "세균", "더러움", "오염", "청소", "화장실", "지하철", "손잡이", "씻"]
        # 확인강박 키워드
        checking_keywords = ["확인", "문 잠금", "가스", "전자제품", "위치", "시간", "체크", "잠갔", "껐"]
        
        contamination_score = sum(1 for keyword in contamination_keywords if keyword in full_context)
        checking_score = sum(1 for keyword in checking_keywords if keyword in full_context)
        
        if contamination_score > checking_score:
            return "오염강박"
        elif checking_score > contamination_score:
            return "확인강박"
        else:
            return "기타"
    
    def _extract_patterns_from_history(self, conversation_history: List[Dict[str, Any]]) -> tuple[str, str]:
        """대화내역에서 불안 사고와 강박 행동 패턴을 추출합니다."""
        # 대화 히스토리에서 사용자 메시지 추출
        user_messages = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_messages.append(self._stringify_message_content(content))
        
        full_context = " ".join(user_messages)
        
        # 간단한 패턴 추출 (LLM 없이)
        anxiety_thoughts = "사용자가 언급한 구체적인 불안 사고가 확인되지 않았습니다."
        compulsive_behaviors = "반복적인 행동 패턴이 확인되지 않았습니다."
        
        # 확인 강박 관련 패턴
        if any(keyword in full_context for keyword in ["확인", "잠갔", "껐", "체크"]):
            compulsive_behaviors = "반복적인 확인 행동"
            anxiety_thoughts = "확인하지 않으면 안전하지 않을 것이라는 불안"
        
        # 손 씻기 강박 관련 패턴
        elif any(keyword in full_context for keyword in ["손 씻기", "손씻기", "씻", "세균", "더러움"]):
            compulsive_behaviors = "반복적인 손 씻기"
            anxiety_thoughts = "오염이나 세균에 대한 불안"
        
        return anxiety_thoughts, compulsive_behaviors
    
    def _extract_patterns_with_llm(self, conversation_history: List[Dict[str, Any]], obsession_type: str) -> tuple[str, str]:
        """LLM을 사용해서 불안 사고와 강박 행동 패턴을 추출합니다."""
        system_prompt = f"""당신은 경험 많은 상담가입니다.
        사용자의 대화내역을 분석하여 구체적인 불안 사고와 강박 행동 패턴을 추출해주세요.
        
        **강박 유형: {obsession_type}**
        
        **응답 형식 (JSON):**
        {{
            "anxiety_thoughts": "사용자가 언급한 구체적인 불안 사고 내용",
            "compulsive_behaviors": "사용자가 언급한 반복 행동 패턴"
        }}
        
        **중요한 규칙:**
        1. anxiety_thoughts: 강박 유형에 따른 일반적인 불안 사고 패턴을 범위적으로 작성
           - 오염강박: "공공 물건을 만지면 병균이나 더러움이 손에 묻었다는 강한 확신"
           - 확인강박: "시간이나 위치를 당장 확인하지 않으면 늦을 것이라는 생각"
           - 기타: "특정 행동을 하지 않으면 나쁜 일이 일어날 것이라는 불안"
        2. compulsive_behaviors: 사용자가 실제로 언급한 반복 행동을 구체적으로 정리
           - 예: "반복적인 손 씻기"
           - 예: "문, 가스, 전자제품 확인을 반복"
        3. anxiety_thoughts는 구체적인 상황보다는 일반적인 사고 패턴으로 작성하세요.
        4. compulsive_behaviors는 사용자가 실제로 말한 내용을 바탕으로 구체적으로 작성하세요.
        5. 내용이 없으면 "확인되지 않았습니다"로 표시하세요."""

        # 대화 히스토리에서 사용자 메시지 추출
        user_messages = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_messages.append(self._stringify_message_content(content))
        
        full_context = " ".join(user_messages)
        user_prompt = f"대화내역: {full_context}"

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # JSON 파싱 시도
            import json
            try:
                # JSON 부분만 추출
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    return (
                        result.get("anxiety_thoughts", "사용자가 언급한 구체적인 불안 사고가 확인되지 않았습니다."),
                        result.get("compulsive_behaviors", "반복적인 행동 패턴이 확인되지 않았습니다.")
                    )
            except:
                pass
            
            # JSON 파싱 실패 시 기본값
            return (
                "사용자가 언급한 구체적인 불안 사고가 확인되지 않았습니다.",
                "반복적인 행동 패턴이 확인되지 않았습니다."
            )
            
        except Exception as e:
            logger.error(f"패턴 추출 중 오류: {e}")
            return (
                "사용자가 언급한 구체적인 불안 사고가 확인되지 않았습니다.",
                "반복적인 행동 패턴이 확인되지 않았습니다."
            )

    def generate_obsession_report2(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        대화내역을 기반으로 맞춤형 5주차 강박개선 플랜을 생성합니다.
        """
        from datetime import datetime, timedelta
        
        # 현재 날짜를 기준으로 5주차 날짜 계산
        start_date = datetime.now()
        weeks = []
        for i in range(5):
            week_start = start_date + timedelta(weeks=i)
            week_end = week_start + timedelta(days=6)
            weeks.append({
                "week": i + 1,
                "start_date": week_start.strftime("%Y-%m-%d"),
                "end_date": week_end.strftime("%Y-%m-%d")
            })
        
        # 대화 히스토리에서 강박 유형 분석
        obsession_type = self._analyze_obsession_type_for_report(conversation_history)
        
        # 강박 유형에 따른 훈련 계획 생성
        if obsession_type == "오염강박":
            training_plan = self._generate_contamination_training_plan(conversation_history, weeks)
            training_explanation = self._get_contamination_training_explanation()
        else:  # 확인강박
            training_plan = self._generate_checking_training_plan(conversation_history, weeks)
            training_explanation = self._get_checking_training_explanation()
        
        return {
            "obsession_type": obsession_type,
            "training_explanation": training_explanation,
            "weeks": weeks,
            "training_plan": training_plan
        }
    
    def _analyze_obsession_type_for_report(self, conversation_history: List[Dict[str, Any]]) -> str:
        """대화내역을 분석하여 강박 유형을 판단합니다."""
        # 대화 히스토리에서 사용자 메시지 추출
        user_messages = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_messages.append(self._stringify_message_content(content))
        
        full_context = " ".join(user_messages).lower()
        
        # 오염강박 키워드
        contamination_keywords = ["손 씻기", "손씻기", "세균", "더러움", "오염", "청소", "화장실", "지하철", "손잡이"]
        # 확인강박 키워드
        checking_keywords = ["확인", "문 잠금", "가스", "전자제품", "위치", "시간", "체크"]
        
        contamination_score = sum(1 for keyword in contamination_keywords if keyword in full_context)
        checking_score = sum(1 for keyword in checking_keywords if keyword in full_context)
        
        if contamination_score > checking_score:
            return "오염강박"
        elif checking_score > contamination_score:
            return "확인강박"
        else:
            return "오염강박"  # 기본값
    
    def _generate_contamination_training_plan(self, conversation_history: List[Dict[str, Any]], weeks: List[Dict]) -> List[Dict[str, Any]]:
        """오염강박용 5주차 훈련 계획을 생성합니다."""
        system_prompt = """당신은 경험 많은 상담가입니다.
        사용자의 대화내역을 분석하여 오염강박용 맞춤형 훈련 계획을 생성해주세요.

        **훈련명 (고정):**
        1주차: 사진 노출 훈련
        2주차: 낮은 수준의 AR 노출 훈련
        3주차: 중간 수준의 AR 노출 훈련
        4주차: 높은 수준의 AR 노출 훈련
        5주차: 노출 훈련 반복과 실생활 적용

        **응답 형식 (JSON):**
        [
            {
                "week": 1,
                "training_name": "사진 노출 훈련",
                "goal": "사용자 맞춤 목표 (약 20자)",
                "content": "사용자 맞춤 내용 (약 50자)"
            },
            ...
        ]

        **중요한 규칙:**
        1. goal은 사용자의 구체적인 상황을 반영하여 20자 내외로 작성
        2. content는 사용자가 언급한 구체적인 상황을 포함하여 50자 내외로 작성
        3. 사용자가 언급한 구체적인 장소나 상황을 반영하세요.
        4. 예시: "AI 사진 노출: 지하철 손잡이를 잡는 손의 사진으로 사진 노출 훈련 3회"
        """
        
        # 대화 히스토리에서 사용자 메시지 추출
        user_messages = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_messages.append(self._stringify_message_content(content))
        
        full_context = " ".join(user_messages)
        user_prompt = f"대화 히스토리: {full_context}"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # JSON 파싱 시도
            import json
            try:
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    return result
            except:
                pass
            
            # JSON 파싱 실패 시 기본값
            return self._get_default_contamination_plan(weeks)
            
        except Exception as e:
            logger.error(f"오염강박 훈련 계획 생성 중 오류: {e}")
            return self._get_default_contamination_plan(weeks)
    
    def _generate_checking_training_plan(self, conversation_history: List[Dict[str, Any]], weeks: List[Dict]) -> List[Dict[str, Any]]:
        """확인강박용 5주차 훈련 계획을 생성합니다."""
        system_prompt = """당신은 경험 많은 상담가입니다.
        사용자의 대화내역을 분석하여 확인강박용 맞춤형 훈련 계획을 생성해주세요.

        **훈련명 (고정):**
        1주차: 기초 상상 노출 훈련
        2주차: 구체적 상황의 상상 노출 훈련
        3주차: 결과까지 상상하는 노출 훈련
        4주차: 실제 노출 훈련1
        5주차: 실제 노출 훈련2

        **응답 형식 (JSON):**
        [
            {
                "week": 1,
                "training_name": "기초 상상 노출 훈련",
                "goal": "사용자 맞춤 목표 (약 20자)",
                "content": "사용자 맞춤 내용 (약 50자)"
            },
            ...
        ]

        **중요한 규칙:**
        1. goal은 사용자의 구체적인 상황을 반영하여 20자 내외로 작성
        2. content는 사용자가 언급한 구체적인 상황을 포함하여 50자 내외로 작성
        3. 사용자가 언급한 구체적인 확인 행동을 반영하세요.
        """
        
        # 대화 히스토리에서 사용자 메시지 추출
        user_messages = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_messages.append(self._stringify_message_content(content))
        
        full_context = " ".join(user_messages)
        user_prompt = f"대화 히스토리: {full_context}"
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # JSON 파싱 시도
            import json
            try:
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    return result
            except:
                pass
            
            # JSON 파싱 실패 시 기본값
            return self._get_default_checking_plan(weeks)
            
        except Exception as e:
            logger.error(f"확인강박 훈련 계획 생성 중 오류: {e}")
            return self._get_default_checking_plan(weeks)
    
    def _get_default_contamination_plan(self, weeks: List[Dict]) -> List[Dict[str, Any]]:
        """오염강박 기본 훈련 계획을 반환합니다."""
        return [
            {
                "week": 1,
                "training_name": "사진 노출 훈련",
                "goal": "나의 강박과 불안 관찰하고 익숙해지기",
                "content": "AI 사진 노출: 지하철 손잡이를 잡는 손의 사진으로 사진 노출 훈련 3회"
            },
            {
                "week": 2,
                "training_name": "낮은 수준의 AR 노출 훈련",
                "goal": "낮은 수준의 불안 상황에 익숙해지기",
                "content": "AR 노출: 공공장소 물건 만지는 상황을 AR로 체험하며 노출 훈련 3회"
            },
            {
                "week": 3,
                "training_name": "중간 수준의 AR 노출 훈련",
                "goal": "중간 수준의 불안 상황에 도전하기",
                "content": "AR 노출: 화장실 문고리를 만지는 상황을 AR로 체험하며 노출 훈련 3회"
            },
            {
                "week": 4,
                "training_name": "높은 수준의 AR 노출 훈련",
                "goal": "높은 수준의 불안 상황에 도전하기",
                "content": "AR 노출: 가장 불안한 상황을 AR로 체험하며 노출 훈련 3회"
            },
            {
                "week": 5,
                "training_name": "노출 훈련 반복과 실생활 적용",
                "goal": "실생활에서 훈련 내용 적용하기",
                "content": "실생활 적용: 훈련한 내용을 실제 상황에서 적용하며 점진적으로 확장"
            }
        ]
    
    def _get_default_checking_plan(self, weeks: List[Dict]) -> List[Dict[str, Any]]:
        """확인강박 기본 훈련 계획을 반환합니다."""
        return [
            {
                "week": 1,
                "training_name": "기초 상상 노출 훈련",
                "goal": "나의 강박과 불안 관찰하고 익숙해지기",
                "content": "상상 노출: 확인하지 않고 버티는 상황을 상상하며 노출 훈련 3회"
            },
            {
                "week": 2,
                "training_name": "구체적 상황의 상상 노출 훈련",
                "goal": "구체적 상황에서의 불안에 익숙해지기",
                "content": "상상 노출: 문을 잠그지 않고 버티는 구체적 상황을 상상하며 훈련 3회"
            },
            {
                "week": 3,
                "training_name": "결과까지 상상하는 노출 훈련",
                "goal": "최악의 결과까지 상상하며 불안에 익숙해지기",
                "content": "상상 노출: 확인하지 않았을 때의 결과까지 상상하며 노출 훈련 3회"
            },
            {
                "week": 4,
                "training_name": "실제 노출 훈련1",
                "goal": "실제 상황에서 낮은 수준의 노출 훈련",
                "content": "실제 노출: 실제로 확인하지 않고 짧은 시간 버티기 훈련 3회"
            },
            {
                "week": 5,
                "training_name": "실제 노출 훈련2",
                "goal": "실제 상황에서 높은 수준의 노출 훈련",
                "content": "실제 노출: 실제로 확인하지 않고 더 긴 시간 버티기 훈련 3회"
            }
        ]
    
    def _get_contamination_training_explanation(self) -> Dict[str, str]:
        """오염강박용 훈련 설명을 반환합니다."""
        return {
            "response_prevention": {
                "title": "반응방지 훈련이란?",
                "description": "강박이 떠오를 때, 바로 그 순간! 그 행동을 하지 않고 그대로 버텨보는 훈련이에요.",
                "when": "언제 하나요?",
                "instruction": "강박이 발생하는 순간에 바로\n반응 방지를 실행해주세요."
            },
            "ai_ar_training": {
                "title": "AI · AR 훈련이란?",
                "description": "불안을 유발하는 사진이나 장면을\nAI를 통해 생성하고, 미리 정한 계획에 따라 일부러 마주해보는 연습이에요.",
                "when": "언제 하나요?",
                "instruction": "계획된 플랜에 따라 일주일에 정해진 횟수만큼 노출 훈련을 진행합니다."
            }
        }
    
    def _get_checking_training_explanation(self) -> Dict[str, str]:
        """확인강박용 훈련 설명을 반환합니다."""
        return {
            "response_prevention": {
                "title": "반응방지 훈련이란?",
                "description": "강박이 떠오를 때, 바로 그 순간! 그 행동을 하지 않고 그대로 버텨보는 훈련이에요.",
                "when": "언제 하나요?",
                "instruction": "강박이 발생하는 순간에 바로\n반응 방지를 실행해주세요."
            },
            "imagination_training": {
                "title": "상상 노출 훈련이란?",
                "description": "불안을 유발하는 사진이나 장면을\nAI를 통해 생성하고, 미리 정한 계획에 따라 일부러 마주해보는 연습이에요.",
                "when": "언제 하나요?",
                "instruction": "계획된 플랜에 따라 일주일에 정해진 횟수만큼 노출 훈련을 진행합니다."
            }
        }