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
           - 안전에 대한 반복적 확인
           - 예: "문을 잠갔는지 계속 확인해요", "가스를 끄지 않았나 걱정돼요"
        
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
            
            # 유효한 카테고리인지 확인
            if category in ["contamination", "checking", "other"]:
                return category
            else:
                logger.warning(f"예상치 못한 카테고리 반환: {category}, 기본값 'other' 사용")
                return "other"
                
        except Exception as e:
            logger.error(f"강박 카테고리 분류 중 오류: {e}")
            return "other"

    def generate_obsession_analysis4_response(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        대화 히스토리를 분석하여 강박 유형별 맞춤 응답을 생성합니다.
        """
        # 1단계: 강박 유형 카테고리화
        obsession_type = self.categorize_obsession_type(conversation_history)
        
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