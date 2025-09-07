import uuid
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

class ChatbotService:
    # 공통 시스템 프롬프트
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
        
        # 대화 히스토리에서 사용자 메시지 추출
        user_messages = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                user_messages.append(msg.get("content", ""))
        
        # 최근 사용자 메시지들을 하나의 텍스트로 결합
        recent_context = " ".join(user_messages[-3:])  # 최근 3개 메시지만 사용
        
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
        
        # 대화 히스토리에서 사용자 메시지 추출
        user_messages = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                user_messages.append(msg.get("content", ""))
        
        # 최근 사용자 메시지들을 하나의 텍스트로 결합
        recent_context = " ".join(user_messages[-5:])  # 최근 5개 메시지 사용
        
        user_prompt = f"대화 히스토리: {recent_context}"
        
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

    def generate_chat_response(self, message: str, conversation_history: List[Dict] = None) -> str:
        """
        일반적인 채팅 응답을 생성합니다.
        """
        messages = [SystemMessage(content=self.COMMON_SYSTEM_PROMPT)]
        
        # 대화 히스토리가 있다면 추가
        if conversation_history:
            for hist in conversation_history[-5:]:  # 최근 5개 메시지만 사용
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