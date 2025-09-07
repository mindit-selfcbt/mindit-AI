from typing import Dict, Any, List

def format_obsession_question(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM이 반환한 강박 분석 질문을 정형화된 형식으로 가공합니다.
    """
    question = raw_response.get("question", "")
    choices = raw_response.get("choices", [])
    
    # 질문에 선택지 안내 문구 추가
    if question and not question.endswith("?"):
        question += "?"
    question += " 아래 선택지를 고르거나 직접 작성해주세요."
    
    # 선택지 형식 통일
    formatted_choices = []
    for choice in choices[:3]:
        choice = choice.strip()
        if not choice.endswith("때"):
            choice += "할 때"
        formatted_choices.append(choice)
    
    return {
        "question": question,
        "choices": formatted_choices
    }

def format_chat_response(raw_response: str) -> str:
    """
    일반 채팅 응답을 상담가 스타일로 포맷팅합니다.
    """
    # 기본 상담가 스타일 접두사/접미사 추가
    if not raw_response.startswith("안녕하세요") and not raw_response.startswith("말씀해주신"):
        raw_response = f"말씀해주신 내용을 잘 들었습니다. {raw_response}"
    
    if not raw_response.endswith(".") and not raw_response.endswith("?"):
        raw_response += "."
    
    return raw_response 