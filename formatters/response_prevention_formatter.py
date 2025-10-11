from typing import Dict, Any

def format_response_prevention_analysis(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    반응방지 분석 결과를 API 응답 형식에 맞게 포맷팅합니다.
    """
    return {
        "obsession_type": raw_response.get("obsession_type", "기타"),
        "content": raw_response.get("content", {
            "encouragement_message": "반응 방지를 잘 마치셨네요. 고생 많았어요! ☺️",
            "cognitive_error_explanation": "",
            "personalized_situation": "",
            "emotion_question": ""
        })
    }
