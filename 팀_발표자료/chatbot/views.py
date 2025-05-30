import logging
import json
import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect
from openai import OpenAI
from community.templatetags.community_filters import filter_curse

# 로깅 설정
logger = logging.getLogger(__name__)

# OpenAI 클라이언트 초기화 (환경 변수 사용 권장)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def chatbot_page(request):
    """챗봇 페이지를 렌더링"""
    return render(request, 'chatbot.html')

@csrf_protect
def filter_message(request):
    """사용자 메시지를 받아 욕설 필터링 후 반환"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')

            logger.info(f"Filtering message: {user_message}")
            filtered_message = filter_curse(user_message)
            logger.debug(f"Filtered result: {filtered_message}")

            return JsonResponse({
                'status': 'success',
                'message': filtered_message
            })
        except Exception as e:
            logger.error(f"Filter Error: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f"필터링 오류: {str(e)}"
            }, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

@csrf_protect
def chat_response(request):
    """사용자 메시지를 받아 필터링 후 챗봇 응답 생성"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')

            # 사용자 입력 로깅
            logger.info(f"User input: {user_message}")

            # 욕설 필터링
            filtered_message = filter_curse(user_message)
            logger.debug(f"Filtered message: {filtered_message}")

            # OpenAI API 호출
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 금융 전문 어시스턴트입니다. "
                            "주식, 투자, 경제, 금융상품, 부동산, 재테크, 가계재무, 금융시장 등 금융과 관련된 모든 주제에 대해 전문적이고 실용적인 답변을 제공합니다. "
                            "금융과 무관한 질문(예: 요리, 여행, 게임 등)에는 '죄송합니다만, 저는 금융 관련 문의사항에 대해서만 도움을 드릴 수 있습니다.'라고 답변하세요. "
                            "모든 답변은 한국어로, 간결하고 실용적으로, 사용자가 이해하기 쉽게 설명하세요."
                        )
                    },
                    {"role": "user", "content": filtered_message}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            # 챗봇 응답 추출 (필터링 없이 그대로)
            bot_response = response.choices[0].message.content
            logger.debug(f"Bot response: {bot_response}")

            return JsonResponse({
                'status': 'success',
                'filtered_message': filtered_message,
                'bot_response': bot_response
            })

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': f"서버 오류: {str(e)}"
            }, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)