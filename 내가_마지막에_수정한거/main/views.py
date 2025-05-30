# main/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
import requests
import json
from django.views.decorators.http import require_http_methods
from community.models import AdminBoard  # AdminBoard 모델 임포트

#add1
def main_coalition(request):
    return render(request, 'main_coalition.html')
def main_faq(request):
    return render(request, 'main_faq.html')
def main_advertise(request):
    return render(request, 'main_advertise.html')

def admin_board_detail(request, pk):
    """
    AdminBoard의 상세 페이지 뷰.
    pk로 특정 게시글을 조회하고, is_visible=True인 경우에만 표시.
    """
    post = get_object_or_404(AdminBoard, pk=pk, is_visible=True)
    context = {
        'post': post,
    }
    return render(request, 'community_admin_content.html', context)  # 템플릿 변경
#add2

def main(request):
    # is_visible=True인 AdminBoard 데이터 가져오기
    admin_posts = AdminBoard.objects.filter(is_visible=True).select_related('user')[:3]
    context = {
        'admin_posts': admin_posts,
    }
    return render(request, 'main.html', context)

def get_naver_news(request):
    client_id = "OCRZok3QLNl9VF2e0Uo_"
    client_secret = "djBL9xrZIM"
    
    # 증시 관련 검색어
    query = "증시 OR 주식 OR 코스피 OR 코스닥"
    
    # 네이버 뉴스 검색 API 호출
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }
    params = {
        "query": query,
        "display": 5,  # 5개로 변경
        "sort": "date"  # 최신순 정렬 (상=>하)
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # 에러 발생시 예외 발생
        news_data = response.json()
        
        # 뉴스 데이터 가공
        news_list = []
        for item in news_data.get('items', []):
            news_list.append({
                'title': item['title'].replace('<b>', '').replace('</b>', ''),
                'link': item['link'],
                'description': item['description'].replace('<b>', '').replace('</b>', ''),
                'pubDate': item['pubDate']
            })
        
        return JsonResponse({'news': news_list})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)