# predict_info/urls.py
from django.urls import path
from . import views

app_name = 'predict_info' # 앱 네임스페이스 설정

urlpatterns = [
    # predict_info.html을 렌더링하는 뷰
    path('', views.predict_info_view, name='predict_info_page'),
    # AJAX를 통해 특정 종목의 주가를 예측하는 API 엔드포인트
    path('predict_stock_ajax/', views.predict_stock_price_ajax, name='predict_stock_price_ajax'),
    # 자동완성 검색을 위한 API 엔드포인트 (예측 페이지용)
    path('search_stocks_ajax/', views.search_stocks_ajax, name='search_stocks_ajax'),
    # 관심 종목 추가/삭제를 위한 API 엔드포인트 (이 부분이 추가되어야 합니다)
    path('toggle_favorite_ajax/', views.toggle_favorite_stock_ajax, name='toggle_favorite_stock_ajax'),
]
