from django.urls import path
from . import views

app_name = 'chart' # 앱 네임스페이스

urlpatterns = [
    path('', views.chart_view, name='chart'), # 이름 변경 (predict_info와 충돌 방지)
    path('api/realtime-price/', views.get_realtime_price, name='realtime_price_chart'), # 이름 변경
    # 차트 페이지 자동완성 검색을 위한 API 엔드포인트 추가
    path('api/search_stocks_chart/', views.search_stocks_ajax_for_chart, name='search_stocks_ajax_chart'),
]
