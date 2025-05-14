# predict_info/urls.py
from django.urls import path 
from . import views

app_name = 'predict_info' # 앱 네임스페이스 설정

urlpatterns = [
    # predict_info.html을 렌더링하고 초기 예측을 수행하는 뷰
    path('', views.predict_info_view, name='predict_info_page'), # main_footer.html의 링크와 맞추기 위해 'predict_info' 대신 'predict_info_page' 사용 (또는 main_footer 수정)
    # AJAX를 통해 특정 종목의 주가를 예측하는 API 엔드포인트
    path('predict_stock_ajax/', views.predict_stock_price_ajax, name='predict_stock_price_ajax'),
]
