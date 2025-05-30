# mypage/urls.py
from django.urls import path
from . import views

app_name = 'mypage'

urlpatterns = [
    path('', views.mypage_view, name='mypage'),
    path('edit/', views.edit_profile_view, name='edit_profile'),
    path('update-greeting/', views.update_greeting_message, name='update_greeting_message'),
    path('unblock/<int:blocked_id>/', views.unblock_user, name='unblock_user'),
    # --- 관심 종목 예측 데이터 AJAX URL 추가 ---
    path('get_favorite_prediction/', views.get_favorite_stock_prediction_ajax, name='get_favorite_prediction_ajax'),
]
