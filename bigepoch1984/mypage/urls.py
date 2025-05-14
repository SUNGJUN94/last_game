from django.urls import path
from . import views

app_name = 'mypage'

urlpatterns = [
    path('', views.mypage_view, name='mypage'),
    path('edit/', views.edit_profile_view, name='edit_profile'),
    path('update-greeting/', views.update_greeting_message, name='update_greeting_message'),  # 추가
    path('unblock/<int:blocked_id>/', views.unblock_user, name='unblock_user'),
]