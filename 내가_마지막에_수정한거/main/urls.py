from django.contrib import admin
from django.urls import path, include
from . import views

app_name = 'main'

urlpatterns = [
    path('', views.main, name='main'),
    path('predict/', include('predict_info.urls')),
    path('api/news/', views.get_naver_news, name='get_naver_news'),
    path('admin-board/<int:pk>/', views.admin_board_detail, name='admin_board_detail'),
    path('faq/', views.main_faq, name='main_faq'),
    path('coalition/', views.main_coalition, name='main_coalition'),
    path('advertise/', views.main_advertise, name='main_advertise'),
]
