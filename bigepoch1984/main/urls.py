from django.contrib import admin
from django.urls import path, include
from . import views

app_name = 'main'

urlpatterns = [
    path('', views.main, name='main'),
    path('predict/', include('predict_info.urls')),
    path('api/news/', views.get_naver_news, name='get_naver_news'),
]
