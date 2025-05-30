from django.urls import path
from . import views

app_name = 'chatbot'

urlpatterns = [
    path('', views.chatbot_page, name='chatbot_page'),
    path('chat/', views.chat_response, name='chat_response'),
    path('filter_message/', views.filter_message, name='filter_message'),
]