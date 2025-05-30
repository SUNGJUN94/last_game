from django.urls import path
from . import views

app_name = 'account'
urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('find_id/', views.find_id_view, name='find_id'),
    path('find_pw/', views.find_pw_view, name='find_pw'),
]