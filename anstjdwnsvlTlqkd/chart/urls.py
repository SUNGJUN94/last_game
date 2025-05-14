from django.urls import path
from . import views

app_name = 'chart'

urlpatterns = [
    path('', views.chart_view, name='chart'),
    path('api/realtime-price/', views.get_realtime_price, name='realtime_price'),
]
