"""
URL configuration for mysite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include('main.urls', namespace='main')),
    path("account/", include('account.urls', namespace='account')),
    path("chart/", include('chart.urls', namespace='chart')),
    path("predict_info/", include('predict_info.urls', namespace='predict_info')),
    path("community/", include('community.urls', namespace='community')),
    path("mypage/", include('mypage.urls', namespace='mypage')),
    path("chatbot/", include('chatbot.urls', namespace='chatbot')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
