from django.urls import path
from . import views

app_name = 'community'

urlpatterns = [
    path('', views.community_view, name='community'),
    path('write/', views.write_view, name='write'),
    path('<int:post_id>/', views.community_detail_view, name='detail'),
    path('<int:post_id>/comment/', views.comment_create, name='comment_create'),
    path('<int:post_id>/like/', views.like_post, name='like_post'),
    path('<int:post_id>/edit/', views.edit_view, name='edit'),
    path('<int:post_id>/delete/', views.delete_view, name='delete'),
    # add1
    path('comment/edit/<int:pk>/', views.comment_edit, name='comment_edit'),
    path('comment/delete/<int:pk>/', views.comment_delete, name='comment_delete'),
    # add2
    path('notifications/', views.notifications_view, name='notifications'),  # 알림 뷰 추가
    path('block/<int:post_id>/', views.block_user, name='block_user'),
    path('report/<int:post_id>/', views.report_user, name='report_user'),
    path('unblock_user/', views.unblock_user, name='unblock_user'),
    path('block_user_by_comment/<int:comment_id>/', views.block_user_by_comment, name='block_user_by_comment'),
    path('check_curse/', views.check_curse_view, name='check_curse'),
]