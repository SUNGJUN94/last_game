from django.contrib import admin
from django.db.models import Q  # Q 임포트 추가
from .models import FreeBoard, FreeBoardComment, FreeBoardLike, Notification, AdminBoard

# AdminBoard Admin 커스터마이징
@admin.register(AdminBoard)
class AdminBoardAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'user', 'reg_dt', 'is_visible')
    list_filter = ('is_visible', 'reg_dt')
    search_fields = ('title', 'content')
    date_hierarchy = 'reg_dt'
    ordering = ('-reg_dt',)
    list_per_page = 25

    # 상세 페이지에서 편집 가능한 필드
    fields = ('user', 'title', 'content', 'image', 'is_visible')

    # 이미지 미리보기 추가
    def image_preview(self, obj):
        if obj.image:
            return f'<img src="{obj.image.url}" style="max-height: 100px;" />'
        return 'No Image'
    image_preview.allow_tags = True
    image_preview.short_description = 'Image Preview'
    list_display += ('image_preview',)

    # 권한 설정: 운영자(superuser)만 작성 가능
    def has_add_permission(self, request):
        return request.user.is_superuser

    def has_change_permission(self, request, obj=None):
        return request.user.is_superuser

    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser
    
# FreeBoard Admin 커스터마이징
@admin.register(FreeBoard)
class FreeBoardAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'user', 'category', 'reg_dt', 'view_count', 'likes_count', 'worried_count', 'is_deleted')
    list_filter = ('category', 'is_deleted', 'reg_dt')
    search_fields = ('title', 'content')  # user__username 제거
    date_hierarchy = 'reg_dt'
    ordering = ('-reg_dt',)
    list_per_page = 25
    fields = ('user', 'title', 'content', 'category', 'image', 'is_deleted')

# FreeBoardComment Admin 커스터마이징
@admin.register(FreeBoardComment)
class FreeBoardCommentAdmin(admin.ModelAdmin):
    list_display = ('id', 'free_board', 'user', 'content', 'reg_dt', 'is_deleted')
    list_filter = ('is_deleted', 'reg_dt')
    search_fields = ('content', 'user__username')
    date_hierarchy = 'reg_dt'
    ordering = ('-reg_dt',)

# FreeBoardLike Admin 커스터마이징
@admin.register(FreeBoardLike)
class FreeBoardLikeAdmin(admin.ModelAdmin):
    list_display = ('id', 'free_board', 'user', 'is_liked', 'is_worried', 'reg_dt')
    list_filter = ('is_liked', 'is_worried', 'reg_dt')
    search_fields = ('user__username',)
    ordering = ('-reg_dt',)

# Notification Admin 커스터마이징
@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ('id', 'recipient', 'sender', 'message', 'created_at', 'is_read')
    list_filter = ('is_read', 'created_at')
    search_fields = ('message', 'recipient__username', 'sender__username')
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)