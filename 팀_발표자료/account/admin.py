from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, ReportedUser, BlockedUser

@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ('login_id', 'nickname', 'email', 'auth_id', 'is_active', 'is_staff', 'created_at')
    list_filter = ('auth_id', 'is_active', 'is_staff')
    search_fields = ('login_id', 'nickname', 'email')
    ordering = ('-created_at',)
    fieldsets = (
        (None, {'fields': ('login_id', 'password')}),
        ('Personal Info', {'fields': ('nickname', 'email', 'auth_id', 'profile_image', 'greeting_message')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('created_at',)}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('login_id', 'nickname', 'email', 'password1', 'password2', 'auth_id', 'is_active', 'is_staff'),
        }),
    )
    readonly_fields = ('created_at',)
    
    actions = ['make_admin', 'make_user']

    def make_admin(self, request, queryset):
        updated = queryset.update(auth_id='admin', is_staff=True)
        self.message_user(request, f"{updated}명의 사용자를 운영자로 변경했습니다.")
    make_admin.short_description = "선택된 사용자를 운영자로 설정"

    def make_user(self, request, queryset):
        updated = queryset.update(auth_id='user', is_staff=False)
        self.message_user(request, f"{updated}명의 사용자를 일반회원으로 변경했습니다.")
    make_user.short_description = "선택된 사용자를 일반회원으로 설정"

@admin.register(ReportedUser)
class ReportedUserAdmin(admin.ModelAdmin):
    list_display = ('reporter', 'reported', 'created_at')
    search_fields = ('reporter__nickname', 'reported__nickname')
    list_filter = ('created_at',)
    ordering = ('-created_at',)

@admin.register(BlockedUser)
class BlockedUserAdmin(admin.ModelAdmin):
    list_display = ('blocker', 'blocked', 'created_at')
    search_fields = ('blocker__nickname', 'blocked__nickname')
    list_filter = ('created_at',)
    ordering = ('-created_at',)