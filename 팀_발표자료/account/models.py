from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager
from django.conf import settings

#add1_250514_13_14
class ReportedUser(models.Model):
    reporter = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='reported_users', on_delete=models.CASCADE)
    reported = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='reported_by', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('reporter', 'reported')  # 중복 신고 방지
        verbose_name = '신고한 유저'
        verbose_name_plural = '신고한 유저 목록'

    def __str__(self):
        return f"{self.reporter.nickname} -> {self.reported.nickname}"

class BlockedUser(models.Model):
    blocker = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='blocked_users', on_delete=models.CASCADE)
    blocked = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='blocked_by', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('blocker', 'blocked')  # 중복 차단 방지
        verbose_name = '차단한 유저'
        verbose_name_plural = '차단한 유저 목록'

    def __str__(self):
        return f"{self.blocker.nickname} -> {self.blocked.nickname}"
#add2

class UserManager(BaseUserManager):
    def create_user(self, login_id, email, nickname, password=None, **extra_fields):
        if not login_id:
            raise ValueError('The Login ID must be set')
        if not email:
            raise ValueError('The Email must be set')
        if not nickname:
            raise ValueError('The Nickname must be set')
        
        email = self.normalize_email(email)
        user = self.model(
            login_id=login_id,
            email=email,
            nickname=nickname,
            **extra_fields
        )
        user.set_password(password)  # 비밀번호 해싱
        user.save(using=self._db)
        return user

    def create_superuser(self, login_id, email, nickname, password=None, **extra_fields):
        extra_fields.setdefault('auth_id', 'admin')
        user = self.create_user(
            login_id=login_id,
            email=email,
            nickname=nickname,
            password=password,
            **extra_fields
        )
        user.is_staff = True
        user.is_superuser = True
        user.save(using=self._db)
        return user

class User(AbstractBaseUser, PermissionsMixin):
    AUTH_CHOICES = [
        ('admin', '운영자'),
        ('user', '일반회원'),
    ]
    user_id = models.AutoField(primary_key=True)
    login_id = models.CharField(max_length=50, unique=True)
    auth_id = models.CharField(max_length=10, choices=AUTH_CHOICES, default='user')
    nickname = models.CharField(max_length=50, unique=True)
    email = models.EmailField(unique=True)
    profile_image = models.ImageField(upload_to='profile_images/', null=True, blank=True, default='profile_images/default.jpg')
    greeting_message = models.CharField(max_length=100, blank=True, default="")  # 인사 메시지 필드 추가
    created_at = models.DateTimeField(auto_now_add=True)  # 생성 시간
    is_active = models.BooleanField(default=True)  # 계정 활성화 여부
    is_staff = models.BooleanField(default=False)  # 관리자 페이지 접근 권한
    name = models.CharField(max_length=30, blank=True, default="")  # 이름 필드 추가
    # UserManager 설정
    objects = UserManager()

    # 필수 필드 설정
    USERNAME_FIELD = 'login_id'  # 로그인 시 사용할 필드
    REQUIRED_FIELDS = ['email', 'nickname']  # 필수 입력 필드

    def __str__(self):
        return self.login_id