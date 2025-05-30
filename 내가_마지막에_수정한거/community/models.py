from django.db import models
from django.conf import settings
from django.utils import timezone
from django.urls import reverse

#add1
class AdminBoard(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='admin_boards')
    title = models.CharField(max_length=200)
    content = models.TextField()
    image = models.ImageField(upload_to='admin_board_images/', blank=True, null=True)
    reg_dt = models.DateTimeField(auto_now_add=True)
    up_dt = models.DateTimeField(auto_now=True)
    is_visible = models.BooleanField(default=True)  # 보여질지 여부 (YN 처리)
    
    class Meta:
        ordering = ['-reg_dt']  # 최신순 정렬
        verbose_name = "관리자 게시글"
        verbose_name_plural = "관리자 게시글 목록"

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        # 추후 상세 페이지가 필요하면 URL 정의
        return reverse('main:admin_board_detail', args=[str(self.id)])
#add2

# 알림 모델
class Notification(models.Model):
    recipient = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='notifications')
    sender = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='sent_notifications')
    comment = models.ForeignKey('FreeBoardComment', on_delete=models.CASCADE, related_name='notifications')
    message = models.CharField(max_length=255)  # 예: "유저2님이 댓글을 작성했습니다."
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    class Meta:
        ordering = ['-created_at']  # 최신순 정렬
        verbose_name = "알림"
        verbose_name_plural = "알림 목록"

    def __str__(self):
        return self.message

    def get_absolute_url(self):
        # 댓글의 게시글 상세 페이지로 이동 (댓글 섹션으로 포커스)
        return f"{reverse('community:detail', args=[self.comment.free_board.id])}#comments"

class FreeBoard(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='free_boards')
    title = models.CharField(max_length=200)
    content = models.TextField()
    reg_dt = models.DateTimeField(auto_now_add=True)
    up_dt = models.DateTimeField(auto_now=True)
    likes_count = models.PositiveIntegerField(default=0)
    comments_count = models.PositiveIntegerField(default=0)
    view_count = models.PositiveIntegerField(default=0)
    worried_count = models.PositiveIntegerField(default=0)
    is_deleted = models.BooleanField(default=False)
    CATEGORY_CHOICES = [
        ('잡담', '잡담'),
        ('실시간뉴스', '실시간뉴스'),
        ('수동공시', '수동공시'),
        ('API공시', 'API공시'),
    ]
    category = models.CharField(
        max_length=20,
        choices=CATEGORY_CHOICES,
        default='잡담',
        db_index=True
    )
    dart_rcept_no = models.CharField(max_length=14, blank=True, null=True, unique=True, help_text="DART API 공시의 경우 원본 접수번호")
    image = models.ImageField(upload_to='community_images/', blank=True, null=True)

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('community:detail', args=[str(self.id)])

class FreeBoardComment(models.Model):
    free_board = models.ForeignKey(FreeBoard, on_delete=models.CASCADE, related_name='comments')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='free_board_comments')
    content = models.TextField()
    reg_dt = models.DateTimeField(auto_now_add=True)
    up_dt = models.DateTimeField(auto_now=True)
    is_deleted = models.BooleanField(default=False)

    def __str__(self):
        return f'Comment by {self.user} on {self.free_board}'

class FreeBoardLike(models.Model):
    free_board = models.ForeignKey(FreeBoard, on_delete=models.CASCADE, related_name='likes')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='free_board_likes')
    reg_dt = models.DateTimeField(auto_now_add=True)
    is_liked = models.BooleanField(default=False)  # 좋아요 상태
    is_worried = models.BooleanField(default=False)  # 걱정돼요 상태

    class Meta:
        unique_together = ('free_board', 'user')

    def __str__(self):
        return f'Like/Worry by {self.user} on {self.free_board}'

class DartDisclosure(models.Model):
    corp_code = models.CharField(max_length=8, help_text="공시대상회사의 고유번호(8자리)")
    corp_name = models.CharField(max_length=255, help_text="공시대상회사명")
    stock_code = models.CharField(max_length=6, blank=True, null=True, help_text="상장회사의 종목코드(6자리)")
    corp_cls = models.CharField(max_length=1, help_text="법인구분 : Y(유가), K(코스닥), N(코넥스), E(기타)")
    report_nm = models.CharField(max_length=255, help_text="보고서명")
    rcept_no = models.CharField(max_length=14, unique=True, primary_key=True, help_text="접수번호(14자리)")
    flr_nm = models.CharField(max_length=255, help_text="공시 제출인명")
    rcept_dt = models.DateField(help_text="접수일자(YYYYMMDD)")
    rm = models.CharField(max_length=10, blank=True, null=True, help_text="비고")
    doc_url = models.URLField(max_length=500, blank=True, null=True, help_text="공시뷰어 URL (doc.dart.fss.or.kr)")
    report_link = models.URLField(max_length=500, blank=True, null=True, help_text="원문 DART 링크 (dart.fss.or.kr)")
    document_content = models.TextField(blank=True, null=True, help_text="공시 상세 본문 내용")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-rcept_dt', '-rcept_no']
        verbose_name = "DART 공시 정보"
        verbose_name_plural = "DART 공시 정보 목록"

    def __str__(self):
        return f"[{self.rcept_dt}] {self.corp_name} - {self.report_nm}"

class NewsArticle(models.Model):
    title = models.CharField(max_length=255, help_text="뉴스 기사 제목")
    original_link = models.URLField(max_length=500, unique=True, help_text="뉴스 원문 링크 (중복 방지용)")
    naver_link = models.URLField(max_length=500, blank=True, null=True, help_text="네이버 뉴스 링크 (API 제공 링크)")
    description = models.TextField(help_text="뉴스 요약 내용")
    pub_date = models.DateTimeField(help_text="뉴스 발행 일시")
    created_at = models.DateTimeField(auto_now_add=True, help_text="DB 저장 시각")

    class Meta:
        ordering = ['-pub_date']
        verbose_name = "네이버 뉴스 기사"
        verbose_name_plural = "네이버 뉴스 기사 목록"

    def __str__(self):
        return self.title