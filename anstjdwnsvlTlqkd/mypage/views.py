from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect
from account.models import User
from community.models import FreeBoard
from django.contrib import messages
import logging
from django.core.files.storage import default_storage
from django.contrib.auth import login
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import JsonResponse
from account.models import ReportedUser
from account.models import BlockedUser
from django.views.decorators.http import require_http_methods

@login_required
def unblock_user(request, blocked_id):
    """
    현재 로그인된 사용자가 특정 사용자를 차단 해제하는 기능
    """
    if request.method == 'POST':
        logger.debug(f"Unblock request received for blocked_id: {blocked_id}")
        blocker = request.user
        blocked_user = get_object_or_404(User, user_id=blocked_id)
        block_relation = BlockedUser.objects.filter(blocker=blocker, blocked=blocked_user).first()
        if not block_relation:
            logger.warning(f"No block relation found for blocker: {blocker.user_id}, blocked: {blocked_id}")
            return redirect('mypage:mypage')
        block_relation.delete()
        logger.info(f"Blocked user {blocked_user.nickname} (user_id: {blocked_id}) unblocked by {blocker.nickname}")
        return redirect('mypage:mypage')
    logger.debug(f"Non-POST request to unblock_user: {request.method}")
    return redirect('mypage:mypage')

@login_required
def mypage(request):
    return render(request, 'mypage/mypage.html')

# 로거 설정
logger = logging.getLogger(__name__)

# views.py
@login_required
def mypage_view(request):
    if not request.user.is_authenticated:
        logger.warning("Unauthorized access to mypage_view - redirecting to login")
        return HttpResponseRedirect('/account/login/?next=/mypage/')
    user = request.user
    # FreeBoard에서 최대 5개 게시글 가져오기
    user_posts_qs = FreeBoard.objects.filter(user=user, is_deleted=False).select_related('user').order_by('-reg_dt')[:5]

    # 시간 경과 계산
    user_posts = []
    for post in user_posts_qs:
        time_diff = timezone.now() - post.reg_dt
        if time_diff.days > 0:
            time_ago = f"{time_diff.days}일 전"
        elif time_diff.seconds // 3600 > 0:
            time_ago = f"{time_diff.seconds // 3600}시간 전"
        elif time_diff.seconds // 60 > 0:
            time_ago = f"{time_diff.seconds // 60}분 전"
        else:
            time_ago = "방금 전"
        user_posts.append({
            'title': post.title,
            'time_ago': time_ago,
            'url': post.get_absolute_url(),
        })

    # 신고/차단 목록 데이터 (최대 5개)
    reported_users = ReportedUser.objects.filter(reporter=user).select_related('reported')[:5]
    # 차단 목록에서 유효한 유저만 가져오기
    blocked_users = BlockedUser.objects.filter(blocker=user).select_related('blocked').filter(blocked__isnull=False)[:5]
    
    # 디버깅: blocked_users의 user_id와 nickname 로깅
    for block in blocked_users:
        logger.debug(f"Blocked user - User ID: {block.blocked.user_id}, Nickname: {block.blocked.nickname}")

    context = {
        'user': user,
        'prediction_items': [
            {'name': '삼성전자', 'price': '82,000원', 'change': '+1.20%'},
            {'name': '비트코인', 'price': '125,000,000원', 'change': '+0.80%'},
        ],
        'watchlist': [
            {'name': '이더리움'},
            {'name': '애플'},
        ],
        'user_posts': user_posts,
        'reported_users': reported_users,
        'blocked_users': blocked_users,
        'now': timezone.now(),
    }
    return render(request, 'mypage.html', context)

@login_required
def update_greeting_message(request):
    """
    사용자의 인사 메시지를 저장하거나 업데이트하는 뷰입니다.
    """
    if request.method == 'POST':
        greeting_message = request.POST.get('greeting_message', '').strip()
        logger.debug(f"Received greeting message: {greeting_message}")

        # 유효성 검사
        if len(greeting_message) > 100:
            logger.warning(f"Greeting message too long: {len(greeting_message)} characters")
            return JsonResponse({'success': False, 'message': '인사 메시지는 최대 100자까지 가능합니다.'}, status=400)

        try:
            user = request.user
            user.greeting_message = greeting_message
            user.save(update_fields=['greeting_message'])
            logger.info(f"Greeting message updated for user {user.login_id}: {greeting_message}")
            return JsonResponse({'success': True, 'message': '인사 메시지가 성공적으로 저장되었습니다.'})
        except Exception as e:
            logger.error(f"Failed to update greeting message for user {request.user.login_id}: {str(e)}")
            return JsonResponse({'success': False, 'message': '저장 중 오류가 발생했습니다.'}, status=500)

    return JsonResponse({'success': False, 'message': '잘못된 요청입니다.'}, status=400)

def edit_profile_view(request):
    if not request.user.is_authenticated:
        logger.warning("Unauthorized access to edit_profile_view - redirecting to login")
        return HttpResponseRedirect('/account/login/?next=/mypage/edit/')

    user = request.user

    if request.method == 'POST':
        logger.debug(f"Received POST data: {request.POST}")
        logger.debug(f"Received FILES data: {request.FILES}")

        nickname = request.POST.get('nickname')
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')
        profile_image = request.FILES.get('profile_image')

        # 닉네임 유효성 검사
        if not nickname or len(nickname) < 2:
            logger.warning(f"Invalid nickname: {nickname}")
            messages.error(request, '닉네임은 최소 2자 이상이어야 합니다.')
            return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})

        if nickname != user.nickname and User.objects.filter(nickname=nickname).exists():
            logger.warning(f"Nickname already exists: {nickname}")
            messages.error(request, '이미 사용 중인 닉네임입니다.')
            return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})

        # 비밀번호 유효성 검사
        if password:
            if len(password) < 8:
                logger.warning(f"Password too short: {len(password)} characters")
                messages.error(request, '비밀번호는 최소 8자 이상이어야 합니다.')
                return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})
            if password != password_confirm:
                logger.warning("Password confirmation does not match")
                messages.error(request, '비밀번호가 일치하지 않습니다.')
                return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})

        # 사용자 정보 업데이트
        user.nickname = nickname
        password_changed = False
        if password:
            user.set_password(password)
            password_changed = True
            logger.info(f"Password updated for user {user.login_id}")

        # 프로필 이미지 처리
        if profile_image:
            logger.debug("Profile image received, processing upload")
            # 기존 이미지 삭제 (기본 이미지가 아닌 경우)
            if user.profile_image and user.profile_image.name != 'profile_images/default.jpg':
                if default_storage.exists(user.profile_image.name):
                    try:
                        default_storage.delete(user.profile_image.name)
                        logger.info(f"Deleted old profile image: {user.profile_image.name}")
                    except Exception as e:
                        logger.error(f"Failed to delete old profile image: {str(e)}")
            user.profile_image = profile_image
            logger.info(f"Assigned new profile image: {user.profile_image.name}")
        else:
            logger.debug("No new profile image uploaded, keeping existing image")

        try:
            user.save()
            logger.info(f"User {user.login_id} updated profile: nickname={user.nickname}, profile_image={user.profile_image.url if user.profile_image else 'default'}")
        except Exception as e:
            logger.error(f"Failed to save user profile: {str(e)}")
            messages.error(request, '프로필 저장 중 오류가 발생했습니다. 다시 시도해주세요.')
            return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})

        if password_changed:
            login(request, user)  # 비밀번호 변경 시 세션 갱신
        messages.success(request, '프로필이 성공적으로 수정되었습니다.')
        return redirect('mypage:mypage')

    return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})