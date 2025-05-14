from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse
from django.urls import reverse
from .models import FreeBoard, FreeBoardComment, FreeBoardLike, DartDisclosure, NewsArticle, Notification
from django.utils import timezone as django_timezone
from django.contrib import messages
from django.db import transaction
import logging
from datetime import datetime, timedelta
from django.db.models import Q
import re

from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger, Page
from account.models import BlockedUser  # BlockedUser 모델 임포트 추가

logger = logging.getLogger(__name__)

@login_required
def report_user(request, post_id):
    return render(request, 'community.html')

@login_required
def block_user(request, post_id):
    """
    게시글 작성자를 차단하는 AJAX �뷰.
    """
    if not request.user.is_authenticated:
        return JsonResponse({'error': '로그인 필요'}, status=401)

    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False)
    target_user = post.user
    if request.user == target_user:
        return JsonResponse({'error': '본인 게시글은 차단할 수 없습니다.'}, status=400)

    target_username = request.POST.get('target_user')
    if target_username and target_username != target_user.nickname:
        return JsonResponse({'error': '유효하지 않은 사용자입니다.'}, status=400)

    BlockedUser.objects.get_or_create(blocker=request.user, blocked=target_user)
    return JsonResponse({'status': 'success', 'message': f'{target_user.nickname}님을 차단했습니다.'})

def notifications_view(request):
    """
    로그인된 사용자의 알림 데이터를 JSON 형식으로 반환합니다.
    최신 3개만 반환하며, 읽음 여부도 포함.
    """
    if not request.user.is_authenticated:
        return JsonResponse({'error': '로그인 필요'}, status=401)

    notifications = Notification.objects.filter(
        recipient=request.user
    ).select_related('sender', 'comment__free_board')[:3]  # 최신 3개만

    notification_data = []
    for notif in notifications:
        time_diff = django_timezone.now() - notif.created_at
        if time_diff.days > 0:
            time_ago = f"{time_diff.days}일 전"
        elif time_diff.seconds // 3600 > 0:
            time_ago = f"{time_diff.seconds // 3600}시간 전"
        elif time_diff.seconds // 60 > 0:
            time_ago = f"{time_diff.seconds // 60}분 전"
        else:
            time_ago = "방금 전"

        sender_name = notif.sender.nickname if hasattr(notif.sender, 'nickname') and notif.sender.nickname else notif.sender.username
        notification_data.append({
            'user': sender_name,
            'preview': notif.message,
            'time': time_ago,
            'link': notif.get_absolute_url(),
            'is_read': notif.is_read
        })

    return JsonResponse(notification_data, safe=False)

def extract_api_disclosure_info(content_str):
    """
    API 공시 내용에서 회사명과 원문 링크를 추출합니다.
    """
    company_name = "정보 없음"
    report_link = "#"
    company_match = re.search(r"회사명: (.*?)\n", content_str)
    if company_match:
        company_name = company_match.group(1).strip()
    link_match = re.search(r"공시 원문 보기: (https?://[^\s]+)", content_str)
    if link_match:
        report_link = link_match.group(1).strip()
    return company_name, report_link

def community_view(request):
    # 기존 코드 유지
    tab = request.GET.get('tab', 'community')
    subtab = request.GET.get('subtab', '')

    # 차단된 사용자 목록 가져오기
    blocked_user_ids = []
    if request.user.is_authenticated:
        blocked_user_ids = BlockedUser.objects.filter(blocker=request.user).values_list('blocked_id', flat=True)

    # 글 목록에서 차단된 사용자의 글 제외
    disclosure_posts_for_carousel_qs = FreeBoard.objects.filter(
        Q(category='API공시') | Q(category='수동공시'),
        is_deleted=False
    ).exclude(user_id__in=blocked_user_ids).select_related('user').order_by('-reg_dt')[:5]

    processed_carousel_disclosure_list = []
    for post in disclosure_posts_for_carousel_qs:
        item_data = {
            'obj': post, 'display_title': post.title,
            'display_company_name': post.user.nickname if hasattr(post.user, 'nickname') and post.user.nickname else post.user.get_username(),
            'display_date': post.reg_dt, 'display_category': post.get_category_display(),
            'link': post.get_absolute_url(), 'is_api': False
        }
        if post.category == 'API공시':
            company_name, report_link = extract_api_disclosure_info(post.content)
            item_data['display_company_name'] = company_name
            item_data['link'] = report_link
            item_data['is_api'] = True
        processed_carousel_disclosure_list.append(item_data)

    paginator_for_carousel = Paginator(processed_carousel_disclosure_list, 5)
    try:
        page_for_community_carousel = paginator_for_carousel.page(1)
    except EmptyPage:
        page_for_community_carousel = Page([], 1, paginator_for_carousel)

    if tab == 'news':
        # 기존 뉴스 탭 코드 유지
        pass

    period = request.GET.get('period', '한달')
    sort = request.GET.get('sort', '최신순')

    now = django_timezone.now()
    # 차단된 사용자의 글 제외
    all_posts_queryset = FreeBoard.objects.filter(
        is_deleted=False,
        category='잡담',
        reg_dt__gte=now - timedelta(days={'하루': 1, '일주일': 7, '한달': 30, '반년': 180}.get(period, 30))
    ).exclude(user_id__in=blocked_user_ids).select_related('user')

    processed_post_list = []
    if request.user.is_authenticated:
        liked_post_ids = FreeBoardLike.objects.filter(
            user=request.user,
            free_board__in=all_posts_queryset,
            is_liked=True
        ).values_list('free_board_id', flat=True)
        liked_post_ids = set(liked_post_ids)
    else:
        liked_post_ids = set()

    for post_obj in all_posts_queryset:
        time_diff = now - post_obj.reg_dt
        days_ago = time_diff.days
        time_ago = "방금 전" if days_ago == 0 and time_diff.seconds < 60 else \
                   f"{time_diff.seconds // 3600}시간 전" if time_diff.seconds // 3600 > 0 else \
                   f"{time_diff.seconds // 60}분 전" if time_diff.seconds // 60 > 0 else \
                   f"{days_ago}일 전"

        processed_post_list.append({
            'id': post_obj.id,
            'user': post_obj.user,
            'username': post_obj.user.nickname if hasattr(post_obj.user, 'nickname') and post_obj.user.nickname else post_obj.user.get_username(),
            'auth_id': post_obj.user.auth_id if hasattr(post_obj.user, 'auth_id') else '',
            'category': getattr(post_obj, 'category', '잡담'),
            'time_ago': time_ago,
            'days_ago': days_ago,
            'title': post_obj.title,
            'content': post_obj.content,
            'likes_count': post_obj.likes_count,
            'comments_count': post_obj.comments_count,
            'view_count': post_obj.view_count,
            'worried_count': post_obj.worried_count,
            'reg_dt': post_obj.reg_dt,
            'important': getattr(post_obj, 'important', 5),
            'is_liked': post_obj.id in liked_post_ids,
            'get_absolute_url': post_obj.get_absolute_url(),
            'tags': post_obj.tags.all() if hasattr(post_obj, 'tags') else [],
        })

    # 기존 정렬 및 페이지네이션 코드 유지
    if sort == '최신순':
        processed_post_list.sort(key=lambda x: x['reg_dt'], reverse=True)
    elif sort == '조회수순':
        processed_post_list.sort(key=lambda x: (-x['view_count'], -x['reg_dt'].timestamp()))
    elif sort == '중요순':
        processed_post_list.sort(key=lambda x: (-x['likes_count'], -x['reg_dt'].timestamp()))
    elif sort == '걱정순':
        processed_post_list.sort(key=lambda x: (-x['worried_count'], -x['reg_dt'].timestamp()))
    else:
        processed_post_list.sort(key=lambda x: x['reg_dt'], reverse=True)

    paginator_community = Paginator(processed_post_list, 10)
    page_number_community = request.GET.get('page', 1)
    try:
        community_page_obj = paginator_community.page(page_number_community)
    except (EmptyPage, PageNotAnInteger):
        community_page_obj = paginator_community.page(1)

    context = {
        'disclosures_for_carousel': page_for_community_carousel,
        'posts': community_page_obj,
        'page_obj': community_page_obj,
        'period': period,
        'sort': sort,
        'community_menus': [{'name': '커뮤니티'}, {'name': '뉴스'}, {'name': '종목'}, {'name': '예측'}, {'name': '공지'}],
        'active_tab': tab,
        'active_subtab': subtab if subtab else ('realtime' if tab == 'news' else ''),
        'ticker_message': '예측 정보 티커 영역 예시: 비트코인 1억 돌파 예측 중!',
    }
    return render(request, 'community.html', context)

def write_view(request):
    """
    새 게시글 작성 뷰입니다.
    로그인된 사용자만 접근 가능하며, 캡차 검증을 포함합니다.
    """
    if not request.user.is_authenticated:
        next_url = request.path
        if request.GET:
            next_url += '?' + request.GET.urlencode()
        return HttpResponseRedirect(f"{reverse('account:login')}?next={next_url}")

    board_type = request.GET.get('board_type', 'freeboard')

    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()
        captcha_value = request.POST.get('captcha_value', '').strip()
        captcha_answer = request.POST.get('captcha_answer', '').strip()

        logger.info(f"write_view POST: title='{title}', content_len={len(content)}, captcha_value='{captcha_value}', captcha_answer='{captcha_answer}', board_type='{board_type}'")

        if not title or not content:
            messages.error(request, '제목과 내용을 모두 입력해주세요.')
            return render(request, 'community_write.html', {
                'error_message': '제목과 내용을 모두 입력해주세요.',
                'title': title,
                'content': content,
                'board_type': board_type,
            })

        if not captcha_value.isdigit() or len(captcha_value) != 4:
            logger.warning(f"Invalid captcha_value provided: {captcha_value}")
            messages.error(request, '잘못 입력했습니다.')
            return render(request, 'community_write.html', {
                'error_message': '잘못 입력했습니다.',
                'title': title,
                'content': content,
                'board_type': board_type,
            })

        if captcha_answer != captcha_value:
            logger.warning(f"Captcha mismatch: answer='{captcha_answer}', expected_value='{captcha_value}'")
            messages.error(request, '잘못 입력했습니다.')
            return render(request, 'community_write.html', {
                'error_message': '잘못 입력했습니다.',
                'title': title,
                'content': content,
                'board_type': board_type,
            })

        post_category = '잡담'
        if board_type == 'realtime_news':
            messages.info(request, "실시간 뉴스는 자동 수집됩니다. 직접 작성 기능은 현재 비활성화되어 있습니다.")
            return redirect(f"{reverse('community:community')}?tab=news&subtab=realtime")
        elif board_type == 'disclosure_manual':
            post_category = '수동공시'

        if board_type != 'realtime_news':
            new_post = FreeBoard.objects.create(
                user=request.user,
                title=title,
                content=content,
                category=post_category
            )
            messages.success(request, '게시물이 성공적으로 등록되었습니다.')
            logger.info(f"Post created successfully: id={new_post.id}, title='{new_post.title}', category='{post_category}'")

        if board_type == 'realtime_news':
            return redirect(f"{reverse('community:community')}?tab=news&subtab=realtime")
        elif board_type == 'disclosure_manual':
            return redirect(f"{reverse('community:community')}?tab=news&subtab=disclosure")
        else:
            return redirect('community:community')

    return render(request, 'community_write.html', {
        'board_type': board_type
    })

def community_detail_view(request, post_id):
    # 차단된 사용자 목록 가져오기
    blocked_user_ids = []
    if request.user.is_authenticated:
        blocked_user_ids = BlockedUser.objects.filter(blocker=request.user).values_list('blocked_id', flat=True)

    # 게시글 조회 (차단된 사용자의 글 접근 시 404)
    post = get_object_or_404(FreeBoard.objects.select_related('user').exclude(user_id__in=blocked_user_ids), id=post_id, is_deleted=False)

    # 댓글 조회 (차단된 사용자의 댓글 제외)
    comments = FreeBoardComment.objects.filter(free_board=post, is_deleted=False).exclude(user_id__in=blocked_user_ids).select_related('user').order_by('reg_dt')

    # 기존 코드 유지
    if request.user.is_authenticated:
        viewed_posts_key = f'viewed_post_{post_id}'
        if not request.session.get(viewed_posts_key, False):
            with transaction.atomic():
                post.view_count += 1
                post.save(update_fields=['view_count'])
                request.session[viewed_posts_key] = True
                request.session.set_expiry(86400)
    else:
        ip_address = request.META.get('REMOTE_ADDR')
        viewed_ips_key = f'viewed_ip_{post_id}_{ip_address}'
        if not request.session.get(viewed_ips_key, False):
            with transaction.atomic():
                post.view_count += 1
                post.save(update_fields=['view_count'])
                request.session[viewed_ips_key] = True
                request.session.set_expiry(86400)

    time_diff = django_timezone.now() - post.reg_dt
    time_ago = "방금 전" if time_diff.days == 0 and time_diff.seconds < 60 else \
               f"{time_diff.seconds // 3600}시간 전" if time_diff.seconds // 3600 > 0 else \
               f"{time_diff.seconds // 60}분 전" if time_diff.seconds // 60 > 0 else \
               f"{time_diff.days}일 전"

    is_liked = False
    is_worried = False
    if request.user.is_authenticated:
        like_obj = FreeBoardLike.objects.filter(free_board=post, user=request.user).first()
        if like_obj:
            is_liked = like_obj.is_liked
            is_worried = like_obj.is_worried

    dart_link_for_detail = None
    company_name_for_detail = None
    if post.category == 'API공시':
        company_name_for_detail, dart_link_for_detail = extract_api_disclosure_info(post.content)

    post_data = {
        'id': post.id,
        'user': post.user,
        'username': post.user.nickname if hasattr(post.user, 'nickname') and post.user.nickname else post.user.get_username(),
        'auth_id': post.user.auth_id if hasattr(post.user, 'auth_id') else '',
        'title': post.title,
        'content': post.content,
        'time_ago': time_ago,
        'reg_dt': post.reg_dt,
        'likes_count': post.likes_count,
        'worried_count': post.worried_count,
        'comments_count': post.comments_count,
        'view_count': post.view_count,
        'is_liked': is_liked,
        'is_worried': is_worried,
        'is_author': request.user == post.user,
        'category': getattr(post, 'category', '잡담'),
        'dart_link': dart_link_for_detail,
        'company_name_for_api_disclosure': company_name_for_detail,
    }
    return render(request, 'community_detail.html', {'post': post_data, 'comments': comments})

@login_required
def like_post(request, post_id):
    """
    게시글 좋아요/걱정돼요 처리 AJAX 뷰.
    좋아요와 걱정돼요는 토글 방식으로 동작: 하나를 누르면 다른 하나는 취소됨.
    """
    if not request.user.is_authenticated:
        return JsonResponse({'error': '로그인 필요'}, status=401)

    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False)
    if post.user == request.user:
        return JsonResponse({'error': '본인 게시글에는 반응할 수 없습니다.'}, status=400)

    action = request.POST.get('action')
    logger.debug(f"Received action: {action}, POST data: {request.POST}")

    if action not in ['like', 'worry']:
        logger.error(f"Invalid or missing action: {action}")
        return JsonResponse({'error': '잘못된 요청: action 값이 필요합니다.'}, status=400)

    response_data = {'status': 'success'}

    with transaction.atomic():
        like_obj, created = FreeBoardLike.objects.get_or_create(
            free_board=post,
            user=request.user,
            defaults={'is_liked': False, 'is_worried': False}
        )

        if action == 'like':
            if like_obj.is_liked:
                # 이미 좋아요 상태 -> 좋아요 취소
                like_obj.is_liked = False
                post.likes_count = max(0, post.likes_count - 1)
            else:
                # 좋아요 추가, 걱정돼요 취소
                like_obj.is_liked = True
                post.likes_count += 1
                if like_obj.is_worried:
                    like_obj.is_worried = False
                    post.worried_count = max(0, post.worried_count - 1)
            like_obj.save()
            post.save(update_fields=['likes_count', 'worried_count'])
            logger.debug(f"Like action processed: is_liked={like_obj.is_liked}, is_worried={like_obj.is_worried}, likes_count={post.likes_count}, worried_count={post.worried_count}")

        elif action == 'worry':
            if like_obj.is_worried:
                # 이미 걱정돼요 상태 -> 걱정돼요 취소
                like_obj.is_worried = False
                post.worried_count = max(0, post.worried_count - 1)
            else:
                # 걱정돼요 추가, 좋아요 취소
                like_obj.is_worried = True
                post.worried_count += 1
                if like_obj.is_liked:
                    like_obj.is_liked = False
                    post.likes_count = max(0, post.likes_count - 1)
            like_obj.save()
            post.save(update_fields=['likes_count', 'worried_count'])
            logger.debug(f"Worry action processed: is_liked={like_obj.is_liked}, is_worried={like_obj.is_worried}, likes_count={post.likes_count}, worried_count={post.worried_count}")

        # 객체가 더 이상 필요 없으면 삭제
        if not like_obj.is_liked and not like_obj.is_worried:
            like_obj.delete()

        response_data.update({
            'likes_count': post.likes_count,
            'worried_count': post.worried_count,
            'is_liked': like_obj.is_liked if like_obj else False,
            'is_worried': like_obj.is_worried if like_obj else False
        })

    return JsonResponse(response_data)

def comment_create(request, post_id):
    """
    댓글 작성 AJAX 뷰.
    """
    if not request.user.is_authenticated:
        return JsonResponse({'error': '로그인 필요'}, status=401)

    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False)
    if request.method == 'POST':
        content = request.POST.get('content', '').strip()
        if not content:
            return JsonResponse({'error': '댓글 내용을 입력해주세요.'}, status=400)

        with transaction.atomic():
            comment = FreeBoardComment.objects.create(free_board=post, user=request.user, content=content)
            post.comments_count += 1
            post.save(update_fields=['comments_count'])
            if post.user != request.user:
                Notification.objects.create(
                    recipient=post.user,
                    sender=request.user,
                    comment=comment,
                    message=content[:30] + '...' if len(content) > 30 else content
                )

        comment_data = {
            'id': comment.id,
            'user': {'nickname': request.user.nickname or request.user.get_username(), 'auth_id': getattr(request.user, 'auth_id', '')},
            'content': comment.content,
            'reg_dt': comment.reg_dt.strftime('%Y-%m-%d %H:%M'),
            'is_author': True,
        }
        return JsonResponse({'status': 'success', 'comment': comment_data, 'comments_count': post.comments_count})

    return JsonResponse({'error': '잘못된 요청'}, status=400)

def edit_view(request, post_id):
    """
    게시글 수정 뷰입니다.
    게시글 작성자만 수정 가능하며, 캡차 검증을 포함합니다.
    """
    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False)
    if request.user != post.user:
        messages.error(request, '수정 권한이 없습니다.')
        return redirect('community:detail', post_id=post_id)

    board_type = getattr(post, 'category', 'freeboard')

    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()
        captcha_value = request.POST.get('captcha_value', '').strip()
        captcha_answer = request.POST.get('captcha_answer', '').strip()

        logger.info(f"edit_view POST: post_id={post_id}, title='{title}', content_len={len(content)}, captcha_value='{captcha_value}', captcha_answer='{captcha_answer}'")

        if not title or not content:
            messages.error(request, '제목과 내용을 모두 입력해주세요.')
            return render(request, 'community_write.html', {
                'error_message': '제목과 내용을 모두 입력해주세요.',
                'title': title,
                'content': content,
                'post_id': post_id,
                'is_edit': True,
                'board_type': board_type,
            })

        if not captcha_value.isdigit() or len(captcha_value) != 4:
            logger.warning(f"Invalid captcha_value during edit: {captcha_value}")
            messages.error(request, '잘못 입력했습니다.')
            return render(request, 'community_write.html', {
                'error_message': '잘못 입력했습니다.',
                'title': title,
                'content': content,
                'post_id': post_id,
                'is_edit': True,
                'board_type': board_type,
            })

        if captcha_answer != captcha_value:
            logger.warning(f"Captcha mismatch during edit: answer='{captcha_answer}', value='{captcha_value}'")
            messages.error(request, '잘못 입력했습니다.')
            return render(request, 'community_write.html', {
                'error_message': '잘못 입력했습니다.',
                'title': title,
                'content': content,
                'post_id': post_id,
                'is_edit': True,
                'board_type': board_type,
            })

        post.title = title
        post.content = content
        post.save()
        messages.success(request, '게시물이 성공적으로 수정되었습니다.')
        logger.info(f"Post edited successfully: id={post.id}, title='{post.title}'")
        return redirect('community:detail', post_id=post_id)

    return render(request, 'community_write.html', {
        'title': post.title,
        'content': post.content,
        'post_id': post_id,
        'is_edit': True,
        'board_type': board_type,
    })

def delete_view(request, post_id):
    """
    게시글 삭제 뷰입니다. (논리적 삭제: is_deleted=True)
    게시글 작성자만 삭제 가능합니다.
    """
    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False)
    if request.user != post.user:
        messages.error(request, '삭제 권한이 없습니다.')
        return redirect('community:detail', post_id=post.id)

    if request.method == 'POST':
        post.is_deleted = True
        post.save(update_fields=['is_deleted'])
        messages.success(request, '게시물이 성공적으로 삭제되었습니다.')

        original_category = getattr(post, 'category', 'freeboard')
        if original_category == '실시간뉴스':
            return redirect(f"{reverse('community:community')}?tab=news&subtab=realtime")
        elif original_category in ['수동공시', 'API공시']:
            return redirect(f"{reverse('community:community')}?tab=news&subtab=disclosure")
        return redirect('community:community')

    return render(request, 'community_delete.html', {'post': post})

@login_required
def comment_edit(request, pk):
    """
    댓글 수정 뷰입니다.
    댓글 작성자만 수정 가능합니다.
    """
    comment = get_object_or_404(FreeBoardComment, pk=pk, is_deleted=False)
    if request.user != comment.user:
        messages.error(request, "본인이 작성한 댓글만 수정할 수 있습니다.")
        return redirect('community:detail', post_id=comment.free_board.id)

    if request.method == "POST":
        content = request.POST.get('content', '').strip()
        if content:
            comment.content = content
            comment.save(update_fields=['content'])
            messages.success(request, "댓글이 수정되었습니다.")
        else:
            messages.error(request, "댓글 내용을 입력해 주세요.")
        return redirect('community:detail', post_id=comment.free_board.id)

    return redirect('community:detail', post_id=comment.free_board.id)

def comment_delete(request, pk):
    """
    댓글 삭제 AJAX 뷰.
    """
    if not request.user.is_authenticated:
        return JsonResponse({'error': '로그인 필요'}, status=401)

    comment = get_object_or_404(FreeBoardComment, pk=pk, is_deleted=False)
    if request.user != comment.user:
        return JsonResponse({'error': '삭제 권한 없음'}, status=403)

    if request.method == 'POST':
        with transaction.atomic():
            comment.is_deleted = True
            comment.save(update_fields=['is_deleted'])
            post = comment.free_board
            post.comments_count = max(0, post.comments_count - 1)
            post.save(update_fields=['comments_count'])

        return JsonResponse({'status': 'success', 'comment_id': pk, 'comments_count': post.comments_count})

    return JsonResponse({'error': '잘못된 요청'}, status=400)