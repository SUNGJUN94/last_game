# mypage/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect, JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import logging
import json # AJAX 요청 처리 시 필요할 수 있음

from account.models import User, BlockedUser # User 모델 경로 확인 필요
from community.models import FreeBoard # community 앱 모델 경로 확인 필요
from predict_info.models import FavoriteStock, PredictedStockPrice # predict_info 앱 모델 임포트


logger = logging.getLogger(__name__)

@login_required
def unblock_user(request, blocked_id):
    if request.method == 'POST':
        blocker = request.user
        try:
            # user_id 필드가 User 모델의 기본 키(pk)가 아닐 경우, 해당 필드로 조회
            # User 모델에 login_id, user_id 등 여러 ID 필드가 있다면 정확한 필드명 사용
            blocked_user = get_object_or_404(User, pk=blocked_id) # 또는 user_id=blocked_id 등
        except User.DoesNotExist:
            logger.warning(f"Attempted to unblock non-existent user_id: {blocked_id}")
            messages.error(request, "차단 해제하려는 사용자를 찾을 수 없습니다.")
            return redirect('mypage:mypage')

        block_relation = BlockedUser.objects.filter(blocker=blocker, blocked=blocked_user).first()
        if not block_relation:
            logger.warning(f"No block relation found for blocker: {blocker.pk}, blocked: {blocked_id}")
            messages.info(request, "해당 사용자와의 차단 관계를 찾을 수 없습니다.")
            return redirect('mypage:mypage')
        
        block_relation.delete()
        logger.info(f"Blocked user {blocked_user.nickname} (ID: {blocked_id}) unblocked by {blocker.nickname}")
        messages.success(request, f"{blocked_user.nickname}님의 차단을 해제했습니다.")
        return redirect('mypage:mypage')
    
    return redirect('mypage:mypage')


@login_required
def mypage_view(request):
    user = request.user

    # 내가 쓴 글 (기존 로직 유지)
    my_posts_qs = FreeBoard.objects.filter(user=user, is_deleted=False).order_by('-reg_dt')
    posts_paginator = Paginator(my_posts_qs, 10)
    page_number = request.GET.get('page', 1)
    try: my_posts_page_obj = posts_paginator.page(page_number)
    except (EmptyPage, PageNotAnInteger): my_posts_page_obj = posts_paginator.page(1)
    
    total_pages = posts_paginator.num_pages
    current_page = my_posts_page_obj.number
    start_page_posts = ((current_page - 1) // 10) * 10 + 1
    end_page_posts = min(start_page_posts + 9, total_pages)
    page_range_posts = range(start_page_posts, end_page_posts + 1)

    # 차단 목록 (기존 로직 유지)
    blocked_users_qs = BlockedUser.objects.filter(blocker=user).select_related('blocked').order_by('-created_at')
    blocked_paginator = Paginator(blocked_users_qs, 5)
    blocked_page_number = request.GET.get('blocked_page', 1)
    try: blocked_page_obj = blocked_paginator.page(blocked_page_number)
    except (EmptyPage, PageNotAnInteger): blocked_page_obj = blocked_paginator.page(1)

    blocked_total_pages = blocked_paginator.num_pages
    current_blocked_page = blocked_page_obj.number
    start_page_blocked = ((current_blocked_page - 1) // 10) * 10 + 1
    end_page_blocked = min(start_page_blocked + 9, blocked_total_pages)
    blocked_page_range = range(start_page_blocked, end_page_blocked + 1)

    # --- 관심 종목 목록 가져오기 ---
    favorite_stocks_list = FavoriteStock.objects.filter(user=user).order_by('stock_name')
    # 페이지네이션은 필요시 추가 (여기서는 전체 목록을 가져옴)

    context = {
        'my_posts_page_obj': my_posts_page_obj,
        'page_range_posts': page_range_posts, # 템플릿 변수명 변경
        'total_pages_posts': total_pages, # 템플릿 변수명 변경
        'user': user,
        'blocked_page_obj': blocked_page_obj,
        'blocked_page_range': blocked_page_range,
        'blocked_total_pages': blocked_total_pages,
        'favorite_stocks_list': favorite_stocks_list, # 관심 종목 목록 추가
    }
    return render(request, 'mypage.html', context)


@login_required
def get_favorite_stock_prediction_ajax(request):
    stock_code = request.GET.get('stock_code')
    if not stock_code:
        return JsonResponse({'error': '종목 코드가 필요합니다.'}, status=400)

    try:
        # 해당 종목의 가장 최근 예측 기준일의 예측 데이터 가져오기
        latest_prediction_base_date_entry = PredictedStockPrice.objects.filter(
            stock_code=stock_code
        ).order_by('-prediction_base_date').first()

        if not latest_prediction_base_date_entry:
            return JsonResponse({'error': f"'{stock_code}'에 대한 저장된 예측 결과를 찾을 수 없습니다."}, status=404)

        prediction_base_date = latest_prediction_base_date_entry.prediction_base_date
        
        # 같은 기준일의 모든 예측(보통 5일치) 가져오기
        predictions = PredictedStockPrice.objects.filter(
            stock_code=stock_code,
            prediction_base_date=prediction_base_date
            # analysis_type은 필요에 따라 필터링 (여기서는 기본 'technical'로 가정하거나, 모든 타입 중 최신)
            # 여기서는 특정 analysis_type을 가정하지 않고, 해당 기준일의 모든 예측을 가져옴
            # 만약 여러 analysis_type이 있다면, 클라이언트에서 선택하게 하거나 기본값을 정해야 함
        ).order_by('predicted_date')

        if not predictions.exists():
            return JsonResponse({'error': f"'{stock_code}' (기준일: {prediction_base_date})에 대한 예측 데이터 구성에 문제가 있습니다."}, status=404)

        predictions_data = [
            {'date': p.predicted_date.strftime('%Y-%m-%d'), 'price': round(float(p.predicted_price))}
            for p in predictions
        ]
        
        # 종목명도 함께 전달
        stock_info = FavoriteStock.objects.filter(user=request.user, stock_code=stock_code).first()
        stock_name = stock_info.stock_name if stock_info else stock_code

        return JsonResponse({
            'stock_code': stock_code,
            'stock_name': stock_name,
            'prediction_base_date': prediction_base_date.strftime('%Y-%m-%d'),
            'predictions': predictions_data,
        })

    except Exception as e:
        logger.error(f"Error fetching prediction for favorite stock {stock_code}: {e}")
        traceback.print_exc()
        return JsonResponse({'error': '예측 데이터를 가져오는 중 오류가 발생했습니다.'}, status=500)


@login_required
def update_greeting_message(request):
    if request.method == 'POST':
        greeting_message = request.POST.get('greeting_message', '').strip()
        if len(greeting_message) > 100:
            return JsonResponse({'success': False, 'message': '인사 메시지는 최대 100자까지 가능합니다.'}, status=400)
        try:
            user = request.user
            user.greeting_message = greeting_message
            user.save(update_fields=['greeting_message'])
            return JsonResponse({'success': True, 'message': '인사 메시지가 성공적으로 저장되었습니다.'})
        except Exception as e:
            logger.error(f"Failed to update greeting message for user {request.user.login_id}: {str(e)}")
            return JsonResponse({'success': False, 'message': '저장 중 오류가 발생했습니다.'}, status=500)
    return JsonResponse({'success': False, 'message': '잘못된 요청입니다.'}, status=400)


@login_required
def edit_profile_view(request):
    # storage = messages.get_messages(request) # 메시지 사용 시 주석 해제
    # list(storage) 

    user = request.user
    if request.method == 'POST':
        # 기존 프로필 수정 로직 ...
        # (이 부분은 이전 코드와 동일하게 유지한다고 가정)
        name = request.POST.get('name', '').strip()
        nickname = request.POST.get('nickname', '').strip()
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')
        profile_image = request.FILES.get('profile_image')

        if not name or len(name) < 2:
            messages.error(request, '이름은 최소 2자 이상이어야 합니다.')
            return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})
        
        if not nickname or len(nickname) < 2:
            messages.error(request, '닉네임은 최소 2자 이상이어야 합니다.')
            return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})

        if nickname != user.nickname and User.objects.filter(nickname=nickname).exclude(pk=user.pk).exists():
            messages.error(request, '이미 사용 중인 닉네임입니다.')
            return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})

        if password:
            if len(password) < 8:
                messages.error(request, '비밀번호는 최소 8자 이상이어야 합니다.')
                return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})
            if password != password_confirm:
                messages.error(request, '비밀번호가 일치하지 않습니다.')
                return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})
            user.set_password(password)

        user.name = name
        user.nickname = nickname
        
        if profile_image:
            # 기존 이미지 삭제 로직 (필요시)
            if user.profile_image and hasattr(user.profile_image, 'name') and user.profile_image.name != 'profile_images/default.jpg':
                from django.core.files.storage import default_storage
                if default_storage.exists(user.profile_image.name):
                    try:
                        default_storage.delete(user.profile_image.name)
                    except Exception as e:
                         logger.error(f"Failed to delete old profile image for {user.login_id}: {str(e)}")
            user.profile_image = profile_image
        
        try:
            user.save()
            messages.success(request, '프로필이 성공적으로 업데이트되었습니다.')
            if password: # 비밀번호 변경 시 재로그인하여 세션 갱신
                from django.contrib.auth import login as auth_login
                auth_login(request, user)
            return redirect('mypage:mypage')
        except Exception as e:
            logger.error(f"Failed to save user profile for {user.login_id}: {str(e)}")
            messages.error(request, '프로필 저장 중 오류가 발생했습니다.')
            
    return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})
