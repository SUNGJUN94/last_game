from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.http import JsonResponse
from .models import User
import logging

logger = logging.getLogger(__name__)

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            logger.info(f"User {username} logged in successfully")
            return render(request, 'account/login_success.html', {'user': user})
        else:
            logger.warning(f"Failed login attempt for {username}")
            return render(request, 'account/login.html', {'error': '잘못된 아이디 또는 비밀번호입니다.'})
    return render(request, 'account/login.html')

def logout_view(request):
    if request.method == 'POST':
        logout(request)
        logger.info("User logged out successfully")
        return JsonResponse({'status': 'success'}, status=200)
    logout(request)
    return redirect('account:login')

def signup_view(request):
    if request.method == 'POST':
        logger.debug(f"POST data: {request.POST}")
        logger.debug(f"FILES data: {request.FILES}")
        login_id = request.POST['login_id']
        password = request.POST['pwd']
        pwd_confirm = request.POST['pwd_confirm']
        nickname = request.POST['nickname']
        email = request.POST['email']
        profile_image = request.FILES.get('profile_image')

        if password != pwd_confirm:
            return render(request, 'signup.html', {'error': '비밀번호가 일치하지 않습니다.'})

        if User.objects.filter(login_id=login_id).exists():
            return render(request, 'signup.html', {'error': '이미 사용 중인 아이디입니다.'})
        if User.objects.filter(nickname=nickname).exists():
            return render(request, 'signup.html', {'error': '이미 사용 중인 닉네임입니다.'})
        if User.objects.filter(email=email).exists():
            return render(request, 'signup.html', {'error': '이미 사용 중인 이메일입니다.'})

        user = User.objects.create_user(
            login_id=login_id,
            email=email,
            nickname=nickname,
            password=password
        )
        if profile_image:
            user.profile_image = profile_image
            user.save()
            logger.info(f"User {login_id} signed up with profile image: {user.profile_image.url}")
        else:
            logger.info(f"User {login_id} signed up with default profile image")
        login(request, user)
        return redirect('main:main')
    return render(request, 'signup.html')