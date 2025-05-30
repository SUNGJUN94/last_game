from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.http import JsonResponse
from .models import User
import logging

logger = logging.getLogger(__name__)

#add1
def find_pw_view(request):
    step = 1
    login_id = name = email = ''
    id_error = id_success = info_error = info_success = pwd_error = None

    if request.method == 'POST':
        logger.debug(f"POST data: {request.POST}")
        action = request.POST.get('action')
        login_id = request.POST.get('login_id', '').strip()
        name = request.POST.get('name', '').strip()
        email = request.POST.get('email', '').strip()

        if action == 'check_id':
            logger.info(f"Checking login_id: {login_id}")
            if not login_id:
                id_error = "아이디를 입력해 주세요."
            elif not User.objects.filter(login_id=login_id).exists():
                id_error = "존재하지 않는 아이디입니다."
            else:
                id_success = "존재하는 아이디입니다. 이름과 이메일을 입력해 주세요."
                step = 2
        elif action == 'check_info':
            logger.info(f"Checking info - login_id: {login_id}, name: {name}, email: {email}")
            if not login_id or not name or not email:
                info_error = "모든 필드를 입력해 주세요."
                step = 2
            else:
                try:
                    user = User.objects.get(login_id=login_id, name=name, email=email)
                    info_success = "본인 확인이 완료되었습니다. 새 비밀번호를 입력해 주세요."
                    step = 3
                except User.DoesNotExist:
                    info_error = "입력하신 정보가 일치하지 않습니다."
                    step = 2
        elif action == 'reset_pwd':
            logger.info(f"Resetting password for login_id: {login_id}")
            pwd = request.POST.get('pwd', '')
            pwd_confirm = request.POST.get('pwd_confirm', '')
            if not login_id or not name or not email:
                pwd_error = "비정상적인 접근입니다. 처음부터 다시 시도해 주세요."
                step = 1
            elif len(pwd) < 8:
                pwd_error = "비밀번호는 8자 이상이어야 합니다."
                step = 3
            elif pwd != pwd_confirm:
                pwd_error = "비밀번호가 일치하지 않습니다."
                step = 3
            else:
                try:
                    user = User.objects.get(login_id=login_id, name=name, email=email)
                    user.set_password(pwd)
                    user.save()
                    logger.info(f"Password reset successful for {login_id}")
                    step = 4  # 완료
                except User.DoesNotExist:
                    pwd_error = "비정상적인 접근입니다. 처음부터 다시 시도해 주세요."
                    step = 1

    return render(request, 'find_pw.html', {
        'step': step,
        'login_id': login_id,
        'name': name,
        'email': email,
        'id_error': id_error,
        'id_success': id_success,
        'info_error': info_error,
        'info_success': info_success,
        'pwd_error': pwd_error,
    })

def find_id_view(request):
    found_id = None
    error = None
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        email = request.POST.get('email', '').strip()
        try:
            user = User.objects.get(name=name, email=email)
            found_id = user.login_id
        except User.DoesNotExist:
            error = "일치하는 회원 정보가 없습니다."
    return render(request, 'find_id.html', {'found_id': found_id, 'error': error})
#add2

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
        name = request.POST.get('name', '')  # 이름 받기

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
            password=password,
            name=name
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