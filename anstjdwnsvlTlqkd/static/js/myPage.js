document.addEventListener('DOMContentLoaded', () => {
  // 로그인 여부 판단
  const isAuthenticated = document.body.dataset.isAuthenticated === 'true';

  if (!isAuthenticated) {
    sessionStorage.setItem('prevPage', 'mypage');
    window.location.href = '/account/login/';
    return;
  }

  const dropdownBtn = document.getElementById('dropdownMenuBtn');
  const dropdownMenu = document.querySelector('#mypageDropdown .dropdown-menu');
  const currentLabel = document.getElementById('currentMenuLabel');
  const profileViewBtn = document.querySelector('.profile-view-btn');
  const nicknameEl = document.getElementById('nickname');
  const openGreetingInputBtn = document.getElementById('openGreetingInputBtn');
  const greetingInputContainer = document.getElementById('greetingInputContainer');
  const cancelGreetingInputBtn = document.getElementById('cancelGreetingInputBtn');
  const confirmGreetingBtn = document.getElementById('confirmGreetingBtn');
  const scrollTopBtn = document.getElementById('scrollTopBtn');

  const contentMap = {
    '마이페이지': 'content-mypage',
    '프로필': 'content-profile',
    '내 스페이스': 'content-space',
    '계정 및 보안': 'content-security'
  };

  // 디버깅 로그
  console.log('Elements initialized:', {
    greetingInputContainer,
    openGreetingInputBtn,
    cancelGreetingInputBtn,
    confirmGreetingBtn
  });

  // 드롭다운 토글
  dropdownBtn?.addEventListener('click', () => {
    dropdownMenu.style.display =
      dropdownMenu.style.display === 'none' || dropdownMenu.style.display === ''
        ? 'block'
        : 'none';
  });

  // 메뉴 클릭 시 콘텐츠 전환
  dropdownMenu?.querySelectorAll('.dropdown-item').forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const label = item.getAttribute('data-label');
      currentLabel.textContent = label;
      dropdownMenu.style.display = 'none';

      Object.values(contentMap).forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
      });

      const selected = contentMap[label];
      if (selected) {
        const el = document.getElementById(selected);
        if (el) el.style.display = 'block';
      }
    });
  });

  // 외부 클릭 시 드롭다운 닫기
  document.addEventListener('click', (e) => {
    if (!document.getElementById('mypageDropdown')?.contains(e.target)) {
      dropdownMenu.style.display = 'none';
    }
  });

  // 닉네임 출력
  const userData = sessionStorage.getItem('user');
  if (userData && nicknameEl) {
    try {
      const user = JSON.parse(userData);
      nicknameEl.textContent = user.nickname || '사용자';
    } catch (e) {
      console.warn('닉네임 파싱 실패:', e);
    }
  }

  // 프로필 보기 버튼
  profileViewBtn?.addEventListener('click', () => {
    Object.values(contentMap).forEach(id => {
      const el = document.getElementById(id);
      if (el) el.style.display = 'none';
    });
    const profileEl = document.getElementById(contentMap['프로필']);
    if (profileEl) profileEl.style.display = 'block';
    currentLabel.textContent = '프로필';
  });

  // 스크롤 버튼 동작
  window.addEventListener('scroll', () => {
    if (window.scrollY > 200) {
      scrollTopBtn.style.display = 'flex';
    } else {
      scrollTopBtn.style.display = 'none';
    }
  });

  scrollTopBtn?.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  // 인사 메시지 입력창 열기
  openGreetingInputBtn?.addEventListener('click', () => {
    if (greetingInputContainer) {
      console.log('Opening greeting input');
      greetingInputContainer.classList.add('show');
    } else {
      console.error('Greeting input container not found');
    }
  });

  // 인사 메시지 입력창 닫기
  cancelGreetingInputBtn?.addEventListener('click', () => {
    if (greetingInputContainer) {
      greetingInputContainer.classList.remove('show');
    }
  });

  // 인사 메시지 저장
  confirmGreetingBtn?.addEventListener('click', async () => {
    const greetingInput = document.getElementById('greetingInput')?.value || '';
    const url = openGreetingInputBtn?.dataset.url || '/mypage/update-greeting/';
    const token = openGreetingInputBtn?.dataset.csrf || document.querySelector('[name=csrfmiddlewaretoken]')?.value;

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'X-CSRFToken': token,
        },
        body: new URLSearchParams({
          'greeting_message': greetingInput,
        }),
      });

      const data = await response.json();
      if (data.success) {
        alert(data.message);
        if (greetingInputContainer) {
          greetingInputContainer.classList.remove('show');
          location.reload();
        }
      } else {
        alert(data.message);
      }
    } catch (error) {
      console.error('Error saving greeting:', error);
      alert('서버와의 통신 중 오류가 발생했습니다.');
    }
  });

  // CSRF 토큰
  const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value;
  if (!csrfToken) {
    console.error('CSRF token not found');
    alert('페이지 로드 중 문제가 발생했습니다. 새로고침해 주세요.');
    return;
  }
  //add1
  // 차단 해제 이벤트 (이벤트 위임)
  const userList = document.querySelector('#blocked .user-list');
  if (userList) {
    userList.addEventListener('click', async (e) => {
      const unblockBtn = e.target.closest('.unblock-btn');
      if (!unblockBtn) return;

      e.preventDefault();
      const blockedId = unblockBtn.getAttribute('data-blocked-id');
      const userName = unblockBtn.parentElement.querySelector('.user-name')?.textContent || '사용자';

      // blockedId 디버깅: 팝업으로 표시
      if (blockedId) {
        alert(`차단 해제 대상 사용자 ID: ${blockedId}`);
      } else {
        alert('오류: blocked_id를 가져올 수 없습니다. (undefined 또는 누락)');
        console.error('Blocked ID is missing for unblock button:', {
          unblockBtn,
          parentElement: unblockBtn.parentElement,
        });
      }

      // 기존 차단 해제 로직 (주석 처리하여 팝업 테스트에 집중)
      /*
      if (!blockedId) {
        console.error('Blocked ID is missing for unblock button:', {
          unblockBtn,
          parentElement: unblockBtn.parentElement,
        });
        alert('차단 해제에 필요한 사용자 ID가 누락되었습니다. 관리자에게 문의하세요.');
        return;
      }

      console.log('Attempting to unblock user:', { blockedId, userName });

      if (!confirm(`${userName}님의 차단을 해제하시겠습니까?`)) {
        return;
      }

      try {
        const response = await fetch(`/mypage/unblock/${blockedId}/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-CSRFToken': csrfToken,
          },
        });

        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Unblock response:', data);

        if (data.status === 'success') {
          alert(data.message);
          // 차단 목록 업데이트
          const blockedUsers = data.blocked_users || [];
          if (blockedUsers.length > 0) {
            userList.innerHTML = blockedUsers
              .map(
                (user) => `
                  <div class="user-item">
                    <span class="user-name">${user.nickname}</span>
                    <a href="#" class="unblock-btn" data-blocked-id="${user.id}">차단 해제</a>
                  </div>
                `
              )
              .join('');
          } else {
            userList.innerHTML = '<div class="user-item">차단한 유저가 없습니다.</div>';
          }
        } else {
          console.error('Unblock failed:', data.error);
          alert(data.error || '차단 해제에 실패했습니다.');
        }
      } catch (error) {
        console.error('Unblock error:', error);
        alert('서버와의 연결에 문제가 발생했습니다. 다시 시도해주세요.');
      }
      */
    });
  } else {
    console.error('User list element not found');
  }
  //add2
});