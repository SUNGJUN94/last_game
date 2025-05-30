document.addEventListener('DOMContentLoaded', () => {
    const isAuthenticated = document.body.dataset.isAuthenticated === 'true';

    if (!isAuthenticated) {
        sessionStorage.setItem('prevPage', 'mypage');
        window.location.href = '/account/login/';
        return;
    }

    const dropdownBtn = document.getElementById('dropdownMenuBtn');
    const dropdownMenu = document.querySelector('#mypageDropdown .dropdown-menu');
    const currentLabel = document.getElementById('currentMenuLabel');
    const nicknameEl = document.getElementById('nickname');

    const contentMap = {
        '마이페이지': 'content-mypage',
        '예측 종목': 'content-profile',
        '내가 쓴 글': 'content-space',
        '차단 계정': 'content-security'
    };



    dropdownBtn.addEventListener('click', () => {
      dropdownMenu.style.display = (dropdownMenu.style.display === 'none' || dropdownMenu.style.display === '') 
        ? 'block' 
        : 'none';
    });



    dropdownMenu.querySelectorAll('.dropdown-item').forEach(item => {
      item.addEventListener('click', (e) => {
        e.preventDefault();
        const label = item.getAttribute('data-label');
        currentLabel.textContent = label;
        dropdownMenu.style.display = 'none';
    
        // ✅ 콘텐츠 영역 스크롤 이동
        switch (label) {
          case '마이페이지':
            window.scrollTo({ top: 0, behavior: 'smooth' });
            break;
          case '관심종목':
            scrollToWithOffset('predictionItemsUl', 80); // ← 이걸로 변경!
            break;
          case '내가 쓴 글':
            scrollToWithOffset('myPostsList', 80);
            break;
          case '차단 계정':
            scrollToWithOffset('blockuser', 80);
            break;
        }
      });
    });  // ← forEach 콜백 종료
    function scrollToWithOffset(elementId, offset = 60) {
      const el = document.getElementById(elementId);
      if (!el) return;
      
      const rect = el.getBoundingClientRect();
      const absoluteY = window.scrollY + rect.top - offset; // offset 만큼 위로 여유 공간 확보
    
      window.scrollTo({
        top: absoluteY,
        behavior: 'smooth'
      });
    }
  
    // 바깥 클릭 시 드롭다운 닫기
    document.addEventListener('click', (e) => {
      if (!document.getElementById('mypageDropdown').contains(e.target)) {
        dropdownMenu.style.display = 'none';
      }
    });

    const userData = sessionStorage.getItem('user');
    if (userData) {
      try {
        const user = JSON.parse(userData);
        const nickname = user.nickname || '사용자';
        document.getElementById('nickname').textContent = nickname;
        document.getElementById('profileNickname').textContent = user.nickname || '닉네임';
      } catch (e) {
        console.warn('닉네임 파싱 실패:', e);
      }
    }

       // "프로필 보기" 버튼 클릭 시 → 프로필 콘텐츠 보여주고 라벨 변경
const profileViewBtn = document.querySelector('.profile-view-btn');
profileViewBtn?.addEventListener('click', () => {
  // 콘텐츠 전부 숨기기
  Object.values(contentMap).forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });

  // 프로필 콘텐츠 보이기
  const profileEl = document.getElementById(contentMap['예측 종목']);
  if (profileEl) profileEl.style.display = 'block';

  // 드롭다운 버튼 라벨도 "프로필"로 바꾸기
  currentLabel.textContent = '예측 종목';
});
renderPredictionItems('predictionItemsUl'); // 마이페이지에서 예측 항목 시세 호출
//renderPredictionItems('predictionItemsProfileUl'); // 예측 종목 드롭다운에서 예측 항목 시세 호출
renderMyPosts('myPostsList'); // 마이페이지에서 내가 쓴 글
//renderMyPosts('myPostsList2'); // 내가 쓴 글 드롭다운에서 내가 쓴 글
renderBlockedUsers('blockedUsersList'); // 마이페이지에서 차단 목록
//renderBlockedUsers('blockedUsersList2'); // 차단 목록 드롭다운에서 차단 목록
});

// 예측 항목 시세
function renderPredictionItems(targetId) {
  const predictionItems = [
    { name: '삼성전자', price: '82,000원', change: '+1.20%', link: 'stock_detail.html?item=삼성전자' },
    { name: '비트코인', price: '125,000,000원', change: '+0.80%', link: 'crypto_detail.html?item=비트코인' },
    { name: '비트코인1', price: '125,000,000원', change: '+0.80%', link: 'crypto_detail.html?item=비트코인' },
    { name: '비트코인2', price: '125,000,000원', change: '+0.80%', link: 'crypto_detail.html?item=비트코인' },
    { name: '비트코인3', price: '125,000,000원', change: '+0.80%', link: 'crypto_detail.html?item=비트코인' },
    { name: '비트코인4', price: '125,000,000원', change: '+0.80%', link: 'crypto_detail.html?item=비트코인' },
    { name: '비트코인5', price: '125,000,000원', change: '+0.80%', link: 'crypto_detail.html?item=비트코인' },
    { name: '비트코인6', price: '125,000,000원', change: '+0.80%', link: 'crypto_detail.html?item=비트코인' },
    { name: '비트코인7', price: '125,000,000원', change: '+0.80%', link: 'crypto_detail.html?item=비트코인' },
  ];

  const listContainer = document.getElementById(targetId);
  listContainer.innerHTML = ''; // 초기화

  if (predictionItems.length === 0) {
    listContainer.innerHTML = `<li class="text-muted">예측 항목이 없습니다.</li>`;
  } else {
    predictionItems.forEach(item => {
      const li = document.createElement('li');   
      li.className = ' d-flex justify-content-between align-items-center mb-2';
      // li.style.border = '1px solid #e0e0e0'; border delete
      // li.style.borderRadius = '10px';
      li.style.padding = '12px 16px';
      li.style.fontSize = '15px';
      li.style.backgroundColor = '#fff';
      li.style.boxShadow = '0 1px 2px rgba(0, 0, 0, 0.03)';
      li.style.cursor = 'pointer';                                    // 이거 나중에 장고에 추가할때는 html파일에 바로 for문으로 때려넣던데 그럼 li 태그 안에 직접 이 style들 적용시켜야함

      li.textContent = `${item.name} ${item.price} (${item.change})`;

      // 👉 클릭 시 해당 링크로 이동
      li.addEventListener('click', () => {
        window.location.href = item.link;
      });

      listContainer.appendChild(li);
    });
  }
}

// 내가 쓴 글
function renderMyPosts(targetId) {
  const userPosts = [
    { title: '분기보고서 (2025.03)', time_ago: '6분 전', url: 'post_detail.html?id=1' },
    { title: '분기보고서 (2025.03)', time_ago: '6분 전', url: 'post_detail.html?id=2' },
    { title: '분기보고서 (2025.04)', time_ago: '6분 전', url: 'post_detail.html?id=3' },
    { title: '분기보고서 (2025.05)', time_ago: '6분 전', url: 'post_detail.html?id=3' },
    { title: '분기보고서 (2025.06)', time_ago: '6분 전', url: 'post_detail.html?id=3' },
    { title: '분기보고서 (2025.07)', time_ago: '6분 전', url: 'post_detail.html?id=3' },
    { title: '분기보고서 (2025.08)', time_ago: '6분 전', url: 'post_detail.html?id=3' },
    { title: '분기보고서 (2025.08)', time_ago: '6분 전', url: 'post_detail.html?id=3' },
    { title: '분기보고서 (2025.09)', time_ago: '6분 전', url: 'post_detail.html?id=3' },
    { title: '분기보고서 (2025.10)', time_ago: '6분 전', url: 'post_detail.html?id=3' },
    { title: '분기보고서 (2025.10)', time_ago: '6분 전', url: 'post_detail.html?id=3' },
    { title: '분기보고서 (2025.10)', time_ago: '6분 전', url: 'post_detail.html?id=3' },
    { title: '분기보고서 (2025.10)', time_ago: '6분 전', url: 'post_detail.html?id=3' },
  ];

  const listContainer = document.getElementById(targetId);
  listContainer.innerHTML = '';

  if (userPosts.length === 0) {
    listContainer.innerHTML = `<li class="list-group-item text-muted">작성한 게시물이 없습니다.</li>`;
    return;
  }

  userPosts.forEach(post => {
    const li = document.createElement('li');
    li.className = 'list-group-item d-flex justify-content-between align-items-center';
    li.style.fontSize = '15px';
    li.style.padding = '10px 16px';
    li.style.cursor = 'pointer';

    // 👉 한 줄에 출력되도록 innerHTML로 처리
    li.innerHTML = `
      <a href="${post.url}" class="text-dark text-decoration-none flex-grow-1">
        ${post.title} ${post.time_ago}
      </a>
      <span class="text-muted" style="font-size: 13px; white-space: nowrap;">${post.time_ago}</span>
    `;

    listContainer.appendChild(li);
  });
}


  // 스크롤 최상단 이동 버튼 기능
  const scrollTopBtn = document.getElementById('scrollTopBtn');

  // 스크롤 내릴 때 버튼 보이기
  window.addEventListener('scroll', () => {
    if (window.scrollY > 80) {
      scrollTopBtn.style.display = 'block';
    } else {
      scrollTopBtn.style.display = 'none';
    }
  });
  
  // 버튼 클릭하면 맨 위로 부드럽게 이동
  scrollTopBtn.addEventListener('click', () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  });