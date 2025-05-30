document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        initializeHeaderFeatures();
        fetchFooter();
        if (typeof toggleMainTab === 'function' && document.getElementById('btn-main-popular')) {
            toggleMainTab('popular');
        }
        initializeScrollToTopButton();
        initializeBanners();
        initializeRealtimeSearch();
        if (typeof renderOriginalsList === 'function' && document.getElementById('originals-list')) {
            renderOriginalsList();
        }
        initializeHeadlines();
        if (typeof renderCoinInfoList === 'function' && document.getElementById('coin-info-list')) {
            renderCoinInfoList();
        }
        if (typeof renderNoticeList === 'function' && document.getElementById('notice-list')) {
            renderNoticeList();
        }
        initializeExchangeList();
        if (typeof renderCoinList === 'function' && document.getElementById('coin-list')) {
            renderCoinList();
        }
        initializeWriteButton();
        fetchNaverNews();
        setInterval(fetchNaverNews, 5 * 60 * 1000);

        const coinSearchInput = document.getElementById('coinSearch');
        const prevPageBtn = document.getElementById('prevPage');
        const nextPageBtn = document.getElementById('nextPage');

        if (coinSearchInput) {
            coinSearchInput.addEventListener('input', (e) => {
                const keyword = e.target.value.trim().toLowerCase();
                filteredCoins = coinData.filter(coin => coin.name.toLowerCase().includes(keyword));
                currentPage = 1;
                renderCoinList();
            });
        }
        if (prevPageBtn) {
            prevPageBtn.addEventListener('click', () => {
                if (currentPage > 1) { currentPage--; renderCoinList(); }
            });
        }
        if (nextPageBtn) {
            nextPageBtn.addEventListener('click', () => {
                const totalPages = Math.ceil(filteredCoins.length / itemsPerPage);
                if (currentPage < totalPages) { currentPage++; renderCoinList(); }
            });
        }
        if (document.getElementById('coin-list')) renderCoinList();
    }, 0);
});

function initializeHeaderFeatures() {
    const alertIcon = document.getElementById('alertIcon');
    const profileIcon = document.getElementById('profileIcon');
    const isLoggedIn = document.body.dataset.isAuthenticated === 'true';

    console.log('initializeHeaderFeatures called');
    console.log('isLoggedIn:', isLoggedIn);
    console.log('alertIcon:', alertIcon);
    console.log('profileIcon:', profileIcon);

    if (isLoggedIn) {
        if (alertIcon) {
            alertIcon.addEventListener('click', (e) => {
                e.stopPropagation();
                console.log('alertIcon clicked');
                togglePopup('notification-popup');
            });
        } else {
            console.warn('alertIcon not found');
        }

        if (profileIcon) {
            profileIcon.addEventListener('click', (e) => {
                e.stopPropagation();
                console.log('profileIcon clicked');
                togglePopup('profile-popup');
            });
        } else {
            console.warn('profileIcon not found');
        }
    } else {
        console.log('User is not logged in, skipping header feature initialization');
    }

    document.addEventListener('click', (event) => {
        const profilePopup = document.getElementById('profile-popup');
        const notificationPopup = document.getElementById('notification-popup');
        if (profilePopup && !profilePopup.contains(event.target) && profileIcon && !profileIcon.contains(event.target)) {
            profilePopup.style.display = 'none';
        }
        if (notificationPopup && !notificationPopup.contains(event.target) && alertIcon && !alertIcon.contains(event.target)) {
            notificationPopup.style.display = 'none';
        }
    });
}

function togglePopup(id) {
    const popup = document.getElementById(id);
    if (popup) {
        console.log(`Toggling popup: ${id}, current display: ${popup.style.display}`);
        popup.style.display = popup.style.display === 'block' ? 'none' : 'block';
        if (id === 'notification-popup' && popup.style.display === 'block') {
            loadNotifications();
        }
    } else {
        console.warn(`Popup with id ${id} not found`);
    }
}

function performLogout() {
    const csrftoken = getCookie('csrftoken');
    const logoutUrl = document.body.dataset.logoutUrl || '/logout/';
    fetch(logoutUrl, {
        method: 'POST',
        headers: {
            'X-CSRFToken': csrftoken,
            'Content-Type': 'application/json',
        },
        credentials: 'include',
    })
    .then(response => {
        if (response.ok) {
            window.location.href = document.body.dataset.mainUrl || '/';
        } else {
            console.error('로그아웃 실패');
            alert('로그아웃 중 오류가 발생했습니다.');
        }
    })
    .catch(error => {
        console.error('로그아웃 오류:', error);
        alert('로그아웃 중 오류가 발생했습니다.');
    });
}

function loadNotifications() {
    const csrftoken = getCookie('csrftoken');
    fetch('/community/notifications/', {
        method: 'GET',
        headers: {
            'X-CSRFToken': csrftoken,
            'Content-Type': 'application/json',
        },
        credentials: 'include',
    })
    .then(response => response.json())
    .then(data => {
        const body = document.getElementById('notification-body');
        body.innerHTML = '';
        let unreadCount = 0;
        if (data.length > 0) {
            data.forEach(notification => {
                if (!notification.is_read) unreadCount++;
                const item = document.createElement('div');
                item.className = 'notification-item p-2 border-bottom';
                item.style.cursor = 'pointer';
                item.style.transition = 'background 0.2s';
                item.style.backgroundColor = notification.is_read ? '#fff' : '#f0f8ff';
                item.onclick = () => {
                    window.location.href = notification.link;
                };
                item.innerHTML = `
                    <div><span class="comment-user" style="font-weight: bold; color: #007bff;">${notification.user}</span>님이 작성한 댓글</div>
                    <span class="comment-preview" style="font-size: 14px; color: #495057; display: block; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px;">${notification.preview}</span>
                    <div class="comment-time" style="font-size: 12px; color: #6c757d; margin-top: 2px;">${notification.time}</div>
                `;
                body.appendChild(item);
            });
        } else {
            body.innerHTML = '<div class="p-2 text-muted">새로운 알림이 없습니다.</div>';
        }
        const unreadBadge = document.getElementById('unread-count');
        if (unreadCount > 0) {
            unreadBadge.textContent = unreadCount;
            unreadBadge.style.display = 'block';
        } else {
            unreadBadge.style.display = 'none';
        }
    })
    .catch(error => console.error('알림 로드 오류:', error));
}

function closeNotificationPopup() {
    const popup = document.getElementById('notification-popup');
    if (popup) {
        popup.style.display = 'none';
    }
}

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function fetchFooter() {
    const footerContainer = document.getElementById('footer-container');
    if (footerContainer) {
        const footerItems = footerContainer.querySelectorAll('.footer-item');
        if (footerItems.length > 0) {
            const currentPath = window.location.pathname.split('/').pop() || 'main';
            footerItems.forEach(item => {
                const href = item.getAttribute('href');
                if (href && (href === currentPath || (currentPath === 'main' && (href === '/' || href === 'main.html')))) {
                    footerItems.forEach(i => i.classList.remove('active'));
                    item.classList.add('active');
                }
            });

            footerItems.forEach(item => {
                item.addEventListener('click', (e) => {
                    footerItems.forEach(i => i.classList.remove('active'));
                    item.classList.add('active');
                });
            });
        }
    }
}

function initializeScrollToTopButton() {
    const scrollTopBtn = document.getElementById('scrollTopBtn');
    if (!scrollTopBtn) return;

    window.addEventListener('scroll', () => {
        if (window.scrollY > 80) {
            scrollTopBtn.style.display = 'flex';
        } else {
            scrollTopBtn.style.display = 'none';
        }
    });

    scrollTopBtn.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}

const banners = [
    { link: "https://example.com/banner1", img: "https://images.pexels.com/photos/6801874/pexels-photo-6801874.jpeg?auto=compress&cs=tinysrgb&w=400&h=300&dpr=1", alt: "금융 차트 배너" },
    { link: "https://example.com/banner2", img: "https://images.pexels.com/photos/6771894/pexels-photo-6771894.jpeg?auto=compress&cs=tinysrgb&w=400&h=300&dpr=1", alt: "디지털 코인 배너" },
    { link: "https://example.com/banner3", img: "https://images.pexels.com/photos/6694543/pexels-photo-6694543.jpeg?auto=compress&cs=tinysrgb&w=400&h=300&dpr=1", alt: "투자 분석 배너" }
];

//add1
function initializeBanners() {
    const carouselInner = document.getElementById('carousel-inner');
    const bannerCountDiv = document.getElementById('carousel-count');
    const carouselElement = document.getElementById('mainBannerCarousel');

    if (!carouselInner || !carouselElement || !bannerCountDiv) {
        console.error('Required elements not found:', { carouselInner, carouselElement, bannerCountDiv });
        return;
    }

    // 서버에서 제공한 admin_posts를 사용 (이미 HTML에서 렌더링됨)
    const items = carouselInner.getElementsByClassName('carousel-item');
    if (items.length > 0) {
        bannerCountDiv.textContent = `1 / ${items.length}`;
        const carousel = new bootstrap.Carousel(carouselElement, {
            interval: 3000,
            wrap: true
        });
        carouselElement.addEventListener('slide.bs.carousel', (e) => {
            const total = items.length;
            let current = e.to + 1;
            if (current > total) {
                current = 1;
            }
            bannerCountDiv.textContent = `${current} / ${total}`;
        });
    } else {
        bannerCountDiv.textContent = `0 / 0`;
    }
}
//add2

const realtimeItems = [
    { rank: 1, title: "밀크", link: "milk.html" }, { rank: 2, title: "cobak-token", link: "cobak-token.html" },
    { rank: 3, title: "bitcoin", link: "bitcoin.html" }, { rank: 4, title: "ethereum", link: "ethereum.html" },
    { rank: 5, title: "파일 암호화폐", link: "filecoin.html" }, { rank: 6, title: "리플", link: "ripple.html" },
    { rank: 7, title: "도지코인", link: "dogecoin.html" }, { rank: 8, title: "pump", link: "pump.html" },
    { rank: 9, title: "퀀텀", link: "quantum.html" }, { rank: 10, title: "메타플래닛", link: "metaplanet.html" }
];

function initializeRealtimeSearch() {
    const realtimeToggle = document.getElementById('realtimeToggle');
    const realtimeDropdown = document.getElementById('realtimeDropdown');
    const realtimeList = document.getElementById('realtimeList');
    const realtimeCloseButton = document.getElementById('realtimeClose');

    if (!realtimeToggle || !realtimeDropdown || !realtimeList || !realtimeCloseButton) return;

    realtimeItems.forEach(item => {
        const li = document.createElement('li');
        li.className = "mb-2";
        li.innerHTML = `<a href="${item.link}" class="text-primary text-decoration-none">${item.rank}. ${item.title}</a>`;
        realtimeList.appendChild(li);
    });

    realtimeToggle.addEventListener('click', toggleRealtime);
    realtimeCloseButton.addEventListener('click', () => {
        realtimeDropdown.classList.add('d-none');
        realtimeToggle.classList.remove('d-none');
    });
}

function toggleRealtime() {
    const realtimeDropdown = document.getElementById('realtimeDropdown');
    const realtimeToggle = document.getElementById('realtimeToggle');
    if (!realtimeDropdown || !realtimeToggle) return;

    if (realtimeDropdown.classList.contains('d-none')) {
        realtimeDropdown.classList.remove('d-none');
        realtimeToggle.classList.add('d-none');
    } else {
        realtimeDropdown.classList.add('d-none');
        realtimeToggle.classList.remove('d-none');
    }
}

const originalsData = [ /* ... 기존 데이터 ... */ ];
function renderOriginalsList() { /* ... 기존 함수 ... */ }

const headlinesData = { /* ... 기존 데이터 ... */ };
function renderHeadlines(type) { /* ... 기존 함수 ... */ }
function switchTab(active) { /* ... 기존 함수 ... */ }
function initializeHeadlines() {
    const btnLatest = document.getElementById("btn-latest");
    const btnPopular = document.getElementById("btn-popular");
    const btnRising = document.getElementById("btn-rising");
    const btnFalling = document.getElementById("btn-falling");
    const btnMarket = document.getElementById('btn-view-market');
    const btnCuration = document.getElementById('btn-view-curation');

    if (btnLatest) renderHeadlines("latest");

    if (btnLatest) btnLatest.addEventListener("click", () => { renderHeadlines("latest"); switchTab("latest"); });
    if (btnPopular) btnPopular.addEventListener("click", () => { renderHeadlines("popular"); switchTab("popular"); });
    if (btnRising) btnRising.addEventListener("click", () => { renderHeadlines("rising"); switchTab("rising"); });
    if (btnFalling) btnFalling.addEventListener("click", () => { renderHeadlines("falling"); switchTab("falling"); });

    if (btnMarket && btnCuration) {
        const tabBasic = document.getElementById('tab-basic');
        const tabCuration = document.getElementById('tab-curation');

        btnMarket.addEventListener('click', () => {
            if (btnMarket.classList.contains('active')) return;
            btnMarket.classList.add('active'); btnCuration.classList.remove('active');
            btnMarket.classList.remove('bg-light', 'text-muted'); btnMarket.classList.add('bg-white', 'text-dark');
            btnCuration.classList.remove('bg-white', 'text-dark'); btnCuration.classList.add('bg-light', 'text-muted');
            if (tabBasic) tabBasic.classList.remove('d-none');
            if (tabCuration) tabCuration.classList.add('d-none');
            renderHeadlines('latest'); switchTab('latest');
        });
        btnCuration.addEventListener('click', () => {
            if (btnCuration.classList.contains('active')) return;
            btnCuration.classList.add('active'); btnMarket.classList.remove('active');
            btnCuration.classList.remove('bg-light', 'text-muted'); btnCuration.classList.add('bg-white', 'text-dark');
            btnMarket.classList.remove('bg-white', 'text-dark'); btnMarket.classList.add('bg-light', 'text-muted');
            if (tabCuration) tabCuration.classList.remove('d-none');
            if (tabBasic) tabBasic.classList.add('d-none');
            renderHeadlines('rising'); switchTab('rising');
        });
        if (btnMarket) btnMarket.classList.add('active'); else if (btnCuration) btnCuration.classList.add('active');
    }
}

const filterData = { /* ... 기존 데이터 ... */ };
function renderFilteredList(type) { /* ... 기존 함수 ... */ }
function toggleMainTab(type) { /* ... 기존 함수 ... */ }

const coinInfoData = [ /* ... 기존 데이터 ... */ ];
function renderCoinInfoList() { /* ... 기존 함수 ... */ }

const noticeListData = [ /* ... 기존 데이터 ... */ ];
function renderNoticeList() { /* ... 기존 함수 ... */ }

const exchangeData = [ /* ... 기존 데이터 ... */ ];
const sampleExchangeLogo = "https://placehold.co/100x100/cccccc/000000?text=EX";
function renderExchangeList() {
    const container = document.getElementById('exchange-list');
    if (!container) { console.error('exchange-list 요소를 찾을 수 없습니다.'); return; }
    container.innerHTML = '';
    [...exchangeData, ...exchangeData].forEach(exchange => {
        const a = document.createElement('a');
        a.href = exchange.link; a.target = '_blank'; a.className = 'exchange-item';
        a.innerHTML = `<img src="${sampleExchangeLogo}" alt="${exchange.name}"><span>${exchange.name}</span>`;
        container.appendChild(a);
    });
}

let exchangeScrollInterval;
function initializeExchangeList() {
    const exchangeListEl = document.querySelector('.exchange-list');
    if (!exchangeListEl) return;
    renderExchangeList();

    if (exchangeScrollInterval) clearInterval(exchangeScrollInterval);

    if (exchangeListEl.clientWidth > 0 && exchangeListEl.scrollWidth > exchangeListEl.clientWidth) {
        let scrollAmount = 0;
        const scrollStep = 73;
        const scrollDelay = 2000;
        exchangeScrollInterval = setInterval(() => {
            scrollAmount += scrollStep;
            if (scrollAmount >= exchangeListEl.scrollWidth / 2) {
                exchangeListEl.scrollTo({ left: 0, behavior: 'auto' });
                scrollAmount = scrollStep;
            }
            exchangeListEl.scrollTo({ left: scrollAmount, behavior: 'smooth' });
        }, scrollDelay);
    }
}

const coinData = [ /* ... 기존 데이터 ... */ ];
let currentPage = 1;
const itemsPerPage = 5;
let filteredCoins = [...coinData];

function renderCoinList() {
    const list = document.getElementById('coin-list');
    if (!list) return;
    list.innerHTML = '';
    const start = (currentPage - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    const coinsToShow = filteredCoins.slice(start, end);
    coinsToShow.forEach(coin => {
        const li = document.createElement('li');
        li.className = 'd-flex align-items-center px-2 py-2';
        li.style.cursor = 'pointer';
        li.onclick = () => window.location.href = coin.link;
        li.innerHTML = `
            <div class="d-flex align-items-center flex-grow-1" style="min-width: 0;">
                <img src="https://placehold.co/24x24/cccccc/000000?text=C" alt="coin" style="width: 24px; height: 24px; object-fit: cover; border-radius: 50%; margin-right: 8px;">
                <span class="text-truncate" style="font-size: 14px;">${coin.name}</span>
            </div>
            <div style="width: 100px; text-align: right; font-size: 14px; ${coin.change >= 0 ? 'color:red;' : 'color:blue;'}">
                ${coin.price.toLocaleString()}
            </div>
            <div style="width: 80px; text-align: right; font-size: 14px; ${coin.change >= 0 ? 'color:red;' : 'color:blue;'}">
                ${coin.change > 0 ? '+' : ''}${coin.change}%
            </div>
        `;
        list.appendChild(li);
    });
    updatePagination();
}

function updatePagination() { /* ... 기존 함수 ... */ }

const recentPosts = [ /* ... 기존 데이터 ... */ ];

function initializeWriteButton() {
    const bodyData = document.body.dataset;
    const isLoggedIn = bodyData.isAuthenticated === 'true';
    const writeUrl = bodyData.writeUrl || 'write.html';
    const currentPath = window.location.pathname.split('/').pop() || 'main';

    if (!isLoggedIn || currentPath === 'main' || currentPath === '') {
        return;
    }

    const writeBtn = document.createElement('button');
    writeBtn.id = 'goToWriteBtn';
    writeBtn.className = 'btn btn-primary rounded-circle d-flex align-items-center justify-content-center';
    writeBtn.style.cssText = `
        position: fixed; bottom: 150px; right: 15px; z-index: 1000;
        width: 45px; height: 45px; font-size: 22px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2); display: flex;`;
    writeBtn.innerHTML = '<i class="bi bi-pencil" style="font-size: 17px;"></i>';
    document.body.appendChild(writeBtn);

    writeBtn.addEventListener('click', () => {
        window.location.href = writeUrl;
    });
}
