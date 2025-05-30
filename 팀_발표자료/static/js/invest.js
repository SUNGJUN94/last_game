const dummyCoins = [
  { id: 1, name: "비트코인", code: "BTC", priceKRW: 137327539, priceUSD: 96223, rate: "+1.44%", starred: false },
  { id: 2, name: "이더리움", code: "ETH", priceKRW: 2637543, priceUSD: 1848.1, rate: "+2.89%", starred: false },
  { id: 3, name: "테더", code: "USDT", priceKRW: 1427, priceUSD: 1, rate: "-0%", starred: false },
  { id: 4, name: "리플", code: "XRP", priceKRW: 3197, priceUSD: 2.24, rate: "+0.74%", starred: false },
  { id: 5, name: "바이낸스 코인", code: "BNB", priceKRW: 861132, priceUSD: 555.1, rate: "+0.36%", starred: false },
  { id: 6, name: "솔라나", code: "SOL", priceKRW: 216303, priceUSD: 130.1, rate: "+3.4%", starred: false },
  { id: 7, name: "유에스디 코인", code: "USDC", priceKRW: 1427, priceUSD: 1, rate: "+0.02%", starred: false },
  { id: 8, name: "도지코인", code: "DOGE", priceKRW: 255.7, priceUSD: 0.19, rate: "+3.01%", starred: false },
  { id: 9, name: "에이다", code: "ADA", priceKRW: 1008, priceUSD: 0.67, rate: "+1.71%", starred: false },
  { id: 10, name: "트론", code: "TRX", priceKRW: 355.2, priceUSD: 0.12, rate: "+1.4%", starred: false },
  { id: 11, name: "리도 스테이크 이더", code: "STETH", priceKRW: 2635716, priceUSD: 1848.5, rate: "+3.16%", starred: false }
];

let currentCurrency = "KRW";
let showFavoritesOnly = false;
let currentTabType = 'all';

function renderCoins() {
  const list = document.getElementById('coin-list');
  const keyword = document.getElementById('search-input').value.trim().toLowerCase();
  list.innerHTML = '';

  let filtered = [...dummyCoins];

  if (currentTabType === 'up') {
      filtered.sort((a, b) => parseRate(b.rate) - parseRate(a.rate));
  } else if (currentTabType === 'down') {
      filtered.sort((a, b) => parseRate(a.rate) - parseRate(b.rate));
  } else if (currentTabType === 'hot') {
      filtered.sort((a, b) => b.name.length - a.name.length);
  }

  filtered.forEach((coin, idx) => {
      if (showFavoritesOnly && !coin.starred) return;
      if (keyword && !coin.name.toLowerCase().includes(keyword) && !coin.code.toLowerCase().includes(keyword)) return;

      const price = currentCurrency === 'KRW'
          ? `${coin.priceKRW.toLocaleString()}원`
          : `$${coin.priceUSD.toLocaleString()}`;
      const rateColor = coin.rate.startsWith('+') ? 'text-success' : 'text-danger';
      const starClass = coin.starred ? 'star active' : 'star';

      list.innerHTML += `
          <div class="d-flex justify-content-between align-items-center coin-item" onclick="goToCoinDetail('${coin.code}')">
              <div class="d-flex align-items-center gap-2">
                  <div class="rank">${idx + 1}</div>
                  <img src="{% static 'img/icon_${coin.code}.png' %}" alt="${coin.code}" />
                  <div>
                      <div class="coin-name">${coin.name} <span class="coin-code">${coin.code}</span></div>
                      <div class="price">
                          ${price}
                          <span class="rate ${rateColor}">${coin.rate}</span>
                      </div>
                  </div>
              </div>
              <i class="bi bi-star-fill ${starClass}" onclick="event.stopPropagation(); toggleStar(${coin.id})"></i>
          </div>
      `;
  });
}

function parseRate(rateStr) {
  return parseFloat(rateStr.replace('%', ''));
}

function goToCoinDetail(code) {
  window.location.href = '#'; // TODO: 상세 화면 URL
}

function toggleStar(id) {
  const coin = dummyCoins.find(c => c.id === id);
  if (coin) {
      coin.starred = !coin.starred;
      renderCoins();
  }
}

document.querySelectorAll('.filter-tab').forEach(btn => {
  btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      currentTabType = btn.getAttribute('data-type');

      document.querySelectorAll('.invest-filter-description .desc').forEach(desc => {
          desc.classList.toggle('active', desc.getAttribute('data-type') === currentTabType);
      });

      renderCoins();
  });
});

document.getElementById('btn-all').addEventListener('click', () => {
  showFavoritesOnly = false;
  document.getElementById('btn-all').classList.add('active');
  document.getElementById('btn-favorite').classList.remove('active');
  renderCoins();
});

document.getElementById('btn-favorite').addEventListener('click', () => {
  showFavoritesOnly = true;
  document.getElementById('btn-all').classList.remove('active');
  document.getElementById('btn-favorite').classList.add('active');
  renderCoins();
});

document.getElementById('currency-select').addEventListener('change', (e) => {
  currentCurrency = e.target.value;
  renderCoins();
});

document.getElementById('search-input').addEventListener('input', () => {
  renderCoins();
});

// 초기 렌더링
renderCoins();