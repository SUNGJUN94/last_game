// Wait for the DOM to be fully loaded before executing any script
document.addEventListener('DOMContentLoaded', function() {
    // Attempt to find the div where the chart will be rendered
    const chartDiv = document.getElementById('stock-chart');
    console.log('[Chart.js] DOMContentLoaded.');

    if (!chartDiv) {
        console.error('[Chart.js] #stock-chart div not found. Chart cannot be rendered.');
        const loadingOverlay = document.querySelector('.loading-overlay');
        if (loadingOverlay) loadingOverlay.classList.remove('active');
        return; 
    }
    console.log('[Chart.js] #stock-chart div found.');

    const chartDataElement = document.getElementById('chart-data-json');
    let parsedChartData = {};

    if (chartDataElement && chartDataElement.textContent) {
        try {
            parsedChartData = JSON.parse(chartDataElement.textContent);
            console.log('[Chart.js] Parsed chart data from #chart-data-json.');
        } catch (e) {
            console.error("[Chart.js] 차트 데이터 JSON 파싱 오류:", e);
            console.error("[Chart.js] HTML 내 #chart-data-json 내용:", chartDataElement.textContent);
            chartDiv.innerHTML = '<div class="data-placeholder p-5 text-center">차트 데이터를 파싱하는 중 오류가 발생했습니다.</div>';
            return; 
        }
    } else {
        console.error("[Chart.js] 차트 데이터(#chart-data-json)를 찾을 수 없거나 내용이 없습니다.");
        // chartDiv.innerHTML = '<div class="data-placeholder p-5 text-center">차트 데이터를 찾을 수 없습니다.</div>';
        // Don't stop execution here if chart data is missing, as autocomplete should still work.
        // But Plotly chart rendering below will depend on this.
    }

    const candleDates = parsedChartData.candleDates || [];
    const openPrices = parsedChartData.openPrices || [];
    const highPrices = parsedChartData.highPrices || [];
    const lowPrices = parsedChartData.lowPrices || [];
    const closePrices = parsedChartData.closePrices || [];
    const ma5 = parsedChartData.ma5 || [];
    const ma20 = parsedChartData.ma20 || [];
    const volume = parsedChartData.volume || [];

    window.chartState = {
        showMA5: false,
        showMA20: false,
        showAvgPrice: true
    };

    let avgPrice = 0;
    if (closePrices && closePrices.length > 0) {
        const sum = closePrices.reduce((a, b) => {
            const valA = parseFloat(a);
            const valB = parseFloat(b);
            return (isNaN(valA) ? 0 : valA) + (isNaN(valB) ? 0 : valB);
        }, 0);
        avgPrice = sum / closePrices.length;
    }


    if (candleDates.length > 0 && closePrices.length > 0) {
        console.log('[Chart.js] Attempting to render Plotly chart.');
        if (typeof Plotly === 'undefined') {
            console.error('[Chart.js] Plotly library is not loaded. Chart cannot be rendered.');
            chartDiv.innerHTML = '<div class="data-placeholder p-5 text-center">차트 라이브러리를 불러오는 데 실패했습니다. 페이지를 새로고침해보세요.</div>';
            // return; // Don't return, autocomplete might still be needed.
        } else {
            const priceTrace = {
                x: candleDates,
                y: closePrices,
                type: 'scatter',
                mode: 'lines',
                name: '종가',
                line: { color: '#FF7043', width: 2.5 },
                yaxis: 'y1'
            };

            const ma5Trace = {
                x: candleDates,
                y: ma5,
                name: '5일선',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#29B6F6', width: 1.5 },
                yaxis: 'y1',
                visible: window.chartState.showMA5 ? true : 'legendonly'
            };

            const ma20Trace = {
                x: candleDates,
                y: ma20,
                name: '20일선',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#FFEE58', width: 1.5 },
                yaxis: 'y1',
                visible: window.chartState.showMA20 ? true : 'legendonly'
            };
            
            const avgPriceTrace = {
                x: candleDates.length > 0 ? [candleDates[0], candleDates[candleDates.length - 1]] : [],
                y: candleDates.length > 0 ? [avgPrice, avgPrice] : [],
                name: '평균가격',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#9E9E9E', width: 1, dash: 'dot' },
                yaxis: 'y1',
                visible: window.chartState.showAvgPrice
            };

            const volumeColors = closePrices.map((close, i) => {
                if (i === 0) return '#BDBDBD'; 
                return parseFloat(close) > parseFloat(closePrices[i-1]) ? '#EF5350' : '#42A5F5';
            });

            const volumeTrace = {
                x: candleDates,
                y: volume,
                name: '거래량',
                type: 'bar',
                marker: { color: volumeColors },
                yaxis: 'y2'
            };
            
            const layout = {
                margin: {t: 30, r: 20, b: 50, l: 50},
                xaxis: {
                    rangeslider: {visible: false},
                    color: '#E0E0E0', gridcolor: '#424242', zerolinecolor: '#616161', 
                },
                yaxis: {
                    title: '가격(원)', color: '#E0E0E0', gridcolor: '#424242',
                    zerolinecolor: '#616161', domain: [0.25, 1] 
                },
                yaxis2: {
                    title: '거래량', color: '#E0E0E0', gridcolor: '#333333', 
                    zerolinecolor: '#616161', domain: [0, 0.22], anchor: 'x'
                },
                legend: {
                    orientation: 'h', x: 0.5, xanchor: 'center',
                    y: 1.15, font: {color: '#E0E0E0'}
                },
                height: 500, paper_bgcolor: '#1E1E1E', plot_bgcolor: '#1E1E1E',  
                font: { color: '#E0E0E0' } 
            };

            const config = {
                responsive: true, displayModeBar: true, 
                modeBarButtonsToRemove: ['sendDataToCloud'], 
                modeBarButtonsToAdd: [ 
                    'drawline', 'drawopenpath', 'drawclosedpath',
                    'drawcircle', 'drawrect', 'eraseshape'
                ]
            };

            Plotly.newPlot(chartDiv, [priceTrace, ma5Trace, ma20Trace, avgPriceTrace, volumeTrace], layout, config);
            window.stockChart = chartDiv; 
            console.log('[Chart.js] Plotly chart rendered.');

            const ma5Button = document.getElementById('ma5Btn');
            if (ma5Button) ma5Button.classList.toggle('active', window.chartState.showMA5);
            
            const ma20Button = document.getElementById('ma20Btn');
            if (ma20Button) ma20Button.classList.toggle('active', window.chartState.showMA20);

            const avgPriceButton = document.getElementById('avgPriceBtn');
            if (avgPriceButton) avgPriceButton.classList.toggle('active', window.chartState.showAvgPrice);
        }
    } else {
        console.warn('[Chart.js] 차트 데이터가 없습니다. (candleDates 또는 closePrices 비어 있음). Plotly 차트 렌더링을 건너<0xEB><07><0x84>니다.');
        if(chartDiv) chartDiv.innerHTML = '<div class="data-placeholder p-5 text-center">차트 데이터를 불러올 수 없습니다. 종목을 검색하거나 기간을 변경해보세요. (JS Placeholder)</div>';
    }

    // --- Autocomplete Logic ---
    const stockQueryInputChart = document.getElementById('stockQueryInputChart');
    const autocompleteResultsDivChart = document.getElementById('autocompleteResultsChart');
    let autocompleteRequestTimeoutChart;
    let isAutocompleteItemSelectedChart = false; // To prevent re-triggering fetch after selection

    console.log('[Chart.js Autocomplete] Initializing...');
    if (stockQueryInputChart && autocompleteResultsDivChart) {
        console.log('[Chart.js Autocomplete] Input field and results div found.');
        stockQueryInputChart.addEventListener('input', function() {
            const query = this.value.trim();
            console.log(`[Chart.js Autocomplete] Input event. Query: "${query}"`);

            if (isAutocompleteItemSelectedChart) {
                console.log('[Chart.js Autocomplete] Item was selected, resetting flag.');
                isAutocompleteItemSelectedChart = false; 
                // Do not return; allow further typing if user wants to modify selection
            }
            
            clearTimeout(autocompleteRequestTimeoutChart);

            if (query.length < 1) {
                console.log('[Chart.js Autocomplete] Query too short, hiding results.');
                autocompleteResultsDivChart.innerHTML = '';
                autocompleteResultsDivChart.style.display = 'none';
                return;
            }

            console.log('[Chart.js Autocomplete] Setting timeout for fetch...');
            autocompleteRequestTimeoutChart = setTimeout(() => {
                console.log(`[Chart.js Autocomplete] Fetching for query: "${query}"`);
                // Ensure the URL is correct and matches your urls.py
                fetch(`/chart/api/search_stocks_chart/?term=${encodeURIComponent(query)}&limit=7`) 
                    .then(response => {
                        console.log('[Chart.js Autocomplete] Response status:', response.status);
                        if (!response.ok) {
                            // Log the response text for more details on server error
                            return response.text().then(text => {
                                console.error('[Chart.js Autocomplete] Network error response text:', text);
                                throw new Error('Network response was not ok: ' + response.statusText + ' - ' + text);
                            });
                        }
                        return response.json();
                    })
                    .then(data => {
                        console.log('[Chart.js Autocomplete] Data received:', JSON.stringify(data));
                        autocompleteResultsDivChart.innerHTML = ''; // Clear previous results

                        if (data.error) {
                            console.warn('[Chart.js Autocomplete] Server returned error:', data.error);
                            autocompleteResultsDivChart.innerHTML = `<div class="list-group-item text-danger small p-2">${data.error}</div>`;
                        } else if (Array.isArray(data) && data.length > 0) {
                            console.log(`[Chart.js Autocomplete] ${data.length} items received. Rendering...`);
                            data.forEach(item => {
                                const div = document.createElement('a');
                                div.href = '#'; 
                                div.classList.add('list-group-item', 'list-group-item-action', 'd-flex', 'justify-content-between', 'align-items-center');
                                // Displaying name, code, and market similar to predict_info.html
                                div.innerHTML = `<span class="stock-name">${item.value || item.name}</span> <small class="stock-code-market">${item.code} | ${item.market}</small>`;
                                // console.log('[Chart.js Autocomplete] Creating item:', item.value, item.code, item.market);
                                
                                div.addEventListener('click', function(e) {
                                    e.preventDefault();
                                    e.stopPropagation();
                                    console.log('[Chart.js Autocomplete] Item clicked:', item.value);
                                    stockQueryInputChart.value = item.value; // Set input to stock name
                                    autocompleteResultsDivChart.innerHTML = '';
                                    autocompleteResultsDivChart.style.display = 'none';
                                    isAutocompleteItemSelectedChart = true; // Set flag
                                    
                                    // Submit the form to reload the page with the selected stock
                                    const form = stockQueryInputChart.closest('form');
                                    if (form) {
                                        console.log('[Chart.js Autocomplete] Submitting form.');
                                        showLoading(); // Show loading indicator
                                        form.submit();
                                    } else {
                                        console.warn('[Chart.js Autocomplete] Form not found for submission.');
                                    }
                                });
                                autocompleteResultsDivChart.appendChild(div);
                            });
                        } else {
                            console.log('[Chart.js Autocomplete] No items or empty array received.');
                            autocompleteResultsDivChart.innerHTML = '<div class="list-group-item text-muted small p-2">검색 결과가 없습니다.</div>';
                        }
                        autocompleteResultsDivChart.style.display = 'block';
                        console.log('[Chart.js Autocomplete] Results div display set to "block". Rect:', autocompleteResultsDivChart.getBoundingClientRect());
                        console.log('[Chart.js Autocomplete] Computed display style:', window.getComputedStyle(autocompleteResultsDivChart).display);

                    })
                    .catch(error => {
                        console.error('[Chart.js Autocomplete] Fetch/Parse error:', error.message, error);
                        autocompleteResultsDivChart.innerHTML = `<div class="list-group-item text-danger small p-2">자동완성 오류: ${error.message}</div>`;
                        autocompleteResultsDivChart.style.display = 'block';
                    });
            }, 300); // Debounce time
        });

        // Hide results when clicking outside
        document.addEventListener('click', function(event) {
            const isClickInsideInput = stockQueryInputChart.contains(event.target);
            const isClickInsideResults = autocompleteResultsDivChart.contains(event.target);
            if (!isClickInsideInput && !isClickInsideResults) {
                // console.log('[Chart.js Autocomplete] Click outside, hiding results.');
                autocompleteResultsDivChart.style.display = 'none';
            }
        });
         // Optional: Hide on blur if not hovering over results
        stockQueryInputChart.addEventListener('blur', function() {
            // console.log('[Chart.js Autocomplete] Input blurred.');
            setTimeout(() => { // Timeout to allow click event on results to fire
                if (!autocompleteResultsDivChart.matches(':hover')) {
                    // console.log('[Chart.js Autocomplete] Hiding results on blur (not hovering).');
                    autocompleteResultsDivChart.style.display = 'none';
                }
            }, 150);
        });

    } else {
        console.warn('[Chart.js Autocomplete] stockQueryInputChart or autocompleteResultsDivChart not found. Autocomplete disabled.');
    }
    // --- End Autocomplete Logic ---


    // Event listeners for table rows and market cap items (to submit form)
    const topStockTables = document.querySelectorAll('.top-stocks-table');
    topStockTables.forEach(table => {
        table.addEventListener('click', function(e) {
            const row = e.target.closest('tr');
            if (row && row.parentElement.tagName === 'TBODY') { 
                const stockNameCell = row.querySelector('td:first-child');
                if (stockNameCell) {
                    const stockName = stockNameCell.textContent.split('(')[0].trim();
                    if (stockQueryInputChart) {
                        stockQueryInputChart.value = stockName;
                        const form = stockQueryInputChart.closest('form');
                        if (form) {
                            showLoading(); 
                            form.submit();
                        }
                    }
                }
            }
        });
    });

    const marketCapSlider = document.querySelector('.market-cap-slider');
    if (marketCapSlider) {
        marketCapSlider.addEventListener('click', function(e) {
            const item = e.target.closest('.market-cap-item');
            if (item) {
                const stockName = item.querySelector('.stock-name')?.textContent.trim();
                if (stockName && stockQueryInputChart) {
                    stockQueryInputChart.value = stockName;
                    const form = stockQueryInputChart.closest('form');
                    if (form) {
                        showLoading(); 
                        form.submit();
                    }
                }
            }
        });
    }
    hideLoading(); // Hide loading on initial load if it was somehow active

    // Slider functionality (remains unchanged for now)
    let currentSlide = 0;
    const slideInterval = 3000; 
    let slideTimer;

    function getItemsPerView() {
        if (window.innerWidth > 992) return 4; 
        if (window.innerWidth > 768) return 3; 
        if (window.innerWidth > 576) return 2; 
        return 1; 
    }

    function updateSliderLayout() {
        const slider = document.querySelector('.market-cap-slider');
        const items = document.querySelectorAll('.market-cap-item');
        if (!slider || items.length === 0) return;

        const itemsPerView = getItemsPerView();
        const itemWidthPercentage = 100 / itemsPerView;
        
        items.forEach(item => {
            item.style.flex = `0 0 calc(${itemWidthPercentage}% - 1rem)`; 
        });
        updateSliderPosition(); 
    }

    function updateSliderPosition() {
        const slider = document.querySelector('.market-cap-slider');
        const items = document.querySelectorAll('.market-cap-item');
        if (!slider || items.length === 0) return;

        const itemsPerView = getItemsPerView();
        const totalItems = items.length;
        const maxSlide = Math.max(0, totalItems - itemsPerView);

        if (currentSlide > maxSlide) currentSlide = 0; 
        if (currentSlide < 0) currentSlide = maxSlide; 

        const itemWidth = (100 / itemsPerView); 
        const offsetPercentage = currentSlide * -itemWidth; 
        
        if (totalItems <= itemsPerView) {
             slider.style.transform = `translateX(0%)`;
        } else {
             slider.style.transform = `translateX(${offsetPercentage}%)`;
        }
    }

    function nextSlide() {
        const items = document.querySelectorAll('.market-cap-item');
        if (items.length === 0) return;
        const itemsPerView = getItemsPerView();
        const maxSlide = Math.max(0, items.length - itemsPerView); 

        currentSlide++;
        if (currentSlide > maxSlide) currentSlide = 0; 
        updateSliderPosition();
    }

    function prevSlide() { 
        const items = document.querySelectorAll('.market-cap-item');
        if (items.length === 0) return;
        const itemsPerView = getItemsPerView();
        const maxSlide = Math.max(0, items.length - itemsPerView);

        currentSlide--;
        if (currentSlide < 0) currentSlide = maxSlide; 
        updateSliderPosition();
    }

    function startAutoSlide() {
        stopAutoSlide(); 
        const items = document.querySelectorAll('.market-cap-item');
        if (items.length > getItemsPerView()){ 
           slideTimer = setInterval(nextSlide, slideInterval);
        }
    }

    function stopAutoSlide() {
        clearInterval(slideTimer);
    }

    updateSliderLayout(); 
    startAutoSlide();     

    const sliderContainer = document.querySelector('.market-cap-slider-container');
    if (sliderContainer) {
        sliderContainer.addEventListener('mouseenter', stopAutoSlide);
        sliderContainer.addEventListener('mouseleave', startAutoSlide);
    }
    
    let resizeTimer;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function() {
            updateSliderLayout();
            startAutoSlide();
        }, 250); 
    });

}); // End DOMContentLoaded for chart rendering and main interactions


// Global functions for chart controls (MA lines, Avg Price)
function toggleMA(lineIndex, buttonId) {
    const btn = document.getElementById(buttonId);
    if (!window.stockChart || !Plotly || !window.stockChart.data || !window.stockChart.data[lineIndex]) { 
        console.warn('toggleMA: Chart or Plotly not ready or trace data missing for index:', lineIndex);
        return;
    }

    const currentTraceVisibility = window.stockChart.data[lineIndex].visible;
    const newVisibility = (currentTraceVisibility === true) ? 'legendonly' : true; 
    
    Plotly.restyle(window.stockChart, 'visible', newVisibility, [lineIndex]);
    if(btn) btn.classList.toggle('active', newVisibility === true);

    if (window.chartState) {
        if (buttonId === 'ma5Btn') window.chartState.showMA5 = (newVisibility === true);
        else if (buttonId === 'ma20Btn') window.chartState.showMA20 = (newVisibility === true);
        else if (buttonId === 'avgPriceBtn') window.chartState.showAvgPrice = (newVisibility === true);
    }
}

function toggleMA5() { toggleMA(1, 'ma5Btn'); }
function toggleMA20() { toggleMA(2, 'ma20Btn'); }
function toggleAvgPrice() { toggleMA(3, 'avgPriceBtn'); }

// Global utility functions
function showLoading() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) overlay.classList.add('active');
}

function hideLoading() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) overlay.classList.remove('active');
}

function saveChart() {
    const chartElement = document.getElementById('stock-chart'); 
    if (chartElement && typeof Plotly !== 'undefined' && window.stockChart) { // Check window.stockChart
        Plotly.toImage(window.stockChart, { // Use window.stockChart which is the Plotly graph div
            format: 'png', width: 1200, height: 800,
        }).then(function(dataUrl) {
            Plotly.downloadImage(window.stockChart, {
                format: 'png', width: 1200, height: 800,
                filename: `stock-chart-${new Date().toISOString().slice(0,10)}`
            });
        }).catch(function(err) {
            console.error("차트 이미지 변환 실패:", err);
            alert("차트 저장에 실패했습니다.");
        });
    } else {
        alert("차트 요소를 찾을 수 없거나 Plotly 라이브러리가 로드되지 않았습니다.");
    }
}

function shareChart() {
    const currentUrl = window.location.href;
    if (navigator.share) {
        navigator.share({
            title: document.title || '주식 차트 공유',
            text: `현재 보고 있는 주식 차트를 확인해보세요: ${document.title || ''}`,
            url: currentUrl
        }).catch(err => {
            console.warn("공유 API 사용 중 오류:", err);
            copyUrlToClipboard(currentUrl);
        });
    } else {
        copyUrlToClipboard(currentUrl);
    }
}

function copyUrlToClipboard(url) {
    navigator.clipboard.writeText(url).then(() => {
        alert('차트 URL이 클립보드에 복사되었습니다.');
    }).catch(err => {
        console.error('클립보드 복사 실패:', err);
        try {
            const textArea = document.createElement("textarea");
            textArea.value = url;
            textArea.style.position = "fixed"; textArea.style.opacity = "0";
            document.body.appendChild(textArea);
            textArea.focus(); textArea.select();
            const successful = document.execCommand('copy');
            if (successful) {
                alert('차트 URL이 클립보드에 복사되었습니다. (대체 방식)');
            } else {
                alert('URL 복사에 실패했습니다. 직접 복사해주세요.');
            }
            document.body.removeChild(textArea);
        } catch (e) {
            alert('URL 복사에 실패했습니다. 직접 복사해주세요.');
            console.error('대체 클립보드 복사 실패:', e);
        }
    });
}
