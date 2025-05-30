document.addEventListener('DOMContentLoaded', () => {
    // --- SHARED VARIABLES & CONFIG ---
    const isAuthenticated = document.body.dataset.isAuthenticated === 'true';
    const csrfToken = document.body.dataset.csrfToken;
    const loginUrl = document.body.dataset.loginUrl || '/account/login/';

    // --- MYPAGE NAVIGATION & DROPDOWN ---
    const dropdownBtn = document.getElementById('dropdownMenuBtn');
    const dropdownMenu = document.querySelector('#mypageDropdown .dropdown-menu');
    const currentLabel = document.getElementById('currentMenuLabel');

    // --- FAVORITE STOCK UNFAVORITE FUNCTIONALITY ---
    const toggleFavoriteUrl = document.body.dataset.toggleFavoriteUrl;
    const unfavoriteButtons = document.querySelectorAll('.unfavorite-btn');
    const favoriteStockActionFeedback = document.getElementById('favoriteStockActionFeedback');
    const confirmUnfavoriteModalElement = document.getElementById('confirmUnfavoriteModal');
    let confirmUnfavoriteModalInstance = null;
    const confirmUnfavoriteBtn = document.getElementById('confirmUnfavoriteBtn');
    const unfavoriteStockNameConfirmSpan = document.getElementById('unfavoriteStockNameConfirm');
    let stockToRemove = null;

    // --- FAVORITE STOCK PREDICTION POPUP ---
    const getPredictionUrl = document.body.dataset.getPredictionUrl;
    const favoriteStockItemsInfo = document.querySelectorAll('.favorite-stock-item-info');
    const predictionPopup = document.getElementById('predictionPopup');
    const predictionPopupBackdrop = document.getElementById('predictionPopupBackdrop');
    const closePredictionPopupBtn = document.getElementById('closePredictionPopup');
    const popupStockNameEl = document.getElementById('popupStockName');
    const popupBaseDateEl = document.getElementById('popupBaseDate');
    const popupPredictionTableBodyEl = document.getElementById('popupPredictionTableBody');
    const popupPredictionChartCanvas = document.getElementById('popupPredictionChart');
    const popupChartTitleEl = document.getElementById('popupChartTitle');
    const popupChartContainer = document.getElementById('popupPredictionChartContainer');
    let popupPredictionChartInstance = null;
    const feedbackMessagePopup = document.getElementById('feedbackMessagePopup'); // For messages inside popup

    // --- SCROLL TO TOP ---
    const scrollTopBtn = document.getElementById('scrollTopBtn');

    // --- INITIALIZATION & CHECKS ---

    // Mypage Login Check (Only if on /mypage path)
    if (!isAuthenticated && window.location.pathname.includes('/mypage')) {
        sessionStorage.setItem('prevPage', window.location.pathname + window.location.search);
        window.location.href = loginUrl;
        return; // Stop further script execution if redirecting
    }

    // Initialize Bootstrap Modal for Unfavorite Confirmation
    if (confirmUnfavoriteModalElement && typeof bootstrap !== 'undefined' && typeof bootstrap.Modal === 'function') {
        try {
            confirmUnfavoriteModalInstance = new bootstrap.Modal(confirmUnfavoriteModalElement);
        } catch (e) {
            console.error('Error initializing Bootstrap Modal for unfavorite:', e);
        }
    } else {
        if (!confirmUnfavoriteModalElement) {
            console.warn('Confirm unfavorite modal element (#confirmUnfavoriteModal) not found.');
        }
        if (typeof bootstrap === 'undefined' || typeof bootstrap.Modal !== 'function') {
            console.warn('Bootstrap Modal component not available for unfavorite confirmation.');
        }
    }

    // --- HELPER FUNCTIONS ---

    function scrollToWithOffset(elementId, offset = 60) {
        const el = document.getElementById(elementId);
        if (!el) {
            console.warn(`Element with ID '${elementId}' not found for scrolling.`);
            return;
        }
        const rect = el.getBoundingClientRect();
        const absoluteY = window.scrollY + rect.top - offset;
        window.scrollTo({ top: absoluteY, behavior: 'smooth' });
    }

    function showFeedback(element, message, type = 'info', duration = 3000) {
        if (!element) {
            // console.warn("Feedback element not found, message:", message);
            return;
        }
        const existingAlert = element.querySelector('.alert');
        if (existingAlert) {
            existingAlert.remove();
        }
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type === 'error' ? 'danger' : 'success'} alert-dismissible fade show small py-2`;
        alertDiv.setAttribute('role', 'alert');
        alertDiv.innerHTML = `${message}
                              <button type="button" class="btn-close btn-sm py-2" data-bs-dismiss="alert" aria-label="Close"></button>`;
        element.appendChild(alertDiv);

        if (duration > 0) {
            setTimeout(() => {
                if (alertDiv && typeof bootstrap !== 'undefined' && bootstrap.Alert) {
                    try {
                        const bsAlert = bootstrap.Alert.getInstance(alertDiv);
                        if (bsAlert) {
                            bsAlert.close();
                        } else {
                            alertDiv.remove(); // Fallback if instance not found
                        }
                    } catch (e) {
                        // console.warn("Error closing bootstrap alert:", e);
                        alertDiv.remove(); // Fallback removal
                    }
                } else if (alertDiv) {
                    alertDiv.remove(); // Fallback if Bootstrap Alert is not available
                }
            }, duration);
        }
    }

    function updateEmptyListMessage(listId, message) {
        const listElement = document.getElementById(listId);
        if (listElement && listElement.children.length === 0) {
            // Check if a message already exists to prevent duplicates
            const existingMessage = listElement.querySelector('.empty-list-message');
            if (existingMessage) existingMessage.remove();

            const messageLi = document.createElement('li');
            messageLi.className = 'list-group-item text-muted border-0 empty-list-message'; // Added class for easier removal/check
            messageLi.textContent = message;
            listElement.appendChild(messageLi);
        }
    }

    // --- DROPDOWN NAVIGATION LOGIC ---
    if (dropdownBtn && dropdownMenu && currentLabel) {
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

                const contentId = item.dataset.contentId; // This might be the main content block ID
                let scrollTargetId = ''; // This will be the specific section ID within a content block

                if (contentId === 'content-mypage') { // Special case for scrolling to top of mypage
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                    return;
                }

                // Determine the target ID for scrolling based on the label or contentId
                // These IDs should correspond to actual elements on the page.
                if (label === "관심종목") { // Assuming "관심종목" section is identified by 'predictionItemsUl' or 'favoriteStockItemsUl_section'
                    scrollTargetId = document.getElementById('predictionItemsUl') ? 'predictionItemsUl' : 'favoriteStockItemsUl_section';
                } else if (label === "내가 쓴 글") {
                    scrollTargetId = 'myPostsList';
                } else if (label === "차단 계정") {
                    scrollTargetId = 'blockuser';
                } else if (contentId) { // Fallback to contentId if specific label logic doesn't match
                    // This handles cases where contentId directly points to the scrollable section
                    // e.g., 'content-favorite-stocks', 'content-my-posts', etc.
                    // The actual scroll target might be the contentId itself or a child element.
                    // For simplicity, we assume contentId is the scroll target here.
                    // Adjust if specific child elements need to be targeted.
                    scrollTargetId = contentId;
                }
                
                if (scrollTargetId && document.getElementById(scrollTargetId)) {
                    scrollToWithOffset(scrollTargetId, 80); // 80px offset for fixed header
                } else if (contentId && document.getElementById(contentId)) {
                    // If specific scrollTargetId wasn't found but the main contentId block exists, scroll to it.
                    scrollToWithOffset(contentId, 80);
                } else {
                    console.warn(`Scroll target not found for label: ${label}, contentId: ${contentId}, determined scrollTargetId: ${scrollTargetId}`);
                }
            });
        });

        // Close dropdown if clicked outside
        document.addEventListener('click', (e) => {
            if (dropdownBtn && (dropdownBtn.contains(e.target) || (dropdownMenu && dropdownMenu.contains(e.target)))) {
                return; // Click was inside the dropdown button or menu
            }
            if (dropdownMenu) {
                dropdownMenu.style.display = 'none';
            }
        });
    }

    // --- UNFAVORITE STOCK FUNCTIONALITY ---
    function proceedUnfavorite(stockCode, stockName, marketName) {
        if (!stockCode || !stockName) {
            console.error('Stock information for unfavorite is incomplete.', { stockCode, stockName, marketName });
            showFeedback(favoriteStockActionFeedback, '삭제할 종목 정보가 올바르지 않습니다.', 'error');
            return;
        }
        if (!toggleFavoriteUrl || !csrfToken) {
            showFeedback(favoriteStockActionFeedback, '페이지 설정 오류. 새로고침 해주세요.', 'error');
            console.error('toggleFavoriteUrl or csrfToken is missing for proceedUnfavorite.');
            return;
        }

        fetch(toggleFavoriteUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify({
                'stock_code': stockCode,
                'stock_name': stockName,
                'market_name': marketName
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => {
                    throw { status: response.status, data: errData, message: errData.message || 'Server error during unfavorite.' };
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.status === 'success' && !data.is_favorite) {
                showFeedback(favoriteStockActionFeedback, `'${stockName}'이(가) 관심 종목에서 삭제되었습니다.`, 'success');
                const listItemsToRemove = document.querySelectorAll(`li[data-stock-code-li="${stockCode}"]`);
                listItemsToRemove.forEach(li => li.remove());

                // Update empty messages for potentially multiple lists
                updateEmptyListMessage('predictionItemsUl', '등록된 관심 종목이 없습니다. 예측 페이지에서 관심 종목을 추가해보세요.');
                updateEmptyListMessage('favoriteStockItemsUl_section', '등록된 관심 종목이 없습니다. 예측 페이지에서 관심 종목을 추가해보세요.');
                // Add other list IDs if necessary
            } else {
                showFeedback(favoriteStockActionFeedback, data.message || '관심 종목 해제 중 오류가 발생했습니다.', 'error');
            }
        })
        .catch(error => {
            console.error("Favorite toggle error (unfavorite):", error);
            let errorMessage = '관심 종목 해제 중 네트워크 오류가 발생했습니다.';
            if (error && error.data && error.data.message) {
                errorMessage = error.data.message;
            } else if (error && error.message && typeof error.message === 'string') {
                errorMessage = error.message;
            }
            showFeedback(favoriteStockActionFeedback, errorMessage, 'error');
        });
    }

    unfavoriteButtons.forEach(button => {
        button.addEventListener('click', function(event) {
            event.stopPropagation();
            const stockCode = this.dataset.stockCode;
            const stockName = this.dataset.stockName;
            const marketName = this.dataset.marketName;

            if (!isAuthenticated) {
                const nextUrl = encodeURIComponent(window.location.pathname + window.location.search);
                showFeedback(favoriteStockActionFeedback, `관심 종목 해제는 <a href="${loginUrl}?next=${nextUrl}" class="alert-link">로그인</a> 후 이용 가능합니다.`, 'error', 5000);
                return;
            }
            if (!csrfToken || !toggleFavoriteUrl) {
                showFeedback(favoriteStockActionFeedback, '오류: 페이지 설정이 완전하지 않습니다. 새로고침 해주세요.', 'error');
                console.error('CSRF token or toggle favorite URL is missing from body data attributes for unfavorite button click.');
                return;
            }

            stockToRemove = { stockCode, stockName, marketName };
            if (unfavoriteStockNameConfirmSpan) {
                unfavoriteStockNameConfirmSpan.textContent = `${stockName} (${stockCode})`;
            }

            if (confirmUnfavoriteModalInstance) {
                confirmUnfavoriteModalInstance.show();
            } else {
                console.warn('Bootstrap Modal for unfavorite not available, falling back to window.confirm.');
                if (window.confirm(`${stockName}(${stockCode})을(를) 관심 종목에서 삭제하시겠습니까?`)) {
                    proceedUnfavorite(stockToRemove.stockCode, stockToRemove.stockName, stockToRemove.marketName);
                } else {
                    stockToRemove = null;
                }
            }
        });
    });

    if (confirmUnfavoriteBtn) {
        confirmUnfavoriteBtn.addEventListener('click', () => {
            if (stockToRemove && confirmUnfavoriteModalInstance) {
                proceedUnfavorite(stockToRemove.stockCode, stockToRemove.stockName, stockToRemove.marketName);
                confirmUnfavoriteModalInstance.hide();
            }
            stockToRemove = null;
        });
    }

    // --- PREDICTION POPUP FUNCTIONALITY ---
    function showPredictionPopup() {
        if (predictionPopup) predictionPopup.style.display = 'block';
        if (predictionPopupBackdrop) predictionPopupBackdrop.style.display = 'block';
        document.body.style.overflow = 'hidden'; // Prevent background scroll
    }

    function hidePredictionPopup() {
        if (predictionPopup) predictionPopup.style.display = 'none';
        if (predictionPopupBackdrop) predictionPopupBackdrop.style.display = 'none';
        document.body.style.overflow = ''; // Restore background scroll

        if (popupPredictionChartInstance) {
            popupPredictionChartInstance.destroy();
            popupPredictionChartInstance = null;
        }
        if (popupPredictionTableBodyEl) popupPredictionTableBodyEl.innerHTML = '';
        if (popupChartContainer) popupChartContainer.style.display = 'none';
        if (feedbackMessagePopup) feedbackMessagePopup.innerHTML = ''; // Clear popup-specific messages
    }

    if (favoriteStockItemsInfo.length > 0 && predictionPopup && getPredictionUrl) {
        favoriteStockItemsInfo.forEach(item => {
            item.addEventListener('click', function () {
                const stockCode = this.dataset.stockCode;
                const stockName = this.dataset.stockName;

                // Reset popup content before loading
                if (popupStockNameEl) popupStockNameEl.textContent = `${stockName} (${stockCode}) 예측 로딩 중...`;
                if (popupBaseDateEl) popupBaseDateEl.textContent = 'N/A';
                if (popupPredictionTableBodyEl) popupPredictionTableBodyEl.innerHTML = '<tr><td colspan="2" class="text-center">데이터를 불러오는 중입니다...</td></tr>';
                if (popupChartContainer) popupChartContainer.style.display = 'none';
                if (feedbackMessagePopup) feedbackMessagePopup.innerHTML = '';
                if (popupPredictionChartInstance) {
                    popupPredictionChartInstance.destroy();
                    popupPredictionChartInstance = null;
                }

                showPredictionPopup();

                const formData = new URLSearchParams();
                formData.append('stock_input', stockCode);
                formData.append('analysis_type', 'technical'); // As per the new script

                fetch(getPredictionUrl, {
                    method: 'POST',
                    headers: {
                        'X-CSRFToken': csrfToken // Ensure csrfToken is available
                    },
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(err => {
                            throw new Error(err.error || `Network response was not ok (${response.status})`);
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.error) {
                        if (popupStockNameEl) popupStockNameEl.textContent = `${stockName} (${stockCode})`;
                        if (popupPredictionTableBodyEl) popupPredictionTableBodyEl.innerHTML = `<tr><td colspan="2" class="text-center text-danger py-3">${data.error}</td></tr>`;
                        if (popupChartContainer) popupChartContainer.style.display = 'none';
                        if (feedbackMessagePopup) feedbackMessagePopup.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                        return;
                    }

                    if (popupStockNameEl) popupStockNameEl.textContent = `${data.stock_name || stockName} (${data.stock_code || stockCode}) 예측`;
                    if (popupBaseDateEl) popupBaseDateEl.textContent = data.prediction_base_date || 'N/A';

                    let tableHtml = '';
                    if (data.predictions && data.predictions.length > 0) {
                        data.predictions.forEach(pred => {
                            tableHtml += `<tr>
                                            <td class="text-center">${pred.date}</td>
                                            <td class="text-end">${pred.price !== null && pred.price !== undefined ? Number(pred.price).toLocaleString() : 'N/A'} 원</td>
                                          </tr>`;
                        });
                    } else {
                        tableHtml = '<tr><td colspan="2" class="text-center">예측 데이터가 없습니다.</td></tr>';
                    }
                    if (popupPredictionTableBodyEl) popupPredictionTableBodyEl.innerHTML = tableHtml;

                    if (data.predictions && data.predictions.length > 0 && popupPredictionChartCanvas && typeof Chart !== 'undefined') {
                        if (popupChartContainer) popupChartContainer.style.display = 'block';
                        if (popupChartTitleEl) popupChartTitleEl.textContent = `${data.stock_name || stockName} - 과거 ${data.past_data ? data.past_data.length : 0}일 및 예측 ${data.predictions.length}일 주가 추이`;

                        if (popupPredictionChartInstance) {
                            popupPredictionChartInstance.destroy();
                        }

                        const pastLabels = data.past_data ? data.past_data.map(d => d.date) : [];
                        const pastPrices = data.past_data ? data.past_data.map(d => d.price) : [];
                        const predictedLabels = data.predictions.map(p => p.date);
                        const predictedPrices = data.predictions.map(p => p.price);

                        let allChartLabels = [];
                        if (pastLabels.length > 0) allChartLabels.push(...pastLabels);
                        // Ensure prediction_base_date is correctly placed if it exists
                        if (data.prediction_base_date && !pastLabels.includes(data.prediction_base_date)) {
                             allChartLabels.push(data.prediction_base_date);
                        }
                        allChartLabels.push(...predictedLabels);
                        allChartLabels = [...new Set(allChartLabels)].sort((a, b) => new Date(a) - new Date(b)); // Sort dates correctly


                        const pastDataForChart = allChartLabels.map(label => {
                            const pastIndex = pastLabels.indexOf(label);
                            if (pastIndex !== -1) return pastPrices[pastIndex];
                            // If the label is the base date and it's not in past data (e.g., it's the connecting point)
                            if (label === data.prediction_base_date) return data.last_actual_close;
                            return null; // Use null for gaps
                        });

                        const predictedDataForChart = allChartLabels.map(label => {
                            // The prediction line should start from the last actual close on the prediction_base_date
                            if (label === data.prediction_base_date) return data.last_actual_close;
                            const predictedIndex = predictedLabels.indexOf(label);
                            if (predictedIndex !== -1) return predictedPrices[predictedIndex];
                            return null; // Use null for gaps
                        });
                        
                        const datasets = [
                            {
                                label: '과거 종가', data: pastDataForChart, borderColor: 'rgba(255, 99, 132, 1)',
                                backgroundColor: 'rgba(255, 99, 132, 0.1)', tension: 0.1, pointRadius: 3, pointHoverRadius: 6,
                                spanGaps: false // Don't connect across nulls in past data unless it's the bridge
                            },
                            {
                                label: '예측 종가', data: predictedDataForChart, borderColor: 'rgba(54, 162, 235, 1)',
                                backgroundColor: 'rgba(54, 162, 235, 0.1)', tension: 0.2, pointRadius: 4, pointHoverRadius: 7,
                                spanGaps: false, // Let Chart.js handle gaps based on nulls. Set to true if you want to connect over nulls.
                                borderDash: [5, 5]
                            }
                        ];

                        popupPredictionChartInstance = new Chart(popupPredictionChartCanvas, {
                            type: 'line',
                            data: { labels: allChartLabels, datasets: datasets },
                            options: {
                                responsive: true, maintainAspectRatio: false,
                                scales: {
                                    y: {
                                        beginAtZero: false,
                                        ticks: { callback: function(value) { return value !== null ? value.toLocaleString() + ' 원' : ''; }, font: {size: 10} }
                                    },
                                    x: {
                                        ticks: { font: {size: 10}, maxRotation: 45, minRotation: 0, autoSkip: true, maxTicksLimit: 15 } // Added autoSkip and maxTicksLimit
                                    }
                                },
                                plugins: {
                                    legend: { display: true, labels: { font: { size: 11 }}},
                                    tooltip: {
                                        enabled: true, mode: 'index', intersect: false, backgroundColor: 'rgba(0,0,0,0.8)',
                                        titleFont: { weight: 'bold', size: 12 }, bodyFont: { size: 11 }, padding: 8,
                                        callbacks: {
                                            label: function(context) {
                                                let label = context.dataset.label || '';
                                                if (label) label += ': ';
                                                if (context.parsed.y !== null) label += context.parsed.y.toLocaleString() + ' 원';
                                                else label += 'N/A'; // Show N/A for null points
                                                return label;
                                            }
                                        }
                                    }
                                },
                                interaction: { // Changed from hover to interaction for Chart.js v3+
                                    mode: 'nearest',
                                    axis: 'x', // Snap to x-axis
                                    intersect: false
                                }
                            }
                        });
                    } else {
                        if (popupChartContainer) popupChartContainer.style.display = 'none';
                        if (!popupPredictionChartCanvas) console.warn("Popup chart canvas not found.");
                        if (typeof Chart === 'undefined') console.warn("Chart.js is not loaded.");
                    }
                })
                .catch(error => {
                    console.error('Error fetching favorite stock prediction:', error);
                    if (popupStockNameEl) popupStockNameEl.textContent = `${stockName} (${stockCode})`;
                    if (popupPredictionTableBodyEl) popupPredictionTableBodyEl.innerHTML = `<tr><td colspan="2" class="text-center text-danger py-3">오류: ${error.message}</td></tr>`;
                    if (popupChartContainer) popupChartContainer.style.display = 'none';
                    if (feedbackMessagePopup) feedbackMessagePopup.innerHTML = `<div class="alert alert-danger">예측 데이터 로드 중 오류 발생: ${error.message}</div>`;
                });
            });
        });

        // Event listeners for closing the prediction popup
        if (closePredictionPopupBtn) closePredictionPopupBtn.addEventListener('click', hidePredictionPopup);
        if (predictionPopupBackdrop) predictionPopupBackdrop.addEventListener('click', hidePredictionPopup);
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape' && predictionPopup && predictionPopup.style.display === 'block') {
                hidePredictionPopup();
            }
        });
    } else {
        if (favoriteStockItemsInfo.length === 0) console.log("No favorite stock items found to attach prediction popup event.");
        if (!predictionPopup) console.warn("Prediction popup element not found.");
        if (!getPredictionUrl) console.warn("Get prediction URL not set in body data.");
    }


    // --- SCROLL TO TOP BUTTON ---
    if (scrollTopBtn) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 80) { // Show button after scrolling 80px
                scrollTopBtn.style.display = 'block';
            } else {
                scrollTopBtn.style.display = 'none';
            }
        });
        scrollTopBtn.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }
});
