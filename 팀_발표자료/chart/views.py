import pandas as pd
from django.shortcuts import render
import json
from datetime import timedelta, datetime, date as date_type
from django.http import JsonResponse
from django.core.cache import cache # Django 캐시 사용
import traceback
from django.db.models import Max, Min
from django.db import close_old_connections # 명시적인 DB 커넥션 관리를 위해 추가

# predict_info 앱의 모델 및 유틸리티를 가져옵니다.
try:
    from predict_info.models import StockPrice, MarketIndex
    # from predict_info.utils import get_krx_stock_list # chart 앱에서는 자체 캐시 함수 사용
except ImportError:
    print("[WARNING] chart/views.py: predict_info.models 또는 utils를 import할 수 없습니다.")
    StockPrice = None
    MarketIndex = None

# 문자열 정규화 및 클리닝을 위한 unicodedata, re 임포트
import unicodedata
import re
import yfinance as yf # yfinance 라이브러리 import

# 캐시 타임아웃 설정 (초 단위)
CACHE_TTL_STOCK_LIST = 60 * 60 * 24  # 종목 리스트: 24시간
CACHE_TTL_STOCK_INFO = 60 * 5        # 개별 종목 현재 정보: 5분
CACHE_TTL_STOCK_DATA = 60 * 30       # 개별 종목 기간별 데이터: 30분
CACHE_TTL_MARKET_SUMMARY = 60 * 10   # 시장 요약: 10분 (yfinance 호출 포함)
CACHE_TTL_RANKINGS = 60 * 60         # 각종 순위 정보: 1시간
CACHE_TTL_52WK = 60 * 60 * 3         # 52주 최고/최저: 3시간


# predict_info.views 에 있는 함수와 동일하게 정의 (코드 중복을 피하려면 공통 utils로 이동하는 것이 좋음)
def normalize_stock_name(name):
    """문자열 정규화 (NFC), 소문자 변환, 양쪽 공백 제거, 괄호 및 내용 제거, 특수문자 일부 제거"""
    if not isinstance(name, str):
        return ""
    name_normalized = unicodedata.normalize('NFC', name)
    name_lower = name_normalized.lower()
    # 괄호와 그 안의 내용 제거 (예: (주), (Reg.S), (舊 ...))
    name_no_parentheses = re.sub(r'\([^)]*\)', '', name_lower)
    # 일부 특수 문자(알파벳, 숫자, 한글, 공백 제외) 제거.
    name_cleaned = re.sub(r'[^a-z0-9가-힣\s]', '', name_no_parentheses) # 영어소문자, 숫자, 한글, 공백만 허용
    name_stripped = name_cleaned.strip()
    return name_stripped

def get_krx_stock_list_for_chart_cached():
    cache_key = 'db_stock_list_chart_app_v3'
    cached_list = cache.get(cache_key)
    if cached_list is not None:
        return cached_list

    if not StockPrice:
        print("[ERROR][chart_app] StockPrice model is not available for get_krx_stock_list_for_chart_cached.")
        cache.set(cache_key, [], timeout=60)
        return []
    
    close_old_connections() # DB 조회 전 연결 확인/정리
    try:
        # OperationalError (2013, 'Lost connection to MySQL server during query') 발생 시
        # 이 쿼리가 너무 오래 실행되는지 확인 필요. StockPrice 테이블에 적절한 인덱싱이 되어 있는지 확인.
        # (stock_code, stock_name, market_name 컬럼 등)
        print("[INFO][chart_app] Fetching unique stock list from StockPrice DB for autocomplete cache...")
        stocks_qs = StockPrice.objects.values('stock_code', 'stock_name', 'market_name').distinct().order_by('stock_name')
        stock_list_of_dicts = [
            {'name': str(s['stock_name']).strip(), 'code': str(s['stock_code']).strip(), 'market': str(s.get('market_name', '')).strip().upper()}
            for s in stocks_qs
        ]
        cache.set(cache_key, stock_list_of_dicts, timeout=CACHE_TTL_STOCK_LIST)
        print(f"[INFO][chart_app] Unique stock list from DB fetched and cached. Total: {len(stock_list_of_dicts)} stocks.")
        return stock_list_of_dicts
    except Exception as e:
        print(f"[ERROR][chart_app] Error fetching or caching unique stock list from DB: {e}")
        traceback.print_exc()
        cache.set(cache_key, [], timeout=60)
        return []


def get_stock_info_from_db(stock_code_query):
    if not StockPrice or not stock_code_query:
        return None
    
    cache_key = f'stock_info_db_{stock_code_query}'
    cached_info = cache.get(cache_key)
    if cached_info is not None:
        return cached_info

    close_old_connections() # DB 조회 전 연결 확인/정리
    try:
        # 이 쿼리가 오래 실행될 가능성은 낮지만, stock_code에 인덱스가 있는지 확인.
        latest_stock_data = StockPrice.objects.filter(stock_code=stock_code_query).order_by('-date').first()
        if latest_stock_data:
            change_percent_val = latest_stock_data.change * 100 if latest_stock_data.change is not None else None
            stock_info = {
                'current_price': latest_stock_data.close_price,
                'previous_close': latest_stock_data.previous_day_close_price,
                'change': latest_stock_data.change_value,
                'change_percent': change_percent_val,
                'name': latest_stock_data.stock_name,
                'code': latest_stock_data.stock_code,
                'market_cap': latest_stock_data.market_cap,
                'market': latest_stock_data.market_name,
                'date': latest_stock_data.date.strftime('%Y-%m-%d')
            }
            cache.set(cache_key, stock_info, timeout=CACHE_TTL_STOCK_INFO)
            return stock_info
    except Exception as e:
        print(f"[ERROR][chart_app] Error fetching latest stock info from DB for {stock_code_query}: {e}")
        traceback.print_exc()
    return None


def get_stock_price_data_from_db(stock_code_query, period_str):
    if not StockPrice or not stock_code_query:
        return pd.DataFrame(), stock_code_query

    cache_key = f'stock_price_data_db_{stock_code_query}_{period_str}'
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data['df'], cached_data['stock_name']

    end_date = datetime.now().date()
    if period_str == '1m': start_date = (pd.to_datetime(end_date) - pd.DateOffset(months=1)).date()
    elif period_str == '3m': start_date = (pd.to_datetime(end_date) - pd.DateOffset(months=3)).date()
    elif period_str == '6m': start_date = (pd.to_datetime(end_date) - pd.DateOffset(months=6)).date()
    elif period_str == '1y': start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=1)).date()
    elif period_str == '3y': start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=3)).date()
    elif period_str == '5y': start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=5)).date()
    elif period_str == 'all': start_date = datetime(1990, 1, 1).date()
    else: start_date = (pd.to_datetime(end_date) - pd.DateOffset(months=6)).date()

    close_old_connections() # DB 조회 전 연결 확인/정리
    try:
        # period_str == 'all' 일 경우 매우 많은 데이터를 조회할 수 있음.
        # (stock_code, date) 복합 인덱스 등이 성능에 도움될 수 있음.
        stock_data_qs = StockPrice.objects.filter(
            stock_code=stock_code_query,
            date__gte=start_date,
            date__lte=end_date
        ).order_by('date').values(
            'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume',
            'MA5', 'MA20'
        )
        
        stock_name_info = StockPrice.objects.filter(stock_code=stock_code_query).values('stock_name').first()
        stock_name_display = stock_name_info['stock_name'] if stock_name_info else stock_code_query

        if not stock_data_qs.exists():
            data_to_cache = {'df': pd.DataFrame(), 'stock_name': stock_name_display}
            cache.set(cache_key, data_to_cache, timeout=CACHE_TTL_STOCK_DATA)
            return pd.DataFrame(), stock_name_display

        df = pd.DataFrame(list(stock_data_qs))
        df = df.rename(columns={
            'date': 'datetime', 'open_price': 'open', 'high_price': 'high',
            'low_price': 'low', 'close_price': 'close', 'volume': 'volume',
            'MA5': 'ma5', 'MA20': 'ma20'
        })
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        data_to_cache = {'df': df, 'stock_name': stock_name_display}
        cache.set(cache_key, data_to_cache, timeout=CACHE_TTL_STOCK_DATA)
        return df, stock_name_display
    except Exception as e:
        print(f"[ERROR][chart_app] Error fetching historical stock data from DB for {stock_code_query}: {e}")
        # 오류 발생 시 stock_name_info를 다시 조회하는 것은 동일한 DB 연결 오류를 유발할 수 있으므로,
        # 이미 조회 시도했던 stock_code_query를 사용하거나, 기본값을 사용합니다.
        stock_name_display_err = stock_code_query # 또는 이전에 성공적으로 가져온 이름이 있다면 그것을 사용
        # 아래 코드는 오류 상황에서 다시 DB를 조회하므로 주석 처리하거나 수정 필요
        # stock_info_err = StockPrice.objects.filter(stock_code=stock_code_query).values('stock_name').first()
        # stock_name_display_err = stock_info_err['stock_name'] if stock_info_err else stock_code_query
        cache.set(cache_key, {'df': pd.DataFrame(), 'stock_name': stock_name_display_err}, timeout=60)
        return pd.DataFrame(), stock_name_display_err


def chart_view(request):
    # ==================================================================================
    # MySQL OperationalError (2013, 'Lost connection') 및 (2002, 'Cant connect to socket') 해결 가이드
    # ==================================================================================
    # 이 오류들은 코드 수정만으로 해결되기 어려우며, DB 서버 및 Django 설정 확인이 필수적입니다.
    #
    # 1. MySQL 서버 상태 확인:
    #    - MySQL 서버가 정상적으로 실행 중인지 확인합니다. (예: `sudo systemctl status mysql` 또는 `sudo service mysql status`)
    #    - MySQL 오류 로그를 확인하여 서버 자체의 문제가 있는지 점검합니다.
    #
    # 2. Django `settings.py`의 `DATABASES` 설정 확인:
    #    - `ENGINE`: `django.db.backends.mysql` 로 올바르게 설정되어 있는지 확인합니다.
    #    - `NAME`, `USER`, `PASSWORD`: 데이터베이스 이름, 사용자, 비밀번호가 정확한지 확인합니다.
    #    - `HOST`:
    #        - MySQL 서버가 Django 애플리케이션과 동일한 서버에 있다면 `localhost` 또는 비워둘 수 있습니다 (소켓 연결 시도).
    #        - 다른 서버에 있다면 해당 서버의 IP 주소 또는 호스트명을 입력합니다 (TCP/IP 연결).
    #        - **(2002) 오류 발생 시 중요**: `HOST`가 `localhost`이거나 비어있으면 MySQL 클라이언트는 기본적으로 유닉스 소켓을 사용하려 합니다.
    #          만약 MySQL 서버가 TCP/IP로만 리스닝하거나 소켓 경로가 다르면 이 오류가 발생합니다.
    #          이 경우 `HOST`를 `'127.0.0.1'`로 명시하여 TCP/IP 연결을 강제하거나, `OPTIONS`에서 정확한 소켓 경로를 지정해야 합니다.
    #    - `PORT`: MySQL 서버가 기본 포트(3306)가 아닌 다른 포트를 사용한다면 명시해야 합니다. `HOST`가 설정된 경우에만 유효합니다.
    #    - `OPTIONS`:
    #        - 유닉스 소켓 경로를 직접 지정해야 하는 경우:
    #          `'OPTIONS': {'unix_socket': '/path/to/your/mysqld.sock'}` (실제 소켓 경로로 변경)
    #        - 기타 연결 옵션 (예: `init_command`, `charset` 등)
    #    - `CONN_MAX_AGE`:
    #        - `0` (기본값): 각 요청마다 새로운 DB 연결을 맺고 요청 후 닫습니다. `wait_timeout` 문제 발생 가능성을 줄입니다.
    #        - 양수: 해당 시간(초)만큼 연결을 재사용합니다. 이 값이 MySQL의 `wait_timeout`보다 길면 (2013) 오류의 원인이 됩니다.
    #          `CONN_MAX_AGE`를 `wait_timeout`보다 짧게 설정하거나 0으로 설정하는 것을 고려합니다.
    #    - `CONN_HEALTH_CHECKS: True` (Django 4.1+ 권장, Django 5.0+ 기본값 True):
    #        - `CONN_MAX_AGE`가 0보다 클 때, 재사용 전에 연결 상태를 확인하여 끊어진 연결 사용을 방지합니다. (2013) 오류 완화에 도움.
    #
    # 3. MySQL 서버 설정 (`my.cnf` 또는 `my.ini`) 확인:
    #    - `wait_timeout`: 유휴 연결 유지 시간입니다. Django의 `CONN_MAX_AGE`와 연관됩니다. 너무 짧으면 (2013) 오류가 자주 발생할 수 있습니다.
    #    - `max_connections`: 동시에 허용되는 최대 연결 수입니다. 부하가 높은 경우 이 한계에 도달할 수 있습니다.
    #    - `max_allowed_packet`: 매우 큰 쿼리나 데이터를 전송할 때 이 설정이 작으면 연결이 끊길 수 있습니다.
    #    - `bind-address`: MySQL 서버가 어떤 IP 주소에서 연결을 허용할지 설정합니다. `127.0.0.1`이면 로컬에서만, `0.0.0.0`이면 모든 IP에서 허용합니다.
    #    - `socket`: MySQL 서버가 사용하는 유닉스 소켓 파일 경로입니다. Django 설정과 일치해야 합니다.
    #
    # 4. 데이터베이스 인덱싱:
    #    - `StockPrice` 테이블: `stock_code`, `date` 컬럼에 대한 단일 및 복합 인덱스. `market_name`, `market_cap`, `change` 등 필터링/정렬에 사용되는 컬럼에도 인덱스 고려.
    #    - `MarketIndex` 테이블: `market_name`, `date` 컬럼에 대한 인덱스.
    #    - 느린 쿼리는 DB 연결을 오래 점유하여 (2013) 오류를 유발할 수 있습니다. `EXPLAIN`을 사용하여 쿼리 실행 계획을 분석하고 최적화합니다.
    #
    # 5. 네트워크 문제:
    #    - Django 애플리케이션 서버와 MySQL 서버 간의 네트워크 연결이 안정적인지 확인합니다. (방화벽, DNS 등)
    #
    # 6. 웹 서버 타임아웃 (Gunicorn, uWSGI 등):
    #    - 웹 서버의 워커 타임아웃이 너무 짧으면, 긴 DB 작업이나 외부 API 호출 중에 요청이 중단될 수 있습니다.
    # ==================================================================================

    close_old_connections() # 요청 시작 시 오래된 연결 정리
    try:
        user_query_from_get = request.GET.get('query') 
        period = request.GET.get('period', '6m')

        stock_name_for_search_bar = user_query_from_get.strip() if user_query_from_get else ""
        query_for_data_processing = user_query_from_get.strip() if user_query_from_get else "005930"

        resolved_stock_code = None
        resolved_stock_name_for_display = query_for_data_processing 

        all_stocks_list = get_krx_stock_list_for_chart_cached()

        if query_for_data_processing.isdigit() and len(query_for_data_processing) == 6:
            resolved_stock_code = query_for_data_processing
            stock_match = next((s for s in all_stocks_list if s['code'] == query_for_data_processing), None)
            if stock_match:
                resolved_stock_name_for_display = stock_match['name']
        else: 
            normalized_query_for_data = normalize_stock_name(query_for_data_processing)
            stock_match = next((s for s in all_stocks_list if normalize_stock_name(s['name']) == normalized_query_for_data), None)
            if stock_match:
                resolved_stock_code = stock_match['code']
                resolved_stock_name_for_display = stock_match['name']
            else: 
                partial_match = next((s for s in all_stocks_list if normalized_query_for_data in normalize_stock_name(s['name'])), None)
                if partial_match:
                    resolved_stock_code = partial_match['code']
                    resolved_stock_name_for_display = partial_match['name']

        if not resolved_stock_code:
            samsung_default = next((s for s in all_stocks_list if s['code'] == '005930'), None)
            if samsung_default:
                resolved_stock_code = samsung_default['code']
                resolved_stock_name_for_display = samsung_default['name']
            elif all_stocks_list : 
                 resolved_stock_code = all_stocks_list[0]['code']
                 resolved_stock_name_for_display = all_stocks_list[0]['name']
            else: 
                resolved_stock_code = '005930' 
                resolved_stock_name_for_display = '삼성전자'


        stock_info_db = get_stock_info_from_db(resolved_stock_code)
        df_db_data, db_stock_name_from_data_func = get_stock_price_data_from_db(resolved_stock_code, period)

        final_display_name_for_page_elements = db_stock_name_from_data_func if db_stock_name_from_data_func != resolved_stock_code else resolved_stock_name_for_display


        candle_dates, open_prices, high_prices, low_prices, close_prices, volume_data = [], [], [], [], [], []
        ma5_data, ma20_data = [], []

        if not df_db_data.empty:
            candle_dates = df_db_data['datetime'].dt.strftime('%Y-%m-%d').tolist()
            open_prices = df_db_data['open'].tolist()
            high_prices = df_db_data['high'].tolist()
            low_prices = df_db_data['low'].tolist()
            close_prices = df_db_data['close'].tolist()
            volume_data = df_db_data['volume'].tolist()
            if 'ma5' in df_db_data.columns:
                ma5_data = [round(x, 2) if pd.notnull(x) else None for x in df_db_data['ma5'].tolist()]
            if 'ma20' in df_db_data.columns:
                ma20_data = [round(x, 2) if pd.notnull(x) else None for x in df_db_data['ma20'].tolist()]

        fifty_two_week_high, fifty_two_week_low = None, None
        if StockPrice and resolved_stock_code:
            cache_key_52wk = f'52wk_hl_{resolved_stock_code}'
            cached_52wk = cache.get(cache_key_52wk)
            if cached_52wk:
                fifty_two_week_high = cached_52wk['high']
                fifty_two_week_low = cached_52wk['low']
            else:
                close_old_connections() # DB 조회 전 연결 확인/정리
                try:
                    one_year_ago_date = (datetime.now() - timedelta(weeks=52)).date()
                    aggregate_52wk = StockPrice.objects.filter(
                        stock_code=resolved_stock_code,
                        date__gte=one_year_ago_date
                    ).aggregate(max_high=Max('high_price'), min_low=Min('low_price')) # 인덱스: (stock_code, date), high_price, low_price
                    fifty_two_week_high = aggregate_52wk.get('max_high')
                    fifty_two_week_low = aggregate_52wk.get('min_low')
                    cache.set(cache_key_52wk, {'high': fifty_two_week_high, 'low': fifty_two_week_low}, timeout=CACHE_TTL_52WK)
                except Exception as e_52wk:
                    print(f"[ERROR][chart_app] Error fetching 52wk high/low for {resolved_stock_code}: {e_52wk}")
                    traceback.print_exc()

        market_cap_rankings = []
        if StockPrice:
            latest_overall_date_obj = None
            close_old_connections() # DB 조회 전 연결 확인/정리
            try:
                latest_overall_date_obj = StockPrice.objects.aggregate(max_date=Max('date')) # 인덱스: date
            except Exception as e_latest_date:
                print(f"[ERROR][chart_app] Error fetching latest_overall_date: {e_latest_date}")
                traceback.print_exc()

            latest_overall_date = latest_overall_date_obj['max_date'] if latest_overall_date_obj and latest_overall_date_obj['max_date'] else None

            if latest_overall_date:
                cache_key_mkt_cap = f'market_cap_rankings_{latest_overall_date.strftime("%Y%m%d")}'
                cached_rankings = cache.get(cache_key_mkt_cap)
                if cached_rankings:
                    market_cap_rankings = cached_rankings
                else:
                    close_old_connections() # DB 조회 전 연결 확인/정리
                    try:
                        # 인덱스: (date, market_cap DESC)
                        top_stocks_qs = StockPrice.objects.filter(date=latest_overall_date, market_cap__isnull=False) \
                                        .order_by('-market_cap')[:30] \
                                        .values('stock_name', 'stock_code', 'market_cap', 'market_name')
                        for stock_data in top_stocks_qs:
                            market_cap_rankings.append({
                                'name': stock_data['stock_name'],
                                'code': stock_data['stock_code'],
                                'market_cap': int(stock_data['market_cap']) if stock_data['market_cap'] else 0,
                                'market': stock_data['market_name']
                            })
                        cache.set(cache_key_mkt_cap, market_cap_rankings, timeout=CACHE_TTL_RANKINGS)
                    except Exception as e_mkt_cap:
                        print(f"[ERROR][chart_app] Error fetching/caching market cap rankings from DB: {e_mkt_cap}")
                        traceback.print_exc()
            else:
                print("[WARNING][chart_app] Could not determine latest overall date for market cap rankings.")

        top5_kospi_gainers_list, top5_kosdaq_gainers_list = [], []
        top5_kospi_losers_list, top5_kosdaq_losers_list = [], []
        if StockPrice:
            latest_stock_data_date_obj_for_top_bottom = None
            close_old_connections() # DB 조회 전 연결 확인/정리
            try:
                latest_stock_data_date_obj_for_top_bottom = StockPrice.objects.order_by('-date').first() # 인덱스: date
            except Exception as e_latest_stock_date:
                 print(f"[ERROR][chart_app] Error fetching latest_stock_data_date_obj_for_top_bottom: {e_latest_stock_date}")
                 traceback.print_exc()

            if latest_stock_data_date_obj_for_top_bottom:
                latest_date = latest_stock_data_date_obj_for_top_bottom.date
                cache_key_top_bottom = f'top_bottom_stocks_{latest_date.strftime("%Y%m%d")}'
                cached_top_bottom = cache.get(cache_key_top_bottom)

                if cached_top_bottom:
                    top5_kospi_gainers_list = cached_top_bottom.get('kp_gain', [])
                    top5_kosdaq_gainers_list = cached_top_bottom.get('kd_gain', [])
                    top5_kospi_losers_list = cached_top_bottom.get('kp_lose', [])
                    top5_kosdaq_losers_list = cached_top_bottom.get('kd_lose', [])
                else:
                    def format_stock_list(qs):
                        formatted_list = []
                        for stock in qs:
                            change_percent = stock.change * 100 if stock.change is not None else 0.0
                            status_color = "price-change-up" if change_percent > 0 else "price-change-down" if change_percent < 0 else "price-change-neutral"
                            formatted_list.append({
                                'name': stock.stock_name, 'code': stock.stock_code,
                                'change_display': f"{'+' if change_percent > 0 else ''}{change_percent:.2f}%",
                                'status': status_color, 'close': stock.close_price
                            })
                        return formatted_list
                    
                    close_old_connections() # DB 조회 전 연결 확인/정리
                    try:
                        # 인덱스: (market_name, date, change DESC/ASC)
                        kospi_top5_gainers_qs = StockPrice.objects.filter(market_name='KOSPI', date=latest_date, change__isnull=False).order_by('-change')[:5]
                        kosdaq_top5_gainers_qs = StockPrice.objects.filter(market_name='KOSDAQ', date=latest_date, change__isnull=False).order_by('-change')[:5]
                        kospi_top5_losers_qs = StockPrice.objects.filter(market_name='KOSPI', date=latest_date, change__isnull=False).order_by('change')[:5]
                        kosdaq_top5_losers_qs = StockPrice.objects.filter(market_name='KOSDAQ', date=latest_date, change__isnull=False).order_by('change')[:5]

                        top5_kospi_gainers_list = format_stock_list(kospi_top5_gainers_qs)
                        top5_kosdaq_gainers_list = format_stock_list(kosdaq_top5_gainers_qs)
                        top5_kospi_losers_list = format_stock_list(kospi_top5_losers_qs)
                        top5_kosdaq_losers_list = format_stock_list(kosdaq_top5_losers_qs)
                        
                        data_to_cache_tb = {
                            'kp_gain': top5_kospi_gainers_list, 'kd_gain': top5_kosdaq_gainers_list,
                            'kp_lose': top5_kospi_losers_list, 'kd_lose': top5_kosdaq_losers_list
                        }
                        cache.set(cache_key_top_bottom, data_to_cache_tb, timeout=CACHE_TTL_RANKINGS)
                    except Exception as e_top_bottom:
                        print(f"[ERROR][chart_app] Error fetching/caching top/bottom stocks: {e_top_bottom}")
                        traceback.print_exc()

        markets_summary = []
        cache_key_market_summary = 'markets_summary_data_v2_yfinance'
        cached_summary = cache.get(cache_key_market_summary)

        if cached_summary:
            markets_summary = cached_summary
        else:
            if MarketIndex: # 인덱스: (market_name, date DESC)
                market_names_for_summary_db = ['KOSPI', 'KOSDAQ']
                for market_label in market_names_for_summary_db:
                    close_old_connections() # DB 조회 전 연결 확인/정리
                    try:
                        latest_index_data = MarketIndex.objects.filter(market_name=market_label).order_by('-date').first()
                        if latest_index_data:
                            change_display_val = ""
                            if latest_index_data.change_value is not None and latest_index_data.change_percent is not None:
                                prefix = "▲ " if latest_index_data.change_value > 0 else "▼ " if latest_index_data.change_value < 0 else ""
                                change_display_val = f"{prefix}{abs(latest_index_data.change_value):.2f} ({latest_index_data.change_percent:+.2f}%)"
                            status_class = "price-change-neutral"
                            if latest_index_data.change_value is not None:
                                if latest_index_data.change_value > 0: status_class = "price-change-up"
                                elif latest_index_data.change_value < 0: status_class = "price-change-down"
                            markets_summary.append({
                                'name': market_label, 'value': f"{latest_index_data.close_price:,.2f}",
                                'change': change_display_val, 'status': status_class
                            })
                        else: 
                            markets_summary.append({'name': market_label, 'value': 'N/A (DB 데이터 없음)', 'change': '', 'status': 'price-change-neutral'})
                    except Exception as e_market_idx:
                        print(f"[ERROR][chart_app] Error fetching market index for {market_label}: {e_market_idx}")
                        traceback.print_exc()
                        markets_summary.append({'name': market_label, 'value': 'N/A (DB 오류)', 'change': '', 'status': 'price-change-neutral'})
            else: 
                markets_summary.append({'name': 'KOSPI', 'value': 'N/A (DB 연결 확인)', 'change': '', 'status': 'price-change-neutral'})
                markets_summary.append({'name': 'KOSDAQ', 'value': 'N/A (DB 연결 확인)', 'change': '', 'status': 'price-change-neutral'})
            
            # --- yfinance 호출 전후로 DB 연결 관리 ---
            close_old_connections()
            try:
                global_indices = {'S&P 500': '^GSPC', '나스닥': '^IXIC'}
                for name, ticker_symbol in global_indices.items():
                    try:
                        ticker = yf.Ticker(ticker_symbol)
                        hist = ticker.history(period="5d", auto_adjust=False) 
                        
                        if hist.empty or len(hist) < 2:
                            print(f"[WARNING][chart_app] yfinance로 {name} 데이터 가져오기 실패: 데이터 부족 (가져온 행 수: {len(hist)})")
                            markets_summary.append({'name': name, 'value': 'N/A (데이터 부족)', 'change': '', 'status': 'price-change-neutral'})
                            continue

                        hist = hist.sort_index(ascending=True)
                        current_price = hist['Close'].iloc[-1]
                        prev_close = hist['Close'].iloc[-2]
                        change_value = current_price - prev_close
                        change_percent = (change_value / prev_close) * 100 if prev_close != 0 else 0
                        prefix = "▲ " if change_value > 0 else "▼ " if change_value < 0 else ""
                        change_display_val = f"{prefix}{abs(change_value):.2f} ({change_percent:+.2f}%)"
                        status_class = "price-change-neutral"
                        if change_value > 0: status_class = "price-change-up"
                        elif change_value < 0: status_class = "price-change-down"
                        markets_summary.append({
                            'name': name, 'value': f"{current_price:,.2f}",
                            'change': change_display_val, 'status': status_class
                        })
                    except Exception as e_yf: # yfinance 호출 관련 예외 처리
                        print(f"[ERROR][chart_app] yfinance로 {name} 데이터 가져오기 실패: {e_yf}")
                        traceback.print_exc()
                        markets_summary.append({'name': name, 'value': 'N/A (오류)', 'change': '', 'status': 'price-change-neutral'})
            finally:
                # yfinance 호출 후 다시 DB 연결 상태 확인/정리
                close_old_connections()
            
            cache.set(cache_key_market_summary, markets_summary, timeout=CACHE_TTL_MARKET_SUMMARY)

        context = {
            'stock_name_searched': stock_name_for_search_bar,
            'stock_name_displayed': final_display_name_for_page_elements,
            'period': period,
            'candle_dates': json.dumps(candle_dates),
            'open_prices': json.dumps(open_prices),
            'high_prices': json.dumps(high_prices),
            'low_prices': json.dumps(low_prices),
            'close_prices': json.dumps(close_prices),
            'ma5': json.dumps(ma5_data),
            'ma20': json.dumps(ma20_data),
            'volume': json.dumps(volume_data),
            'stock_info': stock_info_db,
            'fifty_two_week_high': fifty_two_week_high,
            'fifty_two_week_low': fifty_two_week_low,
            'top5_kospi_gainers': top5_kospi_gainers_list,
            'top5_kosdaq_gainers': top5_kosdaq_gainers_list,
            'top5_kospi_losers': top5_kospi_losers_list,
            'top5_kosdaq_losers': top5_kosdaq_losers_list,
            'markets': markets_summary,
            'market_cap_rankings': market_cap_rankings,
        }
        return render(request, 'chart.html', context)

    except Exception as e:
        # 전역적인 예외 처리. OperationalError 포함 가능.
        print(f"[CRITICAL][chart_app] chart_view에서 처리되지 않은 심각한 오류 발생: {e}")
        traceback.print_exc()
        # 사용자에게 보여줄 수 있는 오류 페이지나 간단한 메시지를 반환하는 것이 좋습니다.
        # 여기서는 간단히 500 에러를 발생시키거나, 별도의 오류 템플릿을 렌더링 할 수 있습니다.
        # 예를 들어: return render(request, 'error_page.html', {'error_message': str(e)})
        # 또는 JsonResponse({'error': '서버 내부 오류가 발생했습니다.'}, status=500)
        # 여기서는 원래 코드의 context 구조를 최대한 유지하며 오류 메시지를 전달 시도
        error_context = {
            'stock_name_searched': request.GET.get('query', '').strip(),
            'period': request.GET.get('period', '6m'),
            'error_message': f'차트 데이터를 불러오는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요. (서버 오류: {type(e).__name__})',
            # 필수 컨텍스트 변수들을 빈 값 또는 기본값으로 채워 템플릿 렌더링 오류 방지
            'candle_dates': '[]', 'open_prices': '[]', 'high_prices': '[]', 'low_prices': '[]', 
            'close_prices': '[]', 'ma5': '[]', 'ma20': '[]', 'volume': '[]',
            'stock_info': None, 'fifty_two_week_high': None, 'fifty_two_week_low': None,
            'top5_kospi_gainers': [], 'top5_kosdaq_gainers': [],
            'top5_kospi_losers': [], 'top5_kosdaq_losers': [],
            'markets': [], 'market_cap_rankings': []
        }
        # 오류 발생 시에도 chart.html을 렌더링하되, error_message를 전달하여 템플릿에서 표시할 수 있도록 합니다.
        # chart.html 템플릿에서 {{ error_message }} 등을 통해 오류를 표시하는 로직 추가 필요.
        # 또한, 클라이언트 측 JS에서 chartDiv.innerHTML에 오류 메시지를 표시할 수 있도록,
        # candle_dates 등이 비어있을 때 JS가 적절히 처리하는지 확인 필요.
        # 현재 chart.js는 candle_dates 등이 비어있으면 "차트 데이터를 불러올 수 없습니다..." 메시지를 표시함.
        return render(request, 'chart.html', error_context)


def get_realtime_price(request):
    close_old_connections() # 요청 시작 시 오래된 연결 정리
    symbol_query = request.GET.get('symbol', '').strip()
    if not symbol_query:
        return JsonResponse({'error': 'Symbol or name is required'}, status=400)

    resolved_stock_code = None
    all_stocks_list = get_krx_stock_list_for_chart_cached() 

    if symbol_query.isdigit() and len(symbol_query) == 6:
        resolved_stock_code = symbol_query
    else:
        normalized_symbol_query = normalize_stock_name(symbol_query)
        stock_match = next((s for s in all_stocks_list if normalize_stock_name(s['name']) == normalized_symbol_query), None)
        if stock_match: resolved_stock_code = stock_match['code']

    if not resolved_stock_code:
         normalized_symbol_query = normalize_stock_name(symbol_query) 
         partial_match = next((s for s in all_stocks_list if normalized_symbol_query in normalize_stock_name(s['name'])), None)
         if partial_match: resolved_stock_code = partial_match['code']

    if not resolved_stock_code:
        return JsonResponse({'error': f"Symbol '{symbol_query}' not found in DB stock list."}, status=404)

    stock_info_from_db = get_stock_info_from_db(resolved_stock_code) # 이 함수 내부에서 close_old_connections 호출됨

    if stock_info_from_db and stock_info_from_db.get('current_price') is not None:
        return JsonResponse(stock_info_from_db)
    return JsonResponse({'error': f"Failed to fetch latest stock data from DB for {resolved_stock_code}"}, status=500)


def search_stocks_ajax_for_chart(request):
    close_old_connections() # 요청 시작 시 오래된 연결 정리
    query = request.GET.get('term', '').strip()
    limit = int(request.GET.get('limit', 7))

    if len(query) < 1 and '*' not in query : 
        return JsonResponse([], safe=False)

    results = []
    try:
        # get_krx_stock_list_for_chart_cached 내부에서 캐시 미스 시 DB 조회하므로,
        # 해당 함수 내부에서 close_old_connections가 호출됨.
        all_krx_stocks = get_krx_stock_list_for_chart_cached()
        if not all_krx_stocks:
            return JsonResponse({'error': '종목 목록을 불러올 수 없습니다. 잠시 후 다시 시도해주세요.'}, status=200) 
        
        if query == '*': 
            results = [
                {"label": f"{s.get('name','')} ({s.get('code','')}, {s.get('market','')})", 
                 "value": s.get('name',''), 
                 "code": s.get('code',''), 
                 "market": s.get('market','').upper()} 
                for s in all_krx_stocks[:limit]
            ]
        else:
            normalized_query = normalize_stock_name(query)
            
            for stock in all_krx_stocks:
                stock_name_original = stock.get('name', '')
                stock_code_original = stock.get('code', '')
                stock_market_original = stock.get('market', '')

                normalized_stock_name = normalize_stock_name(stock_name_original)
                
                name_match = False
                if normalized_query and normalized_stock_name:
                    name_match = normalized_query in normalized_stock_name
                
                code_match = (query.isdigit() and len(query) == 6 and query == stock_code_original)
                exact_name_match = (not query.isdigit() and normalized_query and normalized_stock_name and normalized_query == normalized_stock_name)


                if name_match or code_match or exact_name_match:
                    results.append({
                        "label": f"{stock_name_original} ({stock_code_original}, {stock_market_original})", 
                        "value": stock_name_original, 
                        "code": stock_code_original, 
                        "market": stock_market_original.upper() 
                    })
                if len(results) >= limit:
                    break
             
    except Exception as e:
        print(f"[CRITICAL][chart/search_stocks_ajax] Unexpected error for query '{query}': {e}\n{traceback.format_exc()}")
        return JsonResponse({'error': '검색 중 예기치 않은 서버 오류가 발생했습니다. 관리자에게 문의해주세요.'}, status=200)
    
    return JsonResponse(results, safe=False)
