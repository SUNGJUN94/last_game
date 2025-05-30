import pandas as pd
from django.shortcuts import render
import json
from datetime import timedelta, datetime # datetime 추가
import yfinance as yf
from django.http import JsonResponse
import FinanceDataReader as fdr
from django.core.cache import cache # Django 캐시 사용
import traceback # traceback 추가

# predict_info 앱의 모델을 가져옵니다. (앱 간 모델 참조)
try:
    from predict_info.models import StockPrice
except ImportError:
    print("[WARNING] chart/views.py: predict_info.models.StockPrice를 import할 수 없습니다. 급등주 기능이 제한될 수 있습니다.")
    StockPrice = None

def get_krx_stock_list_for_chart_cached():
    cache_key = 'krx_stock_list_chart_app_v1'
    cached_list = cache.get(cache_key)
    if cached_list is not None:
        return cached_list
    try:
        print("[INFO][chart_app] Fetching KRX stock list from FDR for autocomplete cache...")
        df_krx_fdr = fdr.StockListing('KRX')
        if df_krx_fdr.empty:
            print("[WARNING][chart_app] FDR StockListing('KRX') returned an empty DataFrame.")
            cache.set(cache_key, [], timeout=60*10)
            return []
        code_col = 'Symbol' if 'Symbol' in df_krx_fdr.columns else 'Code'
        if 'Name' not in df_krx_fdr.columns or code_col not in df_krx_fdr.columns:
            print(f"[ERROR][chart_app] Essential columns ('Name', '{code_col}') not found in FDR StockListing.")
            cache.set(cache_key, [], timeout=60*10)
            return []
        df_krx_fdr = df_krx_fdr[['Name', code_col, 'Market']].dropna(subset=['Name', code_col])
        df_krx_fdr = df_krx_fdr.drop_duplicates(subset=[code_col])
        stock_list_of_dicts = []
        for _, row in df_krx_fdr.iterrows():
            stock_list_of_dicts.append({
                'name': str(row['Name']).strip(),
                'code': str(row[code_col]).strip(),
                'market': str(row.get('Market', '')).strip()
            })
        cache.set(cache_key, stock_list_of_dicts, timeout=60*60*24)
        print(f"[INFO][chart_app] KRX stock list fetched and cached. Total: {len(stock_list_of_dicts)} stocks.")
        return stock_list_of_dicts
    except Exception as e:
        print(f"[ERROR][chart_app] Error fetching or caching KRX stock list: {e}")
        traceback.print_exc()
        cache.set(cache_key, [], timeout=60*10)
        return []

def get_symbol_map_for_chart():
    map_cache_key = 'krx_symbol_map_chart_app_v1'
    cached_map = cache.get(map_cache_key)
    if cached_map:
        return cached_map
    stock_list = get_krx_stock_list_for_chart_cached()
    symbol_map = {}
    if not stock_list:
        return symbol_map
    for stock_item in stock_list:
        name = stock_item['name']
        code = stock_item['code']
        market_suffix = ".KS" if stock_item['market'] == 'KOSPI' else ".KQ" if stock_item['market'] == 'KOSDAQ' else ""
        if market_suffix:
            ticker = f"{code}{market_suffix}"
            symbol_map[name] = ticker
            symbol_map[code] = ticker
    cache.set(map_cache_key, symbol_map, timeout=60*60*24)
    return symbol_map

def get_stock_price(symbol_or_name):
    symbol_map = get_symbol_map_for_chart()
    is_code = symbol_or_name.isdigit() and len(symbol_or_name) == 6
    final_symbol_for_yf = None
    stock_name_for_display = symbol_or_name

    if is_code:
        final_symbol_for_yf = symbol_map.get(symbol_or_name)
        stock_list = get_krx_stock_list_for_chart_cached()
        name_found = next((s['name'] for s in stock_list if s['code'] == symbol_or_name), None)
        if name_found:
            stock_name_for_display = name_found
    else:
        final_symbol_for_yf = symbol_map.get(symbol_or_name)
        stock_name_for_display = symbol_or_name

    if not final_symbol_for_yf:
        stock_list_all = get_krx_stock_list_for_chart_cached()
        matched_stock = None
        if is_code:
             matched_stock = next((s for s in stock_list_all if s['code'] == symbol_or_name), None)
        else:
             matched_stock = next((s for s in stock_list_all if s['name'] == symbol_or_name), None)
        if matched_stock:
            market_suffix = ".KS" if matched_stock['market'] == 'KOSPI' else ".KQ" if matched_stock['market'] == 'KOSDAQ' else ""
            if market_suffix:
                final_symbol_for_yf = f"{matched_stock['code']}{market_suffix}"
                stock_name_for_display = matched_stock['name']
            else:
                 print(f"[WARNING][chart_app] Market for {symbol_or_name} not supported by yfinance ticker format.")
                 return None
        else:
            print(f"[WARNING][chart_app] Could not resolve to a yfinance ticker for: {symbol_or_name}")
            return None
    try:
        stock = yf.Ticker(final_symbol_for_yf)
        info = stock.info
        display_name = info.get('shortName', info.get('longName', stock_name_for_display))
        
        current_price_val = info.get('regularMarketPrice')
        previous_close_val = info.get('regularMarketPreviousClose')
        change_val = None
        calculated_change_percent = None
        market_cap = info.get('marketCap')  # 시가총액 정보 가져오기

        if current_price_val is not None and previous_close_val is not None:
            change_val = current_price_val - previous_close_val
            if previous_close_val != 0:
                calculated_change_percent = (change_val / previous_close_val) * 100
            else:
                calculated_change_percent = 0.0 if change_val == 0 else None
        return {
            'current_price': current_price_val,
            'previous_close': previous_close_val,
            'change': change_val,
            'change_percent': calculated_change_percent,
            'name': display_name,
            'code': final_symbol_for_yf.split('.')[0],
            'market_cap': market_cap  # 시가총액 정보 추가
        }
    except Exception as e:
        print(f"[ERROR][chart_app] Error fetching stock data for symbol: {final_symbol_for_yf}, Exception: {e}")
        traceback.print_exc()
        return None

def get_stock_price_data(query_name_or_code, period_str):
    stock_code_to_fetch = None
    stock_name_for_display = query_name_or_code

    if query_name_or_code.isdigit() and len(query_name_or_code) == 6:
        stock_code_to_fetch = query_name_or_code
        stock_list = get_krx_stock_list_for_chart_cached()
        name_found = next((s['name'] for s in stock_list if s['code'] == stock_code_to_fetch), None)
        if name_found:
            stock_name_for_display = name_found
    else:
        stock_list = get_krx_stock_list_for_chart_cached()
        code_found = next((s['code'] for s in stock_list if s['name'] == query_name_or_code), None)
        if code_found:
            stock_code_to_fetch = code_found
            stock_name_for_display = query_name_or_code
        else:
            print(f"[WARNING][chart_app] Code not found for name: {query_name_or_code}")
            return pd.DataFrame(), query_name_or_code

    if not stock_code_to_fetch:
         print(f"[WARNING][chart_app] Failed to resolve query to stock code: {query_name_or_code}")
         return pd.DataFrame(), query_name_or_code
    
    end_date = datetime.now()
    if period_str == '1m':
        start_date = end_date - pd.DateOffset(months=1)
    elif period_str == '3m':
        start_date = end_date - pd.DateOffset(months=3)
    elif period_str == '6m':
        start_date = end_date - pd.DateOffset(months=6)
    elif period_str == '1y':
        start_date = end_date - pd.DateOffset(years=1)
    elif period_str == '3y':
        start_date = end_date - pd.DateOffset(years=3)
    elif period_str == '5y':
        start_date = end_date - pd.DateOffset(years=5)
    elif period_str == 'all':
        start_date = datetime(1990, 1, 1)
    else:
        start_date = end_date - pd.DateOffset(months=6)

    try:
        df = fdr.DataReader(stock_code_to_fetch, start_date, end_date)
        df = df.reset_index()
        df = df.sort_values('Date')
        df = df.rename(columns={'Date': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        return df, stock_name_for_display
    except Exception as e:
        print(f"[ERROR][chart_app] Error fetching FDR data for {stock_code_to_fetch}: {e}")
        return pd.DataFrame(), stock_name_for_display

def chart_view(request):
    query = request.GET.get('query', '삼성전자').strip()
    period = request.GET.get('period', '6m')

    stock_info_yf = get_stock_price(query)
    df_fdr, stock_name_display = get_stock_price_data(query, period)

    #add1
    # 시가총액 순위 데이터 가져오기
    market_cap_rankings = []
    try:
        kospi_stocks = fdr.StockListing('KOSPI')
        kosdaq_stocks = fdr.StockListing('KOSDAQ')

        # 실제 컬럼명 확인
        print('[DEBUG] KOSPI columns:', kospi_stocks.columns)
        print('[DEBUG] KOSDAQ columns:', kosdaq_stocks.columns)

        # 컬럼명 매핑
        code_col = 'Symbol' if 'Symbol' in kospi_stocks.columns else 'Code'
        market_cap_col = 'MarketCap'
        if market_cap_col not in kospi_stocks.columns:
            if 'Marcap' in kospi_stocks.columns:
                market_cap_col = 'Marcap'
            elif '시가총액' in kospi_stocks.columns:
                market_cap_col = '시가총액'
            else:
                raise Exception('시가총액 컬럼을 찾을 수 없습니다.')

        # 필요한 컬럼만 선택하고 결측치 제거
        kospi_stocks = kospi_stocks[['Name', code_col, 'Market', market_cap_col]].dropna(subset=[market_cap_col])
        kosdaq_stocks = kosdaq_stocks[['Name', code_col, 'Market', market_cap_col]].dropna(subset=[market_cap_col])

        all_stocks = pd.concat([kospi_stocks, kosdaq_stocks])
        top_30 = all_stocks.nlargest(30, market_cap_col)

        market_cap_rankings = []
        for _, stock in top_30.iterrows():
            market_cap_rankings.append({
                'name': stock['Name'],
                'code': stock[code_col],
                'market_cap': int(stock[market_cap_col]),
                'market': 'KOSPI' if stock['Market'] == 'KOSPI' else 'KOSDAQ'
            })
        print(f"[INFO][chart_app] Successfully fetched {len(market_cap_rankings)} market cap rankings")
    except Exception as e:
        print(f"[ERROR][chart_app] Error fetching market cap rankings: {e}")
        traceback.print_exc()
    #add2

    candle_dates = []
    open_prices, high_prices, low_prices, close_prices, volume_data = [], [], [], [], []
    ma5_data, ma20_data = [], []

    if not df_fdr.empty:
        candle_dates = df_fdr['datetime'].dt.strftime('%Y-%m-%d').tolist()
        open_prices = df_fdr['open'].tolist()
        high_prices = df_fdr['high'].tolist()
        low_prices = df_fdr['low'].tolist()
        close_prices = df_fdr['close'].tolist()
        volume_data = df_fdr['volume'].tolist()
        
        ma5_series = df_fdr['close'].rolling(window=5).mean()
        ma20_series = df_fdr['close'].rolling(window=20).mean()
        
        ma5_data = [round(x, 2) if pd.notnull(x) else None for x in ma5_series.tolist()]
        ma20_data = [round(x, 2) if pd.notnull(x) else None for x in ma20_series.tolist()]

    fifty_two_week_high, fifty_two_week_low = None, None
    if not df_fdr.empty:
        df_all_for_52wk, _ = get_stock_price_data(query, 'all')
        if not df_all_for_52wk.empty:
            one_year_ago = pd.Timestamp.now() - pd.Timedelta(weeks=52)
            if not pd.api.types.is_datetime64_any_dtype(df_all_for_52wk['datetime']):
                df_all_for_52wk['datetime'] = pd.to_datetime(df_all_for_52wk['datetime'])
            recent_df_for_52wk = df_all_for_52wk[df_all_for_52wk['datetime'] >= one_year_ago]
            if not recent_df_for_52wk.empty:
                fifty_two_week_high = recent_df_for_52wk['high'].max()
                fifty_two_week_low = recent_df_for_52wk['low'].min()

    top5_kospi_gainers_list = []
    top5_kosdaq_gainers_list = []
    top5_kospi_losers_list = []
    top5_kosdaq_losers_list = []
    if StockPrice:
        latest_stock_data_date_obj = StockPrice.objects.order_by('-date').first()
        if latest_stock_data_date_obj:
            latest_date = latest_stock_data_date_obj.date
            # 급등주 TOP 5
            kospi_top5 = StockPrice.objects.filter(market_name='KOSPI', date=latest_date, change_percent__isnull=False).order_by('-change_percent')[:5]
            kosdaq_top5 = StockPrice.objects.filter(market_name='KOSDAQ', date=latest_date, change_percent__isnull=False).order_by('-change_percent')[:5]
            
            #add1
            # 급락주 TOP 5
            kospi_bottom5 = StockPrice.objects.filter(market_name='KOSPI', date=latest_date, change_percent__isnull=False).order_by('change_percent')[:5]
            kosdaq_bottom5 = StockPrice.objects.filter(market_name='KOSDAQ', date=latest_date, change_percent__isnull=False).order_by('change_percent')[:5]
            #add2

            for stock in kospi_top5:
                status_color = "price-change-up" if stock.change_percent > 0 else "price-change-down" if stock.change_percent < 0 else "price-change-neutral"
                top5_kospi_gainers_list.append({
                    'name': stock.stock_name, 
                    'code': stock.stock_code, 
                    'change_display': f"{'+' if stock.change_percent > 0 else ''}{stock.change_percent:.2f}%",
                    'status': status_color,
                    'close': stock.close_price
                })
            for stock in kosdaq_top5:
                status_color = "price-change-up" if stock.change_percent > 0 else "price-change-down" if stock.change_percent < 0 else "price-change-neutral"
                top5_kosdaq_gainers_list.append({
                    'name': stock.stock_name, 
                    'code': stock.stock_code, 
                    'change_display': f"{'+' if stock.change_percent > 0 else ''}{stock.change_percent:.2f}%",
                    'status': status_color,
                    'close': stock.close_price
                })

                #org1
                #250526_16_44
                #git hub 코드 직접 복사/붙여넣기한 소스코드_5개 데이터가 중복으로 5*5개 나옴
                # 급락주 데이터 추가
                # for stock in kospi_bottom5:
                #     status_color = "price-change-up" if stock.change_percent > 0 else "price-change-down" if stock.change_percent < 0 else "price-change-neutral"
                #     top5_kospi_losers_list.append({
                #         'name': stock.stock_name, 
                #         'code': stock.stock_code, 
                #         'change_display': f"{'+' if stock.change_percent > 0 else ''}{stock.change_percent:.2f}%",
                #         'status': status_color,
                #         'close': stock.close_price
                #     })
                # for stock in kosdaq_bottom5:
                #     status_color = "price-change-up" if stock.change_percent > 0 else "price-change-down" if stock.change_percent < 0 else "price-change-neutral"
                #     top5_kosdaq_losers_list.append({
                #         'name': stock.stock_name, 
                #         'code': stock.stock_code, 
                #         'change_display': f"{'+' if stock.change_percent > 0 else ''}{stock.change_percent:.2f}%",
                #         'status': status_color,
                #         'close': stock.close_price
                #     })
                #org2
                #edit1
                # 중복 방지를 위해 리스트 초기화 확인 및 중복 데이터 제거
                top5_kospi_losers_list = []
                top5_kosdaq_losers_list = []

                # 급락주 TOP 5 (중복 방지)
                kospi_bottom5 = StockPrice.objects.filter(market_name='KOSPI', date=latest_date, change_percent__isnull=False).order_by('change_percent')[:5]
                kosdaq_bottom5 = StockPrice.objects.filter(market_name='KOSDAQ', date=latest_date, change_percent__isnull=False).order_by('change_percent')[:5]

                # 중복 방지를 위해 이미 추가된 종목 코드를 추적
                added_kospi_codes = set()
                added_kosdaq_codes = set()

                for stock in kospi_bottom5:
                    if stock.stock_code not in added_kospi_codes:  # 중복 확인
                        status_color = "price-change-up" if stock.change_percent > 0 else "price-change-down" if stock.change_percent < 0 else "price-change-neutral"
                        top5_kospi_losers_list.append({
                            'name': stock.stock_name,
                            'code': stock.stock_code,
                            'change_display': f"{'+' if stock.change_percent > 0 else ''}{stock.change_percent:.2f}%",
                            'status': status_color,
                            'close': stock.close_price
                        })
                        added_kospi_codes.add(stock.stock_code)

                for stock in kosdaq_bottom5:
                    if stock.stock_code not in added_kosdaq_codes:  # 중복 확인
                        status_color = "price-change-up" if stock.change_percent > 0 else "price-change-down" if stock.change_percent < 0 else "price-change-neutral"
                        top5_kosdaq_losers_list.append({
                            'name': stock.stock_name,
                            'code': stock.stock_code,
                            'change_display': f"{'+' if stock.change_percent > 0 else ''}{stock.change_percent:.2f}%",
                            'status': status_color,
                            'close': stock.close_price
                        })
                        added_kosdaq_codes.add(stock.stock_code)
                #edit2
    
    markets_summary = [
        {'name': '코스피', 'value': '2,750.32', 'change': '▲ 10.21 (+0.37%)', 'status': 'price-change-up'},
        {'name': '코스닥', 'value': '850.12', 'change': '▼ 2.05 (-0.24%)', 'status': 'price-change-down'},
        {'name': 'S&P 500', 'value': '5,300.50', 'change': '▲ 20.80 (+0.39%)', 'status': 'price-change-up'},
        {'name': '나스닥', 'value': '16,500.70', 'change': '▼ 15.60 (-0.09%)', 'status': 'price-change-down'},
    ]

    context = {
        'stock_name_searched': query,
        'stock_name_displayed': stock_name_display,
        'period': period,
        'candle_dates': json.dumps(candle_dates),
        'open_prices': json.dumps(open_prices),
        'high_prices': json.dumps(high_prices),
        'low_prices': json.dumps(low_prices),
        'close_prices': json.dumps(close_prices),
        'ma5': json.dumps(ma5_data),
        'ma20': json.dumps(ma20_data),
        'volume': json.dumps(volume_data),
        'stock_info': stock_info_yf,
        'fifty_two_week_high': fifty_two_week_high,
        'fifty_two_week_low': fifty_two_week_low,
        'top5_kospi_gainers': top5_kospi_gainers_list,
        'top5_kosdaq_gainers': top5_kosdaq_gainers_list,
        'top5_kospi_losers': top5_kospi_losers_list,
        'top5_kosdaq_losers': top5_kosdaq_losers_list,
        'markets': markets_summary,
        'market_cap_rankings': market_cap_rankings,  # 시가총액 순위 데이터 추가
    }
    # 템플릿 경로 수정: 'chart/chart.html' -> 'chart.html'
    # 이 변경은 chart 앱의 templates 폴더 바로 밑에 chart.html이 있다고 가정합니다.
    # 즉, PROJECT_ROOT/chart/templates/chart.html
    return render(request, 'chart.html', context) # 여기가 수정된 부분입니다.

def get_realtime_price(request):
    symbol_query = request.GET.get('symbol', '')
    if not symbol_query:
        return JsonResponse({'error': 'Symbol or name is required'}, status=400)
    
    stock_info = get_stock_price(symbol_query)
    if stock_info and stock_info.get('current_price') is not None:
        return JsonResponse(stock_info)
    return JsonResponse({'error': 'Failed to fetch real-time stock data'}, status=500)

def search_stocks_ajax_for_chart(request):
    term = request.GET.get('term', '').strip()
    limit = int(request.GET.get('limit', 7))

    if not term:
        return JsonResponse([], safe=False)

    all_stocks_list = get_krx_stock_list_for_chart_cached()
    if not all_stocks_list:
        return JsonResponse({'error': '종목 목록을 불러올 수 없습니다. 잠시 후 다시 시도해주세요.'}, status=500)

    results = []
    term_upper = term.upper()

    for stock_item in all_stocks_list:
        stock_name_val = stock_item.get('name', '')
        stock_code_val = stock_item.get('code', '')
        match = False
        if term_upper in stock_name_val.upper():
            match = True
        elif term_upper in stock_code_val:
            match = True
        
        if match:
            results.append({
                'label': f"{stock_name_val} ({stock_code_val}) - {stock_item.get('market', '')}",
                'value': stock_name_val,
            })
        if len(results) >= limit:
            break
    return JsonResponse(results, safe=False)
