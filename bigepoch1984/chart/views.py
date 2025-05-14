import pandas as pd
from django.shortcuts import render
import json
from datetime import timedelta
import yfinance as yf
from django.http import JsonResponse
import FinanceDataReader as fdr

# 코스피 전체 종목 리스트
KOSPI_LIST = fdr.StockListing('KOSPI')
# 코스닥 전체 종목 리스트
KOSDAQ_LIST = fdr.StockListing('KOSDAQ')
# 두 시장 모두 SYMBOL_MAP에 포함 (이름, 코드 모두 key로)
SYMBOL_MAP = {row['Name']: f"{row['Code']}.KS" for _, row in KOSPI_LIST.iterrows()}
SYMBOL_MAP.update({row['Code']: f"{row['Code']}.KS" for _, row in KOSPI_LIST.iterrows()})
SYMBOL_MAP.update({row['Name']: f"{row['Code']}.KQ" for _, row in KOSDAQ_LIST.iterrows()})
SYMBOL_MAP.update({row['Code']: f"{row['Code']}.KQ" for _, row in KOSDAQ_LIST.iterrows()})

def get_stock_price(symbol):
    try:
        # 한글명 입력 시 종목코드로 변환
        symbol = SYMBOL_MAP.get(symbol, symbol)
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'current_price': info.get('regularMarketPrice', 0),
            'previous_close': info.get('regularMarketPreviousClose', 0),
            'change': info.get('regularMarketChange', 0),
            'change_percent': info.get('regularMarketChangePercent', 0),
            'name': info.get('longName', symbol)
        }
    except Exception as e:
        print(f"Error fetching stock data for symbol: {symbol}, Exception: {e}")
        return None

def get_stock_price_data(query):
    symbol = SYMBOL_MAP.get(query, query)
    code = symbol.split('.')[0]  # '005930.KS' → '005930'
    df = fdr.DataReader(code, '2020-01-01')  # 원하는 시작일~오늘까지
    df = df.reset_index()
    df = df.sort_values('Date')
    df = df.rename(columns={'Date': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    return df

def chart_view(request):
    query = request.GET.get('query', '삼성전자')
    period = request.GET.get('period', '6m')
    stock_info = get_stock_price(query)
    df = get_stock_price_data(query)
    # 기간별 데이터 추출
    if period == '1m':
        start_date = pd.Timestamp.now() - pd.DateOffset(months=1)
    elif period == '3m':
        start_date = pd.Timestamp.now() - pd.DateOffset(months=3)
    elif period == '6m':
        start_date = pd.Timestamp.now() - pd.DateOffset(months=6)
    elif period == '1y':
        start_date = pd.Timestamp.now() - pd.DateOffset(years=1)
    elif period == '3y':
        start_date = pd.Timestamp.now() - pd.DateOffset(years=3)
    elif period == '5y':
        start_date = pd.Timestamp.now() - pd.DateOffset(years=5)
    elif period == '10y':
        start_date = pd.Timestamp.now() - pd.DateOffset(years=10)
    else:
        start_date = None  # 전체
    if start_date:
        df_period = df[df['datetime'] >= start_date].copy()
    else:
        df_period = df.copy()
    # 실시간 가격을 점(marker)으로만 추가
    realtime_marker = None
    if stock_info and stock_info['current_price']:
        now = pd.Timestamp.now().floor('min')
        realtime_marker = {
            'datetime': now,
            'price': stock_info['current_price']
        }
    future_date = df['datetime'].max() + timedelta(days=14)
    candle_dates = df_period['datetime'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    open_prices = df_period['open'].tolist()
    high_prices = df_period['high'].tolist()
    low_prices = df_period['low'].tolist()
    close_prices = df_period['close'].tolist()
    volume = df_period['volume'].tolist()
    # ma5를 전체 기간의 이동평균선(전체 길이)으로 계산
    ma_all = df_period['close'].expanding().mean().tolist()
    ma20 = df_period['close'].rolling(window=20).mean().fillna(0).tolist()
    # 최근 1년(52주)간의 최고가/최저가 계산
    one_year_ago = pd.Timestamp.now() - pd.Timedelta(weeks=52)
    recent_df = df[df['datetime'] >= one_year_ago]
    fifty_two_week_high = recent_df['high'].max() if not recent_df.empty else None
    fifty_two_week_low = recent_df['low'].min() if not recent_df.empty else None
    context = {
        'stock_name': query,
        'ticker': '',
        'candle_dates': json.dumps(candle_dates),
        'open_prices': json.dumps(open_prices),
        'high_prices': json.dumps(high_prices),
        'low_prices': json.dumps(low_prices),
        'close_prices': json.dumps(close_prices),
        'ma5': json.dumps(ma_all),
        'ma20': json.dumps(ma20),
        'future_date': future_date.strftime('%Y-%m-%d %H:%M'),
        'stock_info': stock_info,
        'realtime_marker': realtime_marker,
        'fifty_two_week_high': fifty_two_week_high,
        'fifty_two_week_low': fifty_two_week_low,
        'period': period,
        'volume': json.dumps(volume),
    }
    return render(request, 'chart.html', context)

def get_realtime_price(request):
    symbol = request.GET.get('symbol', '')
    if not symbol:
        return JsonResponse({'error': 'Symbol is required'}, status=400)
    
    stock_info = get_stock_price(symbol)
    if stock_info:
        return JsonResponse(stock_info)
    return JsonResponse({'error': 'Failed to fetch stock data'}, status=500)