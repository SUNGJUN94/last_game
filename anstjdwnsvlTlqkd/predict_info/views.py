# predict_info/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings 
import pandas as pd
import numpy as np
# import joblib  # 임시 비활성화
# import tensorflow as tf  # 임시 비활성화
import FinanceDataReader as fdr 
from datetime import datetime, timedelta, date as date_type
from pandas.tseries.offsets import BDay
import os
import traceback 
import holidays 

# from .utils import calculate_manual_features  # 임시 비활성화
from .models import MarketIndex, StockPrice 

"""
# --- 임시 비활성화된 상수 정의 ---
ML_MODELS_DIR = os.path.join(settings.BASE_DIR, 'predict_info', 'ml_models')

# Colab 학습 스크립트의 FEATURE_COLUMNS와 정확히 일치해야 함 (13개)
FEATURE_COLUMNS_TRAINING = ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'BB_Lower', 'BB_Mid', 'BB_Upper', 'RSI', 'MACD', 'MACD_Hist', 'MACD_Signal']
TIME_STEPS = 10  
FUTURE_TARGET_DAYS = 5 
MIN_DATA_DAYS_FOR_TA_CALC = 100 

# --- 모델 및 스케일러 로드 함수 ---
models = {
    "kospi_technical": None, "kosdaq_technical": None,
    "kospi_comprehensive": None, "kosdaq_comprehensive": None 
}
scalers_X = {
    "kospi_technical": None, "kosdaq_technical": None,
    "kospi_comprehensive": None, "kosdaq_comprehensive": None
}
scalers_y = {
    "kospi_technical": None, "kosdaq_technical": None,
    "kospi_comprehensive": None, "kosdaq_comprehensive": None
}

def load_all_models_and_scalers():
    markets_config = {
        "kosdaq_technical": {
            "model_file": "kosdaq_technical_model.keras",
            "scaler_x_file": "kosdaq_technical_scaler_X.joblib",
            "scaler_y_file": "kosdaq_technical_scaler_y.joblib"
        },
        "kospi_technical": {
            "model_file": "kospi_technical_model.keras",
            "scaler_x_file": "kospi_technical_scaler_X.joblib",
            "scaler_y_file": "kospi_technical_scaler_y.joblib"
        }
    }

    for model_key, files in markets_config.items():
        try:
            model_path = os.path.join(ML_MODELS_DIR, files["model_file"])
            scaler_x_path = os.path.join(ML_MODELS_DIR, files["scaler_x_file"])
            scaler_y_path = os.path.join(ML_MODELS_DIR, files["scaler_y_file"])

            if os.path.exists(model_path):
                models[model_key] = tf.keras.models.load_model(model_path)
                print(f"[INFO] 모델 로드 성공: {model_path}")
            else:
                print(f"[WARNING] 모델 파일 없음: {model_path} ({model_key})")

            if os.path.exists(scaler_x_path):
                scalers_X[model_key] = joblib.load(scaler_x_path)
                print(f"[INFO] X 스케일러 로드 성공: {scaler_x_path}")
            else:
                print(f"[WARNING] X 스케일러 파일 없음: {scaler_x_path} ({model_key})")

            if os.path.exists(scaler_y_path):
                scalers_y[model_key] = joblib.load(scaler_y_path)
                print(f"[INFO] Y 스케일러 로드 성공: {scaler_y_path}")
            else:
                print(f"[WARNING] Y 스케일러 파일 없음: {scaler_y_path} ({model_key})")
        except Exception as e:
            print(f"[ERROR] {model_key} 모델/스케일러 로드 중 오류: {e}")
            traceback.print_exc()

# load_all_models_and_scalers()  # 임시 비활성화

# --- Helper Functions ---
def get_market_info_from_fdr(stock_input):
    df_krx = fdr.StockListing('KRX') 
    stock_info_fdr = None
    
    if stock_input.isdigit() and len(stock_input) == 6: 
        stock_info_fdr = df_krx[df_krx['Code'] == stock_input]
    else: 
        stock_info_fdr = df_krx[df_krx['Name'] == stock_input]

    if not stock_info_fdr.empty:
        market = stock_info_fdr.iloc[0]['Market']
        code = stock_info_fdr.iloc[0]['Code']
        name = stock_info_fdr.iloc[0]['Name']
        return market, code, name
    return None, None, None

def get_stock_info_from_db_or_fdr(stock_input_query):
    latest_db_date = StockPrice.objects.order_by('-date').first()
    market, code, name = None, None, None

    if latest_db_date:
        latest_date_val = latest_db_date.date
        stock_db_info = None
        if stock_input_query.isdigit() and len(stock_input_query) == 6:
            stock_db_info = StockPrice.objects.filter(stock_code=stock_input_query, date=latest_date_val).first()
        else:
            stock_db_info = StockPrice.objects.filter(stock_name__iexact=stock_input_query, date=latest_date_val).first()
        
        if stock_db_info:
            market = stock_db_info.market_name 
            code = stock_db_info.stock_code
            name = stock_db_info.stock_name
            return market, code, name

    market, code, name = get_market_info_from_fdr(stock_input_query)
    return market, code, name

def get_latest_stock_data_with_features(stock_code, feature_names_for_model_input): 
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=MIN_DATA_DAYS_FOR_TA_CALC * 1.7) 
        
        df_raw = fdr.DataReader(stock_code, start=start_date, end=end_date)

        if df_raw.empty or len(df_raw) < TIME_STEPS + 20: 
            return None, None

        last_data_date = pd.to_datetime(df_raw.index[-1]).date() 

        df_with_ta = calculate_manual_features(df_raw.copy())
        
        available_cols_in_df = [col for col in feature_names_for_model_input if col in df_with_ta.columns]
        if len(available_cols_in_df) != len(feature_names_for_model_input):
            missing_cols = [col for col in feature_names_for_model_input if col not in available_cols_in_df]
            print(f"[WARNING] 일부 피처가 df_with_ta에 존재하지 않습니다: {missing_cols}. 사용 가능한 피처만 사용합니다.")
        
        df_selected_features = df_with_ta[available_cols_in_df].ffill().bfill()
        df_processed = df_selected_features.dropna()

        if len(df_processed) < TIME_STEPS:
            return None, None

        recent_features_df = df_processed.tail(TIME_STEPS)
        
        if not all(col in recent_features_df.columns for col in feature_names_for_model_input):
            final_missing_cols = [col for col in feature_names_for_model_input if col not in recent_features_df.columns]
            print(f"[ERROR] 최종 입력 데이터에 필수 피처 누락: {final_missing_cols}")
            return None, None 

        return recent_features_df[feature_names_for_model_input], last_data_date 
    except Exception as e:
        print(f"[ERROR] 최신 데이터 가져오기/처리 중 오류 ({stock_code}): {e}")
        traceback.print_exc()
        return None, None

def get_future_trading_dates_list(start_date_input, num_days, country='KR'):
    if not isinstance(start_date_input, date_type):
        try:
            start_date_input = pd.to_datetime(start_date_input).date()
        except Exception as e_date:
            print(f"[ERROR] 날짜 변환 실패: {start_date_input}. 오류: {e_date}. 오늘 날짜를 기준으로 합니다.")
            start_date_input = datetime.now().date()

    kr_holidays = holidays.KR(years=[start_date_input.year, start_date_input.year + 1]) 
    future_dates = []
    current_date_pd = pd.Timestamp(start_date_input) 

    while len(future_dates) < num_days:
        current_date_pd += BDay(1) 
        if current_date_pd.date() not in kr_holidays:
            future_dates.append(current_date_pd.date())
    return future_dates
"""

# --- Views ---
def predict_info_view(request):
    context = {
        'stock_name_for_display': request.GET.get('stock_query', '삼성전자'), 
        'ticker': None,
        'error_message': None,
        'prediction_indices': [],
        'top5_kospi_gainers': [],
        'top5_kospi_losers': [],
        'top5_kosdaq_gainers': [],
        'top5_kosdaq_losers': [],
    }
    return render(request, 'predict_info/predict.html', context)

def predict_stock_price_ajax(request):
    return JsonResponse({
        'status': 'info',
        'message': '죄송합니다. 현재 예측 기능은 일시적으로 사용할 수 없습니다. 곧 서비스가 정상화될 예정입니다.'
    })

"""
# 원래 predict_stock_price_ajax 함수
def predict_stock_price_ajax_original(request):
    if request.method == 'POST':
        stock_input = request.POST.get('stock_input', '').strip()
        analysis_type = request.POST.get('analysis_type', 'technical').strip().lower() 

        if not stock_input:
            return JsonResponse({'error': '종목명 또는 종목코드를 입력해주세요.'}, status=400)

        market_raw, stock_code, stock_name = get_stock_info_from_db_or_fdr(stock_input)

        if not market_raw or not stock_code:
            return JsonResponse({'error': f"'{stock_input}'에 해당하는 종목 정보를 찾을 수 없습니다. 정확한 종목명 또는 6자리 코드를 입력해주세요."}, status=400)

        market = market_raw.lower() 
        model_key = f"{market}_{analysis_type}"
        
        selected_model = models.get(model_key)
        selected_scaler_X = scalers_X.get(model_key)
        selected_scaler_y = scalers_y.get(model_key)
        
        if not all([selected_model, selected_scaler_X, selected_scaler_y]):
            error_msg = f"{market_raw} 시장의 '{analysis_type}' 분석 모델 또는 스케일러를 로드할 수 없습니다. "
            if analysis_type == "comprehensive" and not models.get(f"{market}_comprehensive"): 
                error_msg += "종합 분석 모델은 현재 지원되지 않거나 로드되지 않았습니다."
            else:
                error_msg += "서버 설정을 확인해주세요. (모델 파일명, 경로 등)"
            return JsonResponse({'error': error_msg}, status=500)

        recent_features_df, last_data_date = get_latest_stock_data_with_features(stock_code, FEATURE_COLUMNS_TRAINING) 

        if recent_features_df is None or len(recent_features_df) != TIME_STEPS:
            return JsonResponse({'error': f"'{stock_name}({stock_code})'의 예측에 필요한 최근 {TIME_STEPS}일치 데이터를 준비하지 못했습니다. 데이터가 충분한지, 피처 생성에 문제가 없는지 확인해주세요."}, status=400)

        try:
            input_data_scaled = selected_scaler_X.transform(recent_features_df.values) 
            input_data_reshaped = input_data_scaled.reshape(1, TIME_STEPS, len(FEATURE_COLUMNS_TRAINING)) 
            prediction_scaled = selected_model.predict(input_data_reshaped, verbose=0) 
            prediction_actual_prices = selected_scaler_y.inverse_transform(prediction_scaled)[0] 
            future_dates_dt = get_future_trading_dates_list(last_data_date, FUTURE_TARGET_DAYS)
            
            predictions_output = []
            for i in range(FUTURE_TARGET_DAYS):
                predictions_output.append({
                    'date': future_dates_dt[i].strftime('%Y-%m-%d'),
                    'price': round(prediction_actual_prices[i]) 
                })
            
            print(f"[INFO] AJAX 예측 성공: {stock_name}({stock_code}), 유형: {analysis_type}")
            return JsonResponse({
                'stock_code': stock_code,
                'stock_name': stock_name,
                'market': market_raw,
                'analysis_type': analysis_type,
                'predictions': predictions_output,
                'last_data_date': last_data_date.strftime('%Y-%m-%d') if last_data_date else None
            })

        except Exception as e:
            print(f"[ERROR] AJAX 예측 중 오류: {e}")
            traceback.print_exc()
            return JsonResponse({'error': f"예측 처리 중 서버 오류 발생: {str(e)}"}, status=500)

    return JsonResponse({'error': '잘못된 요청입니다 (POST 요청 필요).'}, status=400)
"""

