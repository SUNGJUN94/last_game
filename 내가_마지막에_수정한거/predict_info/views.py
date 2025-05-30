# predict_info/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta, date as date_type
# from pandas.tseries.offsets import BDay # get_future_trading_dates_list에서 사용
import os
import traceback
import holidays # get_future_trading_dates_list에서 사용
from django.core.cache import cache
from django.contrib.auth.decorators import login_required
import json

from .utils import calculate_all_features, get_market_macro_data, PANDAS_TA_AVAILABLE # calculate_all_features는 직접 사용 안함 (generate_daily_predictions에서 사용)
from .models import PredictedStockPrice, FavoriteStock # StockPrice는 직접 사용 안함 (generate_daily_predictions에서 사용)

ML_MODELS_DIR = settings.ML_MODELS_DIR

# --- 모델 입력 피처 정의 ---
# 이 부분은 사용자의 새로운 데이터셋 및 학습된 모델의 실제 입력 피처와 정확히 일치해야 합니다.
# 코랩에서 최종적으로 모델 학습에 사용한 피처 목록을 여기에 반영해야 합니다.

# 1. 기본 OHLCV 및 변동률
BASE_OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']

# 2. 기술적 지표 (기존 + 신규, 컬럼명은 pandas-ta 생성 기준 또는 사용자가 CSV에 저장한 이름 기준)
# startup_tasks.py에서 CSV 생성 시 컬럼명과 일치해야 함
EXISTING_TA_COLS = [
    'ATR_14', 
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', # 볼린저밴드
    'RSI_14',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', # MACD
]
NEW_TA_COLS = [ # startup_tasks.py와 일치
    'STOCHk_14_3_3', 'STOCHd_14_3_3', # 스토캐스틱
    'OBV', # On Balance Volume
    'ADX_14', 'DMP_14', 'DMN_14' # ADX, DMI
]
# 만약 사용자의 새로운 데이터셋에 추가적인 기술적 지표가 있다면 여기에 추가해야 합니다.
# 예: USER_CUSTOM_TA_COLS = ['EMA_50', 'WILLR_14']

# 3. 시장 지수 데이터 (KOSPI 또는 KOSDAQ) - market_name_upper에 따라 동적으로 결정됨
# get_feature_columns_for_market 함수 내에서 처리

# 4. 거시경제 지표
MACRO_DATA_COLS = ['USD_KRW_Close', 'USD_KRW_Change']

# 5. 투자자별 매매동향
INVESTOR_COLS_MODEL_INPUT = ['Indi', 'Foreign', 'Organ'] # startup_tasks.py의 INVESTOR_COLS_FOR_CSV와 일치

# 6. 펀더멘털 데이터
FUNDAMENTAL_COLS_MODEL_INPUT = ['Marcap', 'PER', 'PBR'] # startup_tasks.py의 FUNDAMENTAL_COLS_FOR_CSV와 일치
# 만약 사용자의 새로운 데이터셋에 추가 펀더멘털 지표(예: EPS, BPS, DPS, ROE 등)가 있다면 여기에 추가해야 합니다.
# 예: USER_CUSTOM_FUNDAMENTAL_COLS = ['EPS', 'BPS', 'DPS', 'ROE']


def get_feature_columns_for_market(market_name_upper):
    """
    지정된 시장에 대한 모델 입력 피처 컬럼 목록을 반환합니다.
    이 함수는 generate_daily_predictions.py에서도 동일하게 사용될 수 있도록 일관성을 유지해야 합니다.
    사용자의 새로운 데이터셋에 맞춰 피처 목록을 정확하게 정의해야 합니다.
    """
    market_specific_index_cols = []
    if market_name_upper == "KOSPI":
        market_specific_index_cols = ['KOSPI_Close', 'KOSPI_Change']
    elif market_name_upper == "KOSDAQ":
        market_specific_index_cols = ['KOSDAQ_Close', 'KOSDAQ_Change']
    else:
        # 이 경우는 발생하지 않도록 호출하는 쪽에서 시장명을 KOSPI 또는 KOSDAQ으로 표준화해야 함
        error_message = f"Unsupported market '{market_name_upper}' for feature column definition."
        # print(f"[ERROR][get_feature_columns_for_market] {error_message}") # 로깅
        raise ValueError(error_message) 

    # 모든 피처 그룹을 조합
    final_columns = (
        BASE_OHLCV_COLS + 
        EXISTING_TA_COLS + 
        NEW_TA_COLS + 
        # USER_CUSTOM_TA_COLS + # 필요시 추가
        market_specific_index_cols + 
        MACRO_DATA_COLS +
        INVESTOR_COLS_MODEL_INPUT +
        FUNDAMENTAL_COLS_MODEL_INPUT
        # USER_CUSTOM_FUNDAMENTAL_COLS # 필요시 추가
    )
    
    # 모델 학습 시 사용된 피처의 정확한 개수와 일치해야 함
    # 이 값은 실제 학습된 모델의 입력 피처 수에 따라 달라져야 합니다.
    # 예를 들어, 기존 30개에서 새로운 피처가 추가/삭제되었다면 이 값을 조정해야 합니다.
    EXPECTED_FEATURE_COUNT = len(final_columns) # 동적으로 계산된 피처 수
    
    # print(f"[DEBUG][get_feature_columns_for_market] For {market_name_upper}, expected features ({EXPECTED_FEATURE_COUNT}): {final_columns}")

    if len(final_columns) == 0 : # 피처 정의가 아예 안된 경우 방지
         error_message = (f"Feature column definition IS EMPTY for {market_name_upper}.")
         raise ValueError(error_message)

    # # 주석 처리: 피처 개수 고정 검증은 실제 모델에 따라 유동적이므로, 일단 제거. 필요시 활성화.
    # # 현재는 동적으로 계산된 피처 수를 사용하므로, 이 검증은 항상 통과하게 됨.
    # # 만약 특정 개수로 고정해야 한다면, EXPECTED_FEATURE_COUNT를 상수로 정의하고 아래 주석 해제.
    # FIXED_EXPECTED_COUNT = 30 # 예시: 만약 항상 30개여야 한다면
    # if len(final_columns) != FIXED_EXPECTED_COUNT:
    #     error_message = (f"Feature column definition error for {market_name_upper}: "
    #                      f"{len(final_columns)} features defined, expected {FIXED_EXPECTED_COUNT}. Features: {final_columns}")
    #     # print(f"[ERROR][get_feature_columns_for_market] {error_message}") # 로깅
    #     raise ValueError(error_message) 
    
    return final_columns

# --- 나머지 전역 변수 및 함수들은 기존과 거의 동일하게 유지 ---
TIME_STEPS = 10 
FUTURE_TARGET_DAYS = 5 
MIN_DATA_DAYS_FOR_PREDICT = 150 # 예측을 위해 필요한 최소 과거 데이터 일수 (TA 계산 등 고려)

models_cache = {}
scalers_X_cache = {}
scalers_y_cache = {}

def load_model_and_scalers(model_key_name):
    """지정된 모델 키에 해당하는 모델과 스케일러를 로드하고 캐시합니다."""
    if model_key_name in models_cache:
        # print(f"[INFO][load_model_and_scalers] Using cached model/scalers for {model_key_name}.")
        return models_cache[model_key_name], scalers_X_cache[model_key_name], scalers_y_cache[model_key_name]

    market_type_from_key = model_key_name.split("_")[0].lower() # 'kospi' or 'kosdaq'
    analysis_type_from_key = "_".join(model_key_name.split("_")[1:]) # 'technical' or 'comprehensive' 등

    # 파일명 규칙: {market}_{analysis_type}_model.keras, {market}_{analysis_type}_scaler_X.joblib 등
    model_file = f"{market_type_from_key}_{analysis_type_from_key}_model.keras"
    scaler_x_file = f"{market_type_from_key}_{analysis_type_from_key}_scaler_X.joblib"
    scaler_y_file = f"{market_type_from_key}_{analysis_type_from_key}_scaler_y.joblib"

    model_path = os.path.join(ML_MODELS_DIR, model_file)
    scaler_x_path = os.path.join(ML_MODELS_DIR, scaler_x_file)
    scaler_y_path = os.path.join(ML_MODELS_DIR, scaler_y_file)

    # print(f"[DEBUG][load_model_and_scalers] Attempting to load for key '{model_key_name}':")
    # print(f"  Model: {model_path}")
    # print(f"  Scaler X: {scaler_x_path}")
    # print(f"  Scaler Y: {scaler_y_path}")

    loaded_model = None
    loaded_scaler_X = None
    loaded_scaler_y = None
    
    try:
        import tensorflow as tf 
        import joblib

        if os.path.exists(model_path):
            try:
                loaded_model = tf.keras.models.load_model(model_path)
                # print(f"[INFO][load_model_and_scalers] Successfully loaded model: {model_file}")
            except Exception as e_model:
                print(f"[ERROR][load_model_and_scalers] Error loading model file {model_file}: {e_model}")
        else:
            print(f"[ERROR][load_model_and_scalers] Model file not found: {model_path}")

        if os.path.exists(scaler_x_path):
            try:
                loaded_scaler_X = joblib.load(scaler_x_path)
                # print(f"[INFO][load_model_and_scalers] Successfully loaded scaler X: {scaler_x_file}")
            except Exception as e_scaler_x:
                print(f"[ERROR][load_model_and_scalers] Error loading scaler X file {scaler_x_file}: {e_scaler_x}")
        else:
            print(f"[ERROR][load_model_and_scalers] Scaler X file not found: {scaler_x_path}")

        if os.path.exists(scaler_y_path):
            try:
                loaded_scaler_y = joblib.load(scaler_y_path)
                # print(f"[INFO][load_model_and_scalers] Successfully loaded scaler Y: {scaler_y_file}")
            except Exception as e_scaler_y:
                print(f"[ERROR][load_model_and_scalers] Error loading scaler Y file {scaler_y_file}: {e_scaler_y}")
        else:
            print(f"[ERROR][load_model_and_scalers] Scaler Y file not found: {scaler_y_path}")
        
        if not all([loaded_model, loaded_scaler_X, loaded_scaler_y]):
            print(f"[ERROR][load_model_and_scalers] One or more components failed to load for {model_key_name}.")
            return None, None, None # 하나라도 실패하면 모두 None 반환
            
        models_cache[model_key_name] = loaded_model
        scalers_X_cache[model_key_name] = loaded_scaler_X
        scalers_y_cache[model_key_name] = loaded_scaler_y
        # print(f"[INFO][load_model_and_scalers] Loaded and cached {model_key_name} model and scalers.")
        return loaded_model, loaded_scaler_X, loaded_scaler_y

    except ImportError as ie:
        print(f"[CRITICAL ERROR][load_model_and_scalers] Import error for tensorflow or joblib: {ie}. Please ensure they are installed.")
        traceback.print_exc()
        return None, None, None
    except Exception as e:
        print(f"[CRITICAL ERROR][load_model_and_scalers] An unexpected exception occurred during loading components for {model_key_name}: {e}")
        traceback.print_exc()
        return None, None, None

# 앱 시작 시 주요 모델 미리 로드 (선택 사항, 최초 요청 시 지연 감소 효과)
# 실제 운영 환경에서는 apps.py의 ready() 메소드에서 호출하는 것을 고려할 수 있습니다.
# def preload_main_models():
# print("[INFO][views.py] Preloading main KOSPI/KOSDAQ technical models...")
# load_model_and_scalers("kospi_technical")
# load_model_and_scalers("kosdaq_technical")
# print("[INFO][views.py] Main models preloading attempt finished.")
# preload_main_models() # 개발 중에는 주석 처리 가능


def get_krx_stock_list_predict_cached():
    """FDR을 통해 KRX 전체 종목 목록(종목명, 코드, 시장구분)을 가져와 캐시합니다."""
    cache_key = 'krx_stock_list_predict_app_v6_with_market_standardized' # 캐시 키 버전 관리
    cached_list = cache.get(cache_key)
    if cached_list is not None:
        # print(f"[DEBUG][get_krx_stock_list_predict_cached] Using cached stock list (found {len(cached_list)} items).")
        return cached_list
    
    # print("[DEBUG][get_krx_stock_list_predict_cached] Cache miss. Fetching new stock list from FDR.")
    try:
        df_krx_fdr = fdr.StockListing('KRX') 
        if df_krx_fdr.empty:
            # print("[WARNING][get_krx_stock_list_predict_cached] FDR StockListing('KRX') returned empty DataFrame.")
            cache.set(cache_key, [], timeout=60*5) # 짧은 시간 캐시 후 재시도 유도
            return []

        # 컬럼명 일관성 처리 ('Symbol' 또는 'Code')
        code_col = 'Symbol' if 'Symbol' in df_krx_fdr.columns else 'Code'
        if 'Name' not in df_krx_fdr.columns or code_col not in df_krx_fdr.columns or 'Market' not in df_krx_fdr.columns: 
            # print(f"[ERROR][get_krx_stock_list_predict_cached] Required columns (Name, {code_col}, Market) not found in FDR result.")
            cache.set(cache_key, [], timeout=60*5)
            return []

        # 필요한 컬럼만 선택 및 결측치 처리, 중복 제거
        df_krx_fdr = df_krx_fdr[['Name', code_col, 'Market']].dropna(subset=['Name', code_col, 'Market'])
        df_krx_fdr = df_krx_fdr.drop_duplicates(subset=[code_col]) 
        
        stock_list = []
        for _, r in df_krx_fdr.iterrows():
            market_name = str(r['Market']).strip().upper()
            # 시장명 표준화 (예: "KOSDAQ GLOBAL" -> "KOSDAQ")
            if "KOSDAQ GLOBAL" in market_name: standardized_market = "KOSDAQ"
            elif "KONEX" in market_name: standardized_market = "KONEX" # 코넥스도 별도 처리
            elif market_name in ["KOSPI", "KOSDAQ"]: standardized_market = market_name
            else: standardized_market = "OTHER" # 기타 시장 (예측 대상에서 제외될 수 있음)
            
            stock_list.append({
                'name': str(r['Name']).strip(), 
                'code': str(r[code_col]).strip(), 
                'market_original': market_name, # FDR 원본 시장명
                'market_standardized': standardized_market # 예측 로직에서 사용할 표준화된 시장명
            })
        
        # print(f"[INFO][get_krx_stock_list_predict_cached] Successfully fetched and processed {len(stock_list)} stock items from FDR.")
        cache.set(cache_key, stock_list, timeout=60*60*12) # 12시간 캐시
        return stock_list
    except Exception as e:
        print(f"[ERROR][get_krx_stock_list_predict_cached] KRX stock list fetch/processing error: {e}")
        traceback.print_exc()
        cache.set(cache_key, [], timeout=60*5)
        return []

def get_stock_info_for_predict(stock_input_query):
    """입력된 종목명 또는 코드를 기반으로 표준화된 시장 정보, 코드, 이름 등을 반환합니다."""
    list_of_stock_dicts = get_krx_stock_list_predict_cached()
    if not list_of_stock_dicts:
        # print(f"[WARNING][get_stock_info_for_predict] Stock list is empty. Cannot find info for '{stock_input_query}'.")
        return None, None, None, None 
    
    found_stock_dict = None
    processed_input = stock_input_query.strip()
    processed_input_upper = processed_input.upper()

    # 6자리 숫자인 경우 코드로 먼저 검색
    if processed_input.isdigit() and len(processed_input) == 6:
        found_stock_dict = next((s for s in list_of_stock_dicts if s.get('code') == processed_input), None)
    
    # 코드로 못 찾았거나, 코드가 아닌 경우 이름으로 검색
    if not found_stock_dict:
        found_stock_dict = next((s for s in list_of_stock_dicts if s.get('name') and s.get('name').strip().upper() == processed_input_upper), None)
        # 부분 일치 검색 (필요시 추가)
        # if not found_stock_dict:
        #     found_stock_dict = next((s for s in list_of_stock_dicts if s.get('name') and processed_input_upper in s.get('name').strip().upper()), None)

    if found_stock_dict:
        standardized_market = found_stock_dict.get('market_standardized')
        original_market_name = found_stock_dict.get('market_original')
        stock_code = found_stock_dict.get('code')
        stock_name = found_stock_dict.get('name')

        if not standardized_market or not stock_code or not stock_name: # 필수 정보 누락 체크
            # print(f"[WARNING][get_stock_info_for_predict] Found stock dict for '{stock_input_query}' but essential info missing: {found_stock_dict}")
            return None, None, None, None
            
        # print(f"[DEBUG][get_stock_info_for_predict] Found for '{stock_input_query}': Code={stock_code}, Name={stock_name}, StandardMarket={standardized_market}, OriginalMarket={original_market_name}")
        return standardized_market, stock_code, stock_name, original_market_name
    
    # print(f"[INFO][get_stock_info_for_predict] No stock info found for query: '{stock_input_query}'")
    return None, None, None, None


def get_future_trading_dates_list(start_date_input, num_days):
    """주어진 시작일 다음 거래일부터 향후 num_days 만큼의 거래일 목록을 반환합니다."""
    from pandas.tseries.offsets import BDay # 함수 내에서 임포트

    if not isinstance(start_date_input, date_type):
        try: start_date_input = pd.to_datetime(start_date_input).date()
        except: start_date_input = datetime.now().date() 
    
    # 휴일 계산 범위를 좀 더 동적으로 설정 (예: 시작 연도부터 +2년)
    # 또는 매년 휴일 데이터를 업데이트하는 별도 로직 필요
    kr_holidays_years = list(set([start_date_input.year + i for i in range(3)])) # 시작년도, +1년, +2년
    kr_holidays = holidays.KR(years=kr_holidays_years)
    
    future_dates = []
    # 시작일 다음 날부터 계산 시작
    current_date_pd = pd.Timestamp(start_date_input) + BDay(1) 
    
    # current_date_pd가 주말이거나 휴일이면 다음 거래일로 이동
    while current_date_pd.weekday() >= 5 or current_date_pd.date() in kr_holidays:
        current_date_pd += BDay(1)
        
    while len(future_dates) < num_days:
        future_dates.append(current_date_pd.date())
        current_date_pd += BDay(1) # 다음 날로 이동
        # 다시 주말/휴일 체크
        while current_date_pd.weekday() >= 5 or current_date_pd.date() in kr_holidays:
            current_date_pd += BDay(1)
            
    return future_dates


def predict_info_view(request):
    """주가 예측 페이지를 렌더링합니다. URL 파라미터로 초기 검색어 처리."""
    context = {
        'stock_name_for_display': '', 
        'ticker': '', 
        'error_message': None, 
        'is_favorite': False, 
        'market_name': '' # FDR 원본 시장명 (표시용)
    }
    initial_query = request.GET.get('stock_query', '').strip()

    if initial_query:
        # get_stock_info_for_predict는 표준화된 시장명, 코드, 이름, 원본 시장명을 반환
        std_market, code, name, original_market = get_stock_info_for_predict(initial_query) 
        if code and name: 
            context['stock_name_for_display'] = name
            context['ticker'] = code
            context['market_name'] = original_market # 표시용은 원본 시장명 사용
            if request.user.is_authenticated:
                context['is_favorite'] = FavoriteStock.objects.filter(user=request.user, stock_code=code).exists()
        else: 
            context['stock_name_for_display'] = initial_query # 검색 실패 시 입력값 그대로 표시
            context['error_message'] = f"'{initial_query}'에 대한 종목 정보를 찾을 수 없습니다. 정확한 종목명 또는 6자리 코드를 입력해주세요."
            
    return render(request, 'predict_info/predict_info.html', context)


def predict_stock_price_ajax(request):
    """
    AJAX 요청을 통해 특정 종목의 예측 가격을 DB에서 조회하여 반환합니다.
    이 함수는 직접 예측을 수행하지 않고, generate_daily_predictions.py에 의해 미리 계산된 결과를 사용합니다.
    """
    if request.method == 'POST':
        stock_input = request.POST.get('stock_input', '').strip()
        analysis_type_req = request.POST.get('analysis_type', 'technical').strip().lower()

        if not stock_input: 
            return JsonResponse({'error': '종목명 또는 종목코드를 입력해주세요.'}, status=400)

        # 입력값을 바탕으로 표준화된 시장 정보, 코드, 이름, 원본 시장명 조회
        standardized_market, stock_code, stock_name, original_market_name_for_display = get_stock_info_for_predict(stock_input)
        
        if not standardized_market or not stock_code or not stock_name: 
            return JsonResponse({'error': f"'{stock_input}'에 해당하는 종목 정보를 찾을 수 없습니다. 자동완성 기능을 이용하거나 정확한 정보를 입력해주세요."}, status=400)

        # 현재는 KOSPI, KOSDAQ만 지원 (모델이 준비된 시장)
        if standardized_market not in ["KOSPI", "KOSDAQ"]: 
            return JsonResponse({'error': f"'{stock_name}'({original_market_name_for_display}) 시장은 현재 예측을 지원하지 않습니다. (지원 시장: KOSPI, KOSDAQ)"}, status=400)

        # DB에서 예측 결과 조회
        try:
            # 가장 최근 예측 기준일의 예측 결과 가져오기
            latest_prediction_entry = PredictedStockPrice.objects.filter(
                stock_code=stock_code,
                analysis_type=analysis_type_req, # 요청된 분석 유형 사용
                market_name=standardized_market.upper() # DB에는 대문자로 저장되어 있을 것이므로
            ).order_by('-prediction_base_date').first()

            if not latest_prediction_entry:
                return JsonResponse({'error': f"'{stock_name}'에 대한 '{analysis_type_req}' 분석 예측 결과를 찾을 수 없습니다. (데이터 업데이트 전이거나 해당 종목/분석 유형의 예측이 생성되지 않았을 수 있습니다.)"}, status=404)

            prediction_base_date_from_db = latest_prediction_entry.prediction_base_date
            
            # 해당 기준일의 모든 예측일자 데이터 가져오기
            predictions_from_db = PredictedStockPrice.objects.filter(
                stock_code=stock_code,
                prediction_base_date=prediction_base_date_from_db,
                analysis_type=analysis_type_req,
                market_name=standardized_market.upper()
            ).order_by('predicted_date')

            if predictions_from_db.exists():
                predictions_output = [{'date': p.predicted_date.strftime('%Y-%m-%d'), 
                                       'price': round(float(p.predicted_price))} 
                                      for p in predictions_from_db]
                
                is_favorite_stock = False
                if request.user.is_authenticated:
                    is_favorite_stock = FavoriteStock.objects.filter(user=request.user, stock_code=stock_code).exists()

                return JsonResponse({
                    'stock_code': stock_code, 
                    'stock_name': stock_name, 
                    'market_name': original_market_name_for_display, # 표시용은 원본 시장명
                    'analysis_type': analysis_type_req, 
                    'predictions': predictions_output,
                    'last_data_date': prediction_base_date_from_db.strftime('%Y-%m-%d'), 
                    'is_favorite': is_favorite_stock, 
                    'data_source': 'database_prediction', # 데이터 출처 명시
                    'is_authenticated': request.user.is_authenticated 
                })
            else:
                # 이 경우는 latest_prediction_entry는 있었는데, filter 결과가 없는 이상한 상황
                print(f"[ERROR][predict_stock_price_ajax] Inconsistency for {stock_name}({stock_code}). Base entry found for {prediction_base_date_from_db}, but no series predictions.")
                return JsonResponse({'error': f"'{stock_name}'에 대한 예측 데이터 구성에 문제가 있습니다. 관리자에게 문의해주세요."}, status=500)
        
        except Exception as e_db_lookup:
            print(f"[ERROR][predict_stock_price_ajax] DB 조회 중 예외 발생 ({stock_name}, {analysis_type_req}): {e_db_lookup}")
            traceback.print_exc()
            return JsonResponse({'error': f"'{stock_name}'의 예측 결과를 가져오는 중 서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요."}, status=500)
            
    return JsonResponse({'error': '잘못된 요청입니다 (POST 요청 필요).'}, status=400)


def search_stocks_ajax(request):
    """종목 검색 자동완성을 위한 AJAX 핸들러."""
    term = request.GET.get('term', '').strip()
    limit = int(request.GET.get('limit', 7)) 
    
    if not term: 
        return JsonResponse([], safe=False)
        
    all_stocks_list = get_krx_stock_list_predict_cached()
    if not all_stocks_list: 
        # print("[WARNING][search_stocks_ajax] Stock list for autocomplete is empty.")
        return JsonResponse({'error': '종목 목록을 불러오는 데 실패했습니다. 잠시 후 다시 시도해주세요.'}, status=500)
        
    results = []
    term_upper = term.upper() 
    
    for item in all_stocks_list:
        stock_name_val = item.get('name','')
        stock_code_val = item.get('code','')
        # market_val = item.get('market_standardized','') # 표준화된 시장명
        market_display_val = item.get('market_original', item.get('market_standardized', '')) # 표시용은 원본, 없으면 표준

        # 코넥스 및 기타 시장은 자동완성에서 제외 (선택적)
        if item.get('market_standardized') not in ['KOSPI', 'KOSDAQ']:
            continue

        if term_upper in stock_name_val.upper() or term_upper in stock_code_val:
            results.append({
                'label': f"{stock_name_val} ({stock_code_val}) - {market_display_val}", 
                'value': stock_name_val, # 자동완성 선택 시 입력창에 채워질 값
                'code': stock_code_val, 
                'market': market_display_val # 선택 시 활용할 시장 정보 (원본)
            })
        if len(results) >= limit: 
            break
            
    return JsonResponse(results, safe=False)


@login_required 
def toggle_favorite_stock_ajax(request):
    """관심 종목 추가/삭제를 처리하는 AJAX 핸들러."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            stock_code = data.get('stock_code')
            stock_name = data.get('stock_name')
            # market_name은 FDR 원본 시장명을 받도록 HTML에서 수정됨
            market_name_from_frontend = data.get('market_name') 
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': '잘못된 JSON 형식입니다.'}, status=400)

        if not all([stock_code, stock_name, market_name_from_frontend]):
            return JsonResponse({'status': 'error', 'message': '종목 코드, 이름, 시장 정보가 모두 필요합니다.'}, status=400)

        user = request.user
        MAX_FAVORITES = getattr(settings, 'MAX_FAVORITE_STOCKS', 10) # 설정에서 최대 개수 가져오기

        try:
            favorite_obj, created = FavoriteStock.objects.get_or_create(
                user=user,
                stock_code=stock_code,
                defaults={
                    'stock_name': stock_name, 
                    'market_name': market_name_from_frontend # 프론트에서 받은 시장명 저장
                }
            )

            if created: 
                if FavoriteStock.objects.filter(user=user).count() > MAX_FAVORITES:
                    favorite_obj.delete() # 한도 초과 시 방금 만든 것 삭제
                    return JsonResponse({
                        'status': 'error',
                        'message': f'관심 종목은 최대 {MAX_FAVORITES}개까지 추가할 수 있습니다.',
                        'is_favorite': False # 실제로는 추가 안됨
                    })
                return JsonResponse({
                    'status': 'success', 
                    'message': f"'{stock_name}'을(를) 관심 종목에 추가했습니다.",
                    'is_favorite': True
                })
            else: # 이미 존재하면 삭제 (토글)
                favorite_obj.delete()
                return JsonResponse({
                    'status': 'success',
                    'message': f"'{stock_name}'을(를) 관심 종목에서 삭제했습니다.",
                    'is_favorite': False
                })
        except Exception as e:
            print(f"[ERROR][toggle_favorite_stock_ajax] 관심종목 처리 중 오류 ({user.username}, {stock_code}): {e}")
            traceback.print_exc()
            return JsonResponse({'status': 'error', 'message': '관심 종목 처리 중 오류가 발생했습니다. 다시 시도해주세요.'}, status=500)

    return JsonResponse({'status': 'error', 'message': '잘못된 요청입니다 (POST 방식 필요).'}, status=400)
