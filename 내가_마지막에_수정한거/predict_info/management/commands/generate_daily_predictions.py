# predict_info/management/commands/generate_daily_predictions.py
import time
import traceback
from datetime import datetime, timedelta, date as date_type

import FinanceDataReader as fdr # 직접 사용은 줄었지만, get_market_macro_data 등에서 간접 사용
import numpy as np
import pandas as pd
# import tensorflow as tf # load_model_and_scalers 함수에서 임포트
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone
# from pandas.tseries.offsets import BDay # get_future_trading_dates_list에서 사용
# import holidays # get_future_trading_dates_list에서 사용

from predict_info.models import PredictedStockPrice, StockPrice
from predict_info.utils import calculate_all_features, get_market_macro_data, PANDAS_TA_AVAILABLE
from predict_info.views import ( 
    ML_MODELS_DIR, TIME_STEPS, FUTURE_TARGET_DAYS, MIN_DATA_DAYS_FOR_PREDICT,
    get_feature_columns_for_market, load_model_and_scalers,
    get_krx_stock_list_predict_cached, get_future_trading_dates_list
) # views.py의 상수 및 함수 재활용

DAYS_TO_KEEP_PREDICTIONS_DEFAULT = 14 # DB에 예측 결과를 보관할 기본 일수

# --- DB 필드명 (StockPrice 모델 기준) ---
# 이 목록은 StockPrice 모델의 실제 필드명과 일치해야 합니다.
BASE_OHLCV_COLS_FROM_DB = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
INVESTOR_COLS_FROM_DB = ['indi_volume', 'foreign_volume', 'organ_volume'] 
FUNDAMENTAL_COLS_FROM_DB = ['market_cap', 'per', 'pbr'] # EPS, BPS 등 추가 펀더멘털도 필요시 추가
# USER_CUSTOM_FUNDAMENTAL_COLS_FROM_DB = ['eps', 'bps', 'dps', 'roe'] # 예시

# --- Pandas DataFrame 및 모델 입력 시 사용할 표준 컬럼명 ---
# 이 목록은 views.py의 get_feature_columns_for_market 함수가 반환하는 목록의 일부와 일치해야 합니다.
BASE_OHLCV_COLS_FOR_DF = ['Open', 'High', 'Low', 'Close', 'Volume'] # 첫 글자 대문자
INVESTOR_COLS_FOR_MODEL = ['Indi', 'Foreign', 'Organ']
FUNDAMENTAL_COLS_FOR_MODEL = ['Marcap', 'PER', 'PBR'] # EPS, BPS 등 추가 펀더멘털도 필요시 추가
# USER_CUSTOM_FUNDAMENTAL_COLS_FOR_MODEL = ['EPS', 'BPS', 'DPS', 'ROE'] # 예시


class Command(BaseCommand):
    help = ('Generates daily stock price predictions for KOSPI/KOSDAQ stocks using pre-trained models '
            'based on data from the StockPrice DB table, and saves predictions to PredictedStockPrice table.')

    def add_arguments(self, parser):
        parser.add_argument(
            '--markets', 
            type=str, 
            default='KOSPI,KOSDAQ', 
            help='Comma-separated list of markets to process (e.g., KOSPI,KOSDAQ).'
        )
        parser.add_argument(
            '--analysis_type', 
            type=str, 
            default='technical', # 기본 분석 유형 (모델 파일명과 연관)
            help='Analysis model type to use for prediction (e.g., technical, comprehensive).'
        )
        parser.add_argument(
            '--delete_old_after_days', 
            type=int, 
            default=DAYS_TO_KEEP_PREDICTIONS_DEFAULT, 
            help=f'Days to keep old predictions in DB. Default: {DAYS_TO_KEEP_PREDICTIONS_DEFAULT}. Set to 0 to disable deletion.'
        )

    def _get_stock_data_for_prediction(self, stock_code, stock_name, market_name_upper, feature_names_for_model_input_list):
        """
        DB(StockPrice)에서 특정 종목의 데이터를 가져와 전처리 후 모델 입력용 피처 DataFrame을 반환합니다.
        market_name_upper: 'KOSPI' 또는 'KOSDAQ'
        feature_names_for_model_input_list: 해당 시장의 모델이 학습한 피처 컬럼명 리스트
        """
        self.stdout.write(f"  [DATA_PREP] Preparing data for prediction: {stock_name}({stock_code}) - Market: {market_name_upper}")
        
        if not PANDAS_TA_AVAILABLE: # pandas-ta는 calculate_all_features 내부에서 사용됨
            self.stderr.write(f"    [CRITICAL_ERROR] pandas_ta library unavailable. Cannot generate features for {stock_name}. Skipping.")
            return None, None # (DataFrame, prediction_base_date)
        
        try:
            # 예측 실행일 기준 (오늘)
            prediction_execution_date = timezone.now().date() 
            
            # DB에서 데이터를 가져올 시작 날짜 (TA 계산 및 시계열 구성에 충분한 기간)
            # MIN_DATA_DAYS_FOR_PREDICT (예: 150일) + TIME_STEPS (예: 10일) + TA 계산용 버퍼 (예: 90일)
            required_data_start_date_in_db = prediction_execution_date - timedelta(days=MIN_DATA_DAYS_FOR_PREDICT + TIME_STEPS + 90) 
            
            self.stdout.write(f"    Fetching data from StockPrice DB for {stock_code} from {required_data_start_date_in_db} to {prediction_execution_date}")

            # DB에서 가져올 컬럼 목록 (StockPrice 모델 필드명과 일치)
            db_columns_to_fetch_list = ['date'] + BASE_OHLCV_COLS_FROM_DB + \
                                       INVESTOR_COLS_FROM_DB + FUNDAMENTAL_COLS_FROM_DB
                                       # + USER_CUSTOM_FUNDAMENTAL_COLS_FROM_DB (필요시)
            
            stock_prices_queryset = StockPrice.objects.filter(
                stock_code=stock_code,
                date__gte=required_data_start_date_in_db,
                date__lte=prediction_execution_date # 오늘까지의 데이터 포함 (실제로는 어제 거래일까지가 최신일 것)
            ).order_by('date').values(*db_columns_to_fetch_list)

            if not stock_prices_queryset.exists():
                self.stderr.write(f"    [WARNING] No historical data in StockPrice DB for {stock_name}({stock_code}) in the required range. Skipping.")
                return None, None
            
            df_from_db_raw = pd.DataFrame.from_records(stock_prices_queryset)
            self.stdout.write(f"    Fetched {len(df_from_db_raw)} records from DB for {stock_code}.")

            # 예측에 필요한 최소 데이터 길이 체크 (TA 계산 및 TIME_STEPS 구성 고려)
            if len(df_from_db_raw) < (TIME_STEPS + 60): # 최소 60일치 데이터는 있어야 TA 계산이 의미있음 (MACD 등)
                self.stderr.write(f"    [WARNING] Insufficient DB data for {stock_name}({stock_code}) for robust TA calculation. Found {len(df_from_db_raw)}, need at least {TIME_STEPS + 60}. Skipping.")
                return None, None

            # Date 컬럼명 변경 및 타입 변환, 인덱스 설정
            df_from_db_raw.rename(columns={'date': 'Date'}, inplace=True)
            df_from_db_raw['Date'] = pd.to_datetime(df_from_db_raw['Date'])
            df_from_db_raw.set_index('Date', inplace=True)

            # DB 컬럼명(소문자) -> Pandas DataFrame 및 모델 입력용 표준 컬럼명(대문자 시작)으로 변경
            db_to_df_col_map = {}
            for db_col, df_col in zip(BASE_OHLCV_COLS_FROM_DB, BASE_OHLCV_COLS_FOR_DF): db_to_df_col_map[db_col] = df_col
            for db_col, model_col in zip(INVESTOR_COLS_FROM_DB, INVESTOR_COLS_FOR_MODEL): db_to_df_col_map[db_col] = model_col
            for db_col, model_col in zip(FUNDAMENTAL_COLS_FROM_DB, FUNDAMENTAL_COLS_FOR_MODEL): db_to_df_col_map[db_col] = model_col
            # for db_col, model_col in zip(USER_CUSTOM_FUNDAMENTAL_COLS_FROM_DB, USER_CUSTOM_FUNDAMENTAL_COLS_FOR_MODEL): db_to_df_col_map[db_col] = model_col # 예시

            df_renamed_cols = df_from_db_raw.rename(columns=db_to_df_col_map)

            # OHLCV 데이터 숫자형 변환 및 'Change' 컬럼 계산
            for col in BASE_OHLCV_COLS_FOR_DF: 
                if col not in df_renamed_cols.columns:
                    self.stderr.write(f"    [CRITICAL_ERROR] Column '{col}' missing after renaming from DB data for {stock_name}. Skipping.")
                    return None, None
                df_renamed_cols[col] = pd.to_numeric(df_renamed_cols[col], errors='coerce')
            
            if 'Close' in df_renamed_cols.columns:
                df_renamed_cols['Change'] = df_renamed_cols['Close'].pct_change()
            else: df_renamed_cols['Change'] = np.nan # 'Close'가 없으면 'Change'도 NaN

            # 투자자 및 펀더멘털 데이터 숫자형 변환 및 NaN 처리 (0.0으로)
            cols_to_convert_and_fillna = INVESTOR_COLS_FOR_MODEL + FUNDAMENTAL_COLS_FOR_MODEL # + USER_CUSTOM_FUNDAMENTAL_COLS_FOR_MODEL
            for col_name in cols_to_convert_and_fillna:
                if col_name in df_renamed_cols.columns:
                    df_renamed_cols[col_name] = pd.to_numeric(df_renamed_cols[col_name], errors='coerce').fillna(0.0)
                else: 
                    # self.stdout.write(f"    [DATA_PREP] Column '{col_name}' not found after renaming, initializing to 0.0 for {stock_code}.")
                    df_renamed_cols[col_name] = 0.0 # 모델 입력에 필요하면 0으로라도 채움
            
            # 실제 데이터의 마지막 날짜 (이 날짜까지의 데이터를 기반으로 예측)
            actual_prediction_base_date_val = df_renamed_cols.index.max().date()
            self.stdout.write(f"    Actual prediction base date for {stock_code} is {actual_prediction_base_date_val}.")
            
            # 거시경제 및 시장 지수 데이터 추가
            fetch_start_date_for_others = df_renamed_cols.index.min().date() # DB에서 가져온 데이터의 가장 이른 날짜
            fetch_end_date_for_others = actual_prediction_base_date_val # DB에서 가져온 데이터의 가장 늦은 날짜
            
            market_fdr_code_for_index = 'KS11' if market_name_upper == 'KOSPI' else 'KQ11'
            df_market_idx_data, df_macro_fx_data = get_market_macro_data(
                fetch_start_date_for_others, fetch_end_date_for_others, 
                market_fdr_code=market_fdr_code_for_index
            )
            
            df_merged_for_ta = df_renamed_cols.copy() 
            
            market_index_cols_expected = [col for col in feature_names_for_model_input_list if market_name_upper in col and ("_Close" in col or "_Change" in col)]
            if not df_market_idx_data.empty: 
                df_merged_for_ta = df_merged_for_ta.join(df_market_idx_data, how='left')
            else: # 시장 지수 데이터 못가져오면 해당 컬럼 NaN으로 채움
                for col in market_index_cols_expected: df_merged_for_ta[col] = np.nan
            
            macro_cols_expected_in_features = [col for col in feature_names_for_model_input_list if "USD_KRW" in col]
            if not df_macro_fx_data.empty: 
                df_merged_for_ta = df_merged_for_ta.join(df_macro_fx_data, how='left')
            else: # 환율 데이터 못가져오면 해당 컬럼 NaN으로 채움
                for col in macro_cols_expected_in_features: df_merged_for_ta[col] = np.nan
            
            # 외부 데이터 join 후 ffill (TA 계산 전)
            cols_to_ffill_after_join = market_index_cols_expected + macro_cols_expected_in_features
            df_merged_for_ta[cols_to_ffill_after_join] = df_merged_for_ta[cols_to_ffill_after_join].ffill()

            # 기술적 지표 계산 (utils.py의 함수 사용)
            df_with_all_features_calc = calculate_all_features(df_merged_for_ta.copy(), market_name_upper=market_name_upper)
            self.stdout.write(f"    Calculated Technical Indicators for {stock_code}. DataFrame shape: {df_with_all_features_calc.shape}")

            # 모델 입력에 필요한 모든 피처가 있는지 확인하고, 없으면 NaN으로 채움
            missing_features_in_df = []
            for col_model_needs in feature_names_for_model_input_list: 
                if col_model_needs not in df_with_all_features_calc.columns:
                    self.stderr.write(f"      [CRITICAL_ERROR] Feature '{col_model_needs}' MISSING for {stock_name} after TA calculation. Adding as NaN.")
                    df_with_all_features_calc[col_model_needs] = np.nan 
                    missing_features_in_df.append(col_model_needs)
            
            if missing_features_in_df: # 중요한 피처가 누락되면 예측 정확도에 문제
                 self.stderr.write(f"    [CRITICAL_ERROR] Critical features missing for {stock_name} before final selection: {missing_features_in_df}. Predictions might be unreliable.")

            # 최종적으로 모델 입력에 사용할 피처만 선택
            df_final_features_for_model = df_with_all_features_calc[feature_names_for_model_input_list].copy()
            
            # 최종 ffill -> bfill 로 NaN 처리 (TA 계산 후 발생한 앞부분 NaN 등)
            df_final_features_for_model.ffill(inplace=True).bfill(inplace=True) 
            
            # 그래도 NaN이 남아있다면 0.0으로 채움 (최후의 수단)
            nan_cols_final_check = df_final_features_for_model.columns[df_final_features_for_model.isnull().any()].tolist()
            if nan_cols_final_check:
                self.stderr.write(f"    [WARNING] NaN values found in final features for {stock_name} even after ffill/bfill: {nan_cols_final_check}. Filling these with 0.0.")
                df_final_features_for_model.fillna(0.0, inplace=True)
            
            # 최종 NaN 재확인 (이 시점에는 NaN이 없어야 함)
            if df_final_features_for_model.isnull().values.any():
                self.stderr.write(f"    [CRITICAL_ERROR] Unrecoverable NaN values persist in final features for {stock_name}. Cannot proceed with prediction. Skipping.")
                return None, None

            # 시계열 데이터 구성 (최근 TIME_STEPS 만큼)
            if len(df_final_features_for_model) < TIME_STEPS:
                self.stderr.write(f"    [ERROR] Insufficient final feature data rows for {stock_name} to form time-series. Found {len(df_final_features_for_model)}, need {TIME_STEPS}. Skipping.")
                return None, None

            recent_features_df_for_predict = df_final_features_for_model.tail(TIME_STEPS)
            self.stdout.write(f"    Successfully prepared final features for {stock_name}. Shape for model: {recent_features_df_for_predict.shape}. Prediction base date: {actual_prediction_base_date_val}")
            return recent_features_df_for_predict, actual_prediction_base_date_val
            
        except Exception as e_data_prep:
            self.stderr.write(f"    [CRITICAL_ERROR] Exception during data preparation for {stock_name}({stock_code}): {e_data_prep}")
            self.stderr.write(traceback.format_exc())
            return None, None


    def handle(self, *args, **options):
        start_time_script = time.time()
        self.stdout.write(self.style.SUCCESS(f"Starting daily prediction generation command at {timezone.now()}..."))

        markets_to_process_input = [m.strip().upper() for m in options['markets'].split(',') if m.strip()]
        analysis_type_for_model = options['analysis_type'].lower() # 예: 'technical'
        delete_predictions_older_than_days = options['delete_old_after_days']

        if not PANDAS_TA_AVAILABLE: # utils.py에서 이 변수를 설정
            self.stderr.write(self.style.ERROR("pandas_ta library is not available. Predictions cannot be generated as TA features are crucial."))
            return

        all_krx_stocks_list = get_krx_stock_list_predict_cached() # views.py의 함수 재활용
        if not all_krx_stocks_list:
            self.stderr.write(self.style.ERROR("Could not fetch KRX stock list. Aborting prediction generation."))
            return

        total_processed_ok_count, total_skipped_count, total_error_count = 0, 0, 0
        
        for market_name_to_run in markets_to_process_input:
            if market_name_to_run not in ["KOSPI", "KOSDAQ"]: # 지원하는 시장인지 확인
                self.stderr.write(self.style.WARNING(f"Market '{market_name_to_run}' is not supported for prediction. Skipping."))
                continue

            self.stdout.write(self.style.SUCCESS(f"\n--- Processing {market_name_to_run} market for '{analysis_type_for_model}' analysis ---"))
            
            # 해당 시장 및 분석 유형에 맞는 모델 키 생성 (views.py의 load_model_and_scalers와 일관성)
            model_key_for_loading = f"{market_name_to_run.lower()}_{analysis_type_for_model}"
            
            # 모델 및 스케일러 로드
            ml_model_instance, scaler_X_instance, scaler_y_instance = load_model_and_scalers(model_key_for_loading)

            if not all([ml_model_instance, scaler_X_instance, scaler_y_instance]):
                self.stderr.write(self.style.ERROR(f"Failed to load model/scalers for key '{model_key_for_loading}'. Skipping {market_name_to_run}."))
                total_error_count +=1 # 모델 로드 실패도 에러로 간주
                continue

            # 해당 시장의 모델 입력 피처 목록 가져오기
            try:
                feature_columns_for_this_market = get_feature_columns_for_market(market_name_to_run)
            except ValueError as e_feat_cols:
                self.stderr.write(self.style.ERROR(f"Error defining feature columns for {market_name_to_run}: {e_feat_cols}. Skipping market."))
                total_error_count +=1
                continue
            
            # 해당 시장의 종목만 필터링
            stocks_to_predict_in_this_market = [
                s for s in all_krx_stocks_list 
                if s.get('market_standardized', '').upper() == market_name_to_run # 표준화된 시장명 사용
            ]
            if not stocks_to_predict_in_this_market:
                self.stdout.write(f"No stocks found for market {market_name_to_run} in the cached list.")
                continue

            num_stocks_this_market = len(stocks_to_predict_in_this_market)
            self.stdout.write(f"Found {num_stocks_this_market} stocks in {market_name_to_run} to process.")

            for i, stock_info_dict in enumerate(stocks_to_predict_in_this_market):
                stock_code_val = stock_info_dict.get('code')
                stock_name_val = stock_info_dict.get('name')
                
                if not stock_code_val or not stock_name_val:
                    self.stderr.write(f"  Skipping stock with missing code/name: {stock_info_dict}"); total_skipped_count +=1; continue
                
                self.stdout.write(f"  ({i+1}/{num_stocks_this_market}) Processing: {stock_name_val} ({stock_code_val})")
                
                # 데이터 준비
                df_features_for_input, actual_base_date_for_pred = self._get_stock_data_for_prediction(
                    stock_code_val, stock_name_val, market_name_to_run, feature_columns_for_this_market
                )

                if df_features_for_input is None or actual_base_date_for_pred is None:
                    self.stderr.write(f"    Data preparation failed for {stock_name_val}. Skipping prediction for this stock."); total_skipped_count += 1; continue
                
                # 피처 개수 및 NaN 재확인 (데이터 준비 함수에서 이미 했지만, 한 번 더)
                if len(df_features_for_input.columns) != len(feature_columns_for_this_market):
                    self.stderr.write(f"    [CRITICAL_ERROR] Feature column count mismatch for {stock_name_val} before scaling. Expected {len(feature_columns_for_this_market)}, got {len(df_features_for_input.columns)}. Skipping.")
                    total_error_count +=1; continue
                if df_features_for_input.isnull().values.any():
                    nan_cols_debug = df_features_for_input.columns[df_features_for_input.isnull().any()].tolist()
                    self.stderr.write(f"    [CRITICAL_ERROR] NaN values detected in final features for {stock_name_val} just before scaling: {nan_cols_debug}. Skipping.")
                    total_error_count +=1; continue

                # 예측 수행
                try:
                    input_data_numpy = df_features_for_input[feature_columns_for_this_market].values # 순서 보장
                    input_data_scaled_np = scaler_X_instance.transform(input_data_numpy)
                    input_data_reshaped_for_model = input_data_scaled_np.reshape(1, TIME_STEPS, len(feature_columns_for_this_market))
                    
                    prediction_scaled_output = ml_model_instance.predict(input_data_reshaped_for_model, verbose=0) # Keras predict
                    
                    if prediction_scaled_output.shape != (1, FUTURE_TARGET_DAYS):
                         self.stderr.write(f"    [ERROR] Unexpected scaled prediction shape for {stock_name_val}: {prediction_scaled_output.shape}. Expected (1, {FUTURE_TARGET_DAYS}). Skipping.")
                         total_error_count += 1; continue
                    
                    # 스케일 역변환 (scaler_y는 타겟(종가)에 대해서만 학습되었을 것)
                    prediction_reshaped_for_inverse_scale = prediction_scaled_output.reshape(-1, 1) # (FUTURE_TARGET_DAYS, 1) 형태
                    actual_price_predictions_raw = scaler_y_instance.inverse_transform(prediction_reshaped_for_inverse_scale).flatten() # (FUTURE_TARGET_DAYS,) 형태
                    
                    # 예측 대상 미래 거래일 계산
                    future_target_trading_dates = get_future_trading_dates_list(actual_base_date_for_pred, FUTURE_TARGET_DAYS)

                    if len(future_target_trading_dates) != FUTURE_TARGET_DAYS:
                        self.stderr.write(f"    Error calculating future trading dates for {stock_name_val}. Expected {FUTURE_TARGET_DAYS}, got {len(future_target_trading_dates)}. Skipping.")
                        total_error_count += 1; continue
                    
                    # 모델이 로그 변환된 타겟으로 학습되었는지 여부 (모델명 등으로 판단 - 개선 필요)
                    # 예시: 모델 파일명에 "_logtarget" 등이 포함되어 있다면 True
                    model_name_or_path_heuristic = getattr(ml_model_instance, 'name', "") or getattr(ml_model_instance, 'filepath', "") 
                    model_used_log_transform = "_log_" in model_name_or_path_heuristic.lower() # 단순 휴리스틱

                    predictions_to_save_in_db = []
                    # 가격 클리핑을 위한 기준 가격 (예측 기준일의 실제 종가)
                    last_actual_close_price = df_features_for_input['Close'].iloc[-1] 
                    if pd.isna(last_actual_close_price) or last_actual_close_price <=0: # 기준 종가가 유효하지 않으면 클리핑 어려움
                        self.stderr.write(f"    [WARNING] Invalid last actual close price ({last_actual_close_price}) for {stock_name_val}. Clipping might be ineffective.")
                        # 이 경우, 클리핑을 건너뛰거나, 다른 기준 사용 필요. 일단은 그대로 진행.
                        last_actual_close_price = np.mean(actual_price_predictions_raw) if not pd.isna(np.mean(actual_price_predictions_raw)) else 1.0 # 임시 대체

                    current_base_price_for_clipping_iter = last_actual_close_price

                    for day_idx in range(FUTURE_TARGET_DAYS):
                        predicted_price_this_day_raw = actual_price_predictions_raw[day_idx]
                        
                        # 로그 변환된 모델이었다면 역변환 (np.expm1)
                        final_predicted_price_this_day = np.expm1(predicted_price_this_day_raw) if model_used_log_transform else predicted_price_this_day_raw
                        
                        # 가격 클리핑 (예: 하루 변동폭 +-30% 이내, 또는 이전 예측일 가격 기준)
                        # 여기서는 예측 기준일 종가 대비 +-30%로 클리핑 후, 다음날은 이전 예측일 가격 기준으로 클리핑
                        upper_limit = current_base_price_for_clipping_iter * 1.30 
                        lower_limit = current_base_price_for_clipping_iter * 0.70 
                        clipped_price_final = np.clip(final_predicted_price_this_day, lower_limit, upper_limit)
                        
                        predictions_to_save_in_db.append({
                            'stock_code': stock_code_val, 
                            'stock_name': stock_name_val, 
                            'market_name': market_name_to_run.upper(), # DB 저장은 대문자 시장명
                            'prediction_base_date': actual_base_date_for_pred, 
                            'predicted_date': future_target_trading_dates[day_idx],
                            'predicted_price': round(float(clipped_price_final)), # 소수점 제거
                            'analysis_type': analysis_type_for_model,
                        })
                        current_base_price_for_clipping_iter = clipped_price_final # 다음날 클리핑 기준은 현재 예측일 가격
                    
                    # DB 저장 전, 해당 (종목코드, 기준일, 분석유형)의 기존 예측 데이터 삭제
                    num_deleted, _ = PredictedStockPrice.objects.filter(
                        stock_code=stock_code_val, 
                        prediction_base_date=actual_base_date_for_pred, 
                        analysis_type=analysis_type_for_model,
                        market_name=market_name_to_run.upper()
                    ).delete()
                    if num_deleted > 0: 
                        self.stdout.write(f"    Deleted {num_deleted} existing predictions for {stock_name_val} (base_date: {actual_base_date_for_pred}, type: {analysis_type_for_model}).")
                    
                    # 새 예측 결과 bulk_create
                    PredictedStockPrice.objects.bulk_create([PredictedStockPrice(**data) for data in predictions_to_save_in_db])
                    self.stdout.write(self.style.SUCCESS(f"    Successfully saved {len(predictions_to_save_in_db)} days of predictions for {stock_name_val} based on data up to {actual_base_date_for_pred}."))
                    total_processed_ok_count += 1
                
                except Exception as e_pred_save:
                    self.stderr.write(self.style.ERROR(f"    Error during prediction or saving for {stock_name_val} ({stock_code_val}): {e_pred_save}"))
                    self.stderr.write(traceback.format_exc()); total_error_count += 1
                
                if (i+1) % 50 == 0: # 진행 상황 로그
                    self.stdout.write(f"  ... processed {i+1}/{num_stocks_this_market} stocks in {market_name_to_run} for prediction ...")
                time.sleep(0.03) # CPU 사용량 조절을 위한 약간의 딜레이 (선택적)

        self.stdout.write(self.style.SUCCESS(f"\n--- Daily Prediction Generation Summary ---"))
        self.stdout.write(f"Total stocks considered from KRX list: {len(all_krx_stocks_list)}")
        self.stdout.write(f"Successfully predicted and saved: {total_processed_ok_count} stocks")
        self.stdout.write(f"Skipped (e.g., insufficient data, data prep fail): {total_skipped_count} stocks")
        self.stdout.write(f"Errors (model load, feature mismatch, prediction fail, DB save fail): {total_error_count} stocks")

        # 오래된 예측 데이터 삭제
        if delete_predictions_older_than_days > 0:
            cutoff_date_for_deletion = timezone.now().date() - timedelta(days=delete_predictions_older_than_days)
            self.stdout.write(f"\nDeleting predictions with prediction_base_date older than {cutoff_date_for_deletion}...")
            try:
                deleted_info_dict = PredictedStockPrice.objects.filter(prediction_base_date__lt=cutoff_date_for_deletion).delete()
                # deleted_info_dict는 (삭제된 총 객체 수, 타입별 삭제된 객체 수 dict) 형태의 튜플
                self.stdout.write(self.style.SUCCESS(f"Successfully deleted {deleted_info_dict[0]} old prediction records (details: {deleted_info_dict[1]})."))
            except Exception as e_delete_old: 
                self.stderr.write(self.style.ERROR(f"Error deleting old predictions: {e_delete_old}"))
        else: 
            self.stdout.write(f"Old prediction deletion skipped as per --delete_old_after_days={delete_predictions_older_than_days}.")

        end_time_script = time.time()
        self.stdout.write(self.style.SUCCESS(f"Daily prediction generation command finished in {end_time_script - start_time_script:.2f} seconds."))

