# predict_Info/startup_tasks.py
import pandas as pd
import FinanceDataReader as fdr
from django.utils import timezone
from datetime import timedelta, datetime, date as date_type
import holidays
import time
import os
import glob
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model 
from tensorflow.keras.layers import Input, LSTM, Dense 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from .utils import calculate_manual_features 
from django.conf import settings

# --- 설정값 ---
FEATURE_COLUMNS_TRAINING = ['Open', 'High', 'Low', 'Close', 'Volume', 
                            'ATR', 'BB_Lower', 'BB_Mid', 'BB_Upper', 
                            'RSI', 'MACD', 'MACD_Hist', 'MACD_Signal']
TARGET_COLUMN_TRAINING = 'Close'
TIME_STEPS_TRAINING = 10
FUTURE_TARGET_DAYS_TRAINING = 5
EPOCHS_FOR_DAILY_RETRAIN = 3
APPROX_5YR_TRADING_DAYS = 5 * 252 

# --- Helper Functions (데이터 업데이트용) ---
def get_trading_day_before(date_reference, days_offset=1):
    if isinstance(date_reference, datetime):
        current_check_date = date_reference.date()
    elif isinstance(date_reference, date_type):
        current_check_date = date_reference
    else:
        try:
            current_check_date = pd.to_datetime(date_reference).date()
        except Exception as e:
            print(f"[오류] get_trading_day_before: 날짜 변환 실패 ({date_reference}): {e}. 오늘 날짜 사용.")
            current_check_date = timezone.now().date() 
    
    kr_holidays = holidays.KR(years=[current_check_date.year -1, current_check_date.year, current_check_date.year + 1])
    trading_days_found = 0
    temp_date = current_check_date
    while trading_days_found < days_offset:
        temp_date -= timedelta(days=1)
        if temp_date.weekday() < 5 and temp_date not in kr_holidays: 
            trading_days_found += 1
    return temp_date

# --- Helper Functions (모델 학습용) ---
def _create_lstm_sequences_for_retrain(features_np, target_np, time_steps, future_target_days):
    X_data, y_data = [], []
    for i in range(len(features_np) - time_steps): 
        X_data.append(features_np[i:(i + time_steps)])
        y_data.append(target_np[i + time_steps - 1]) 
    if not X_data or not y_data:
        return np.array([]).reshape(0, time_steps, features_np.shape[1] if features_np.ndim > 1 and features_np.shape[1]>0 else 0 ), \
               np.array([]).reshape(0, future_target_days if target_np.ndim > 1 and target_np.shape[1] == future_target_days else 0)
    return np.array(X_data), np.array(y_data)

def _load_and_prepare_training_data_from_csv(data_folder_path, market_identifier, feature_cols_for_model, target_col, future_target_days):
    all_features_list = []
    all_targets_list = []
    search_pattern = os.path.join(data_folder_path, f"*{market_identifier}*features_manualTA.csv")
    file_paths = glob.glob(search_pattern)

    if not file_paths:
        print(f"학습 데이터 로드 경고: {data_folder_path}에서 '{market_identifier}' CSV 파일을 찾을 수 없습니다. (패턴: {search_pattern})")
        return pd.DataFrame(), pd.DataFrame()

    # print(f"'{market_identifier}' 시장 학습 데이터 로드 중... 총 {len(file_paths)}개 파일.") # 로그 빈도 조절
    loaded_count = 0
    for file_path in file_paths:
        df_stock = None 
        try:
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                continue
            try:
                df_stock = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                # print(f"[경고] _load_and_prepare: 파일이 비어있거나 파싱할 컬럼이 없습니다: {file_path}. 건너<0xEB><0><0xA9>니다.")
                continue 
            except Exception as e_read: 
                # print(f"[경고] _load_and_prepare: {file_path} 파일 읽기 중 예상치 못한 오류: {e_read}. 건너<0xEB><0><0xA9>니다.")
                continue

            if df_stock is None or df_stock.empty: 
                continue

            df_stock['Date'] = pd.to_datetime(df_stock['Date']) 
            df_stock = df_stock.sort_values('Date').reset_index(drop=True) # 여기서 인덱스 초기화
            
            if not all(col in df_stock.columns for col in feature_cols_for_model):
                missing = [col for col in feature_cols_for_model if col not in df_stock.columns]
                # print(f"파일 {file_path}에 일부 모델 피처 컬럼 누락: {missing}. 건너<0xEB><0><0xA9>니다.")
                continue

            # 타겟 컬럼 생성
            target_cols_names = [f'Target_Day_{d}' for d in range(1, future_target_days + 1)]
            for i, col_name in enumerate(target_cols_names):
                df_stock[col_name] = df_stock[target_col].shift(-(i+1))
            
            # NaN 제거: 모델 피처와 모든 타겟 컬럼에 대해 한 번에 수행
            cols_to_dropna = feature_cols_for_model + target_cols_names
            df_cleaned = df_stock.dropna(subset=cols_to_dropna)
            
            if df_cleaned.empty:
                continue

            final_features = df_cleaned[feature_cols_for_model].reset_index(drop=True)
            final_targets = df_cleaned[target_cols_names].reset_index(drop=True)

            if not final_features.empty and not final_targets.empty:
                all_features_list.append(final_features)
                all_targets_list.append(final_targets)
                loaded_count +=1
        except Exception as e: 
            print(f"학습 데이터 파일 {file_path} 처리 중 루프 내 오류: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # print(f"실제로 로드 및 처리된 학습 데이터 파일 수: {loaded_count}") # 로그 빈도 조절
    if not all_features_list or not all_targets_list:
        return pd.DataFrame(), pd.DataFrame()

    return pd.concat(all_features_list, ignore_index=True), pd.concat(all_targets_list, ignore_index=True)


def update_csv_and_retrain_market_model(market_name, market_csv_folder, market_id_original, 
                                        market_id_file_prefix_django, 
                                        ml_models_dir_param, is_initial_scaler_fit=False, retrain_model_enabled=False):
    print(f"\n--- {market_name} 시장 CSV 업데이트 및 모델 재학습 (재학습 활성화: {retrain_model_enabled}) ---")
    
    print(f"{market_name} 시장의 학습용 CSV 파일 업데이트 중...")
    stock_listing_raw = fdr.StockListing(market_name) 
    
    if 'Symbol' not in stock_listing_raw.columns and 'Code' in stock_listing_raw.columns:
        print(f"'{market_name}' 주식 목록에 'Symbol' 컬럼이 없어 'Code' 컬럼을 'Symbol'로 변경합니다.")
        stock_listing_raw.rename(columns={'Code': 'Symbol'}, inplace=True)
    elif 'Symbol' not in stock_listing_raw.columns:
        print(f"[오류] '{market_name}' 주식 목록에 'Symbol' 또는 'Code' 컬럼이 없습니다. CSV 업데이트를 건너<0xEB><0><0xA9>니다.")
        return
        
    stock_listing = stock_listing_raw[stock_listing_raw['Symbol'].notna() & stock_listing_raw['Name'].notna()]
    yesterday_trading_date = get_trading_day_before(timezone.now(), 1)
    num_updated_csv = 0

    for _, stock_row in stock_listing.iterrows():
        stock_code = stock_row['Symbol']
        target_csv_pattern = os.path.join(market_csv_folder, f"{stock_code}{market_id_original}*features_manualTA.csv")
        found_csv_files = glob.glob(target_csv_pattern)

        if not found_csv_files:
            continue
        
        if len(found_csv_files) > 1:
            try:
                found_csv_files.sort(key=os.path.getmtime, reverse=True)
            except Exception as e_sort:
                 print(f"[경고] {stock_code} CSV 파일 정렬 중 오류: {e_sort}. 첫 번째 파일 사용: {os.path.basename(found_csv_files[0])}")
        stock_csv_path_original = found_csv_files[0] 

        df_existing_csv = None 
        try:
            if not os.path.exists(stock_csv_path_original) or os.path.getsize(stock_csv_path_original) == 0:
                # print(f"[경고] update_csv: 파일이 존재하지 않거나 비어있습니다(0 bytes): {stock_csv_path_original}. 이 파일 처리를 건너<0xEB><0><0xA9>니다.")
                continue

            try:
                df_existing_csv = pd.read_csv(stock_csv_path_original)
            except pd.errors.EmptyDataError: 
                # print(f"[경고] update_csv (EmptyDataError): 파일이 비어있거나 파싱할 컬럼이 없습니다: {stock_csv_path_original}. 건너<0xEB><0><0xA9>니다.")
                continue 
            except Exception as e_read_csv: 
                # print(f"[경고] update_csv (Exception): {stock_csv_path_original} 파일 읽기 중 예상치 못한 오류. 타입: {type(e_read_csv)}, 메시지: {e_read_csv}. 건너<0xEB><0><0xA9>니다.")
                continue

            if df_existing_csv is None or df_existing_csv.empty: 
                # print(f"CSV 데이터 없음 (읽기 실패 또는 행 없음): {stock_csv_path_original}. 건너<0xEB><0><0xA9>니다.")
                continue

            df_existing_csv['Date'] = pd.to_datetime(df_existing_csv['Date']).dt.date 
            
            last_csv_date = df_existing_csv['Date'].max()
            df_combined_data_for_ta = df_existing_csv.copy() 

            if last_csv_date < yesterday_trading_date:
                start_fetch_date = last_csv_date + timedelta(days=1)
                df_new_data_raw = fdr.DataReader(stock_code, start=start_fetch_date, end=yesterday_trading_date)
                
                if not df_new_data_raw.empty:
                    df_new_data_raw.index = pd.to_datetime(df_new_data_raw.index).date
                    df_new_data_raw = df_new_data_raw.reset_index().rename(columns={'index': 'Date'})
                    
                    cols_from_fdr = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df_new_data_ohlcv = df_new_data_raw[cols_from_fdr].copy()
                    
                    existing_cols_to_keep = [col for col in cols_from_fdr if col in df_existing_csv.columns]
                    df_existing_base = df_existing_csv[existing_cols_to_keep].copy()
                    
                    df_combined_data_for_ta = pd.concat([df_existing_base, df_new_data_ohlcv], ignore_index=True)
                    df_combined_data_for_ta = df_combined_data_for_ta.drop_duplicates(subset=['Date'], keep='last')
                    df_combined_data_for_ta = df_combined_data_for_ta.sort_values('Date').reset_index(drop=True)
            
            if len(df_combined_data_for_ta) > APPROX_5YR_TRADING_DAYS:
                df_combined_data_rolled = df_combined_data_for_ta.tail(APPROX_5YR_TRADING_DAYS).reset_index(drop=True)
            else:
                df_combined_data_rolled = df_combined_data_for_ta.copy()

            if df_combined_data_rolled.empty:
                # print(f"[경고] 데이터 처리 후 DataFrame이 비어있습니다 (롤링 후): {stock_code}. 이 파일 처리를 건너<0xEB><0><0xA9>니다.")
                continue

            df_updated_ta = calculate_manual_features(df_combined_data_rolled.copy()) 
            
            if df_updated_ta.empty: 
                # print(f"[경고] 기술적 지표 계산 후 DataFrame이 비어있습니다: {stock_code}. 이 파일 처리를 건너<0xEB><0><0xA9>니다.")
                continue

            if 'Date' not in df_updated_ta.columns:
                print(f"[오류] CSV 저장: {stock_code} - 기술적 지표 계산 후 df_updated_ta에 'Date' 컬럼이 없습니다. 이 파일 처리를 건너<0xEB><0><0xA9>니다.")
                continue 
            
            df_to_process_for_save = df_updated_ta.copy()
            df_to_process_for_save['Date_str_formatted'] = pd.to_datetime(df_to_process_for_save['Date']).dt.strftime('%Y-%m-%d')
            
            df_to_save_final = pd.DataFrame()
            df_to_save_final['Date'] = df_to_process_for_save['Date_str_formatted']

            for col in FEATURE_COLUMNS_TRAINING: 
                if col in df_to_process_for_save.columns:
                    df_to_save_final[col] = df_to_process_for_save[col]
                else:
                    df_to_save_final[col] = np.nan 
            
            df_to_save_final = df_to_save_final[['Date'] + FEATURE_COLUMNS_TRAINING]

            if df_to_save_final.empty or df_to_save_final[FEATURE_COLUMNS_TRAINING].dropna(how='all').empty : 
                # print(f"[경고] 최종 저장할 DataFrame(df_to_save_final)이 비어있거나 모든 피처가 NaN입니다: {stock_code}. CSV 파일을 저장하지 않습니다.")
                continue

            temp_date_col_for_filename = pd.to_datetime(df_to_save_final['Date']) 
            if temp_date_col_for_filename.empty:
                # print(f"[경고] 파일명 생성: {stock_code} - 'Date' 컬럼이 비어있어 날짜 범위를 결정할 수 없습니다. 기존 파일명 사용.")
                new_stock_csv_path = stock_csv_path_original 
            else:
                min_date_in_df_str = temp_date_col_for_filename.min().strftime('%Y%m%d')
                max_date_in_df_str = temp_date_col_for_filename.max().strftime('%Y%m%d')
                
                new_filename_parts = [
                    stock_code, market_id_original, "daily_",
                    min_date_in_df_str, "_", max_date_in_df_str,
                    "_features_manualTA.csv"
                ]
                new_filename = "".join(new_filename_parts)
                new_stock_csv_path = os.path.join(market_csv_folder, new_filename)

            df_to_save_final.to_csv(new_stock_csv_path, index=False)

            if stock_csv_path_original != new_stock_csv_path and os.path.exists(stock_csv_path_original):
                try:
                    os.remove(stock_csv_path_original)
                except Exception as e_del:
                    print(f"[오류] 이전 CSV 파일 삭제 실패 ({stock_csv_path_original}): {e_del}")
            
            num_updated_csv += 1
            if num_updated_csv > 0 and num_updated_csv % 100 == 0: 
                 print(f"{market_name} CSV 업데이트 진행: {num_updated_csv}개 완료...")

        except Exception as e_csv_outer: 
            print(f"CSV 파일 {stock_csv_path_original} 처리 중 외부 루프 오류: {e_csv_outer}")
            import traceback
            traceback.print_exc()
            continue
            
    print(f"{market_name} 시장 총 {num_updated_csv}개 CSV 파일 업데이트 완료.")

    if not retrain_model_enabled: 
        print(f"{market_name} 모델 재학습 비활성화됨. CSV 업데이트만 수행 완료.")
        return 

    print(f"{market_name} 모델 재학습 시작...") 
    if market_id_file_prefix_django == "kosdaq_technical":
        model_file_path = os.path.join(ml_models_dir_param, 'kosdaq_technical_model.keras')
        scaler_X_file_path = os.path.join(ml_models_dir_param, 'kosdaq_technical_scaler_X.joblib') 
        scaler_y_file_path = os.path.join(ml_models_dir_param, 'kosdaq_technical_scaler_y.joblib') 
    elif market_id_file_prefix_django == "kospi_technical":
        model_file_path = os.path.join(ml_models_dir_param, 'kospi_technical_model.keras') 
        scaler_X_file_path = os.path.join(ml_models_dir_param, 'kospi_technical_scaler_X.joblib') 
        scaler_y_file_path = os.path.join(ml_models_dir_param, 'kospi_technical_scaler_y.joblib') 
    else:
        print(f"[오류] 알 수 없는 market_id_file_prefix_django: {market_id_file_prefix_django}. 모델/스케일러 경로를 설정할 수 없습니다.")
        return

    # print(f"모델 파일 경로 (재학습용): {model_file_path}") # 로그 빈도 조절
    # print(f"X 스케일러 파일 경로 (재학습용): {scaler_X_file_path}")
    # print(f"Y 스케일러 파일 경로 (재학습용): {scaler_y_file_path}")

    X_market_all_df, y_market_all_df = _load_and_prepare_training_data_from_csv(
        market_csv_folder, market_id_original, 
        FEATURE_COLUMNS_TRAINING, TARGET_COLUMN_TRAINING, FUTURE_TARGET_DAYS_TRAINING
    )
    if X_market_all_df.empty or y_market_all_df.empty:
        print(f"{market_name} 재학습용 데이터 없음. 건너<0xEB><0><0xA9>니다.")
        return

    if X_market_all_df.shape[1] != len(FEATURE_COLUMNS_TRAINING):
        print(f"[오류] 스케일러 적용 전 피처 개수 불일치. 예상: {len(FEATURE_COLUMNS_TRAINING)}, 실제: {X_market_all_df.shape[1]}")
        print(f"사용된 피처 컬럼: {X_market_all_df.columns.tolist()}")
        return

    if is_initial_scaler_fit or not (os.path.exists(scaler_X_file_path) and os.path.exists(scaler_y_file_path)):
        print(f"{market_name}: 새로운 스케일러 생성 및 저장 (또는 초기 학습으로 간주)...")
        scaler_X = MinMaxScaler()
        X_train_scaled_np = scaler_X.fit_transform(X_market_all_df.values) 
        joblib.dump(scaler_X, scaler_X_file_path)
        
        scaler_y = MinMaxScaler()
        y_train_scaled_np = scaler_y.fit_transform(y_market_all_df.values) 
        joblib.dump(scaler_y, scaler_y_file_path)
        print(f"{market_name} 스케일러 저장 완료: {scaler_X_file_path}, {scaler_y_file_path}")
    else:
        # print(f"{market_name}: 기존 스케일러 로드 중...") # 로그 빈도 조절
        scaler_X = joblib.load(scaler_X_file_path)
        scaler_y = joblib.load(scaler_y_file_path)
        X_train_scaled_np = scaler_X.transform(X_market_all_df.values)
        y_train_scaled_np = scaler_y.transform(y_market_all_df.values)

    X_train_seq, y_train_seq = _create_lstm_sequences_for_retrain(
        X_train_scaled_np, y_train_scaled_np, TIME_STEPS_TRAINING, FUTURE_TARGET_DAYS_TRAINING
    )
    if X_train_seq.size == 0:
        print(f"{market_name} 재학습용 시퀀스 데이터 없음.")
        return
    
    current_model = None
    if os.path.exists(model_file_path):
        # print(f"{market_name}: 기존 모델 로드: {model_file_path}") # 로그 빈도 조절
        current_model = load_model(model_file_path)
    else:
        print(f"{market_name}: 새로운 모델 생성 (경로: {model_file_path})")
        current_model = tf.keras.models.Sequential([
            Input(shape=(TIME_STEPS_TRAINING, len(FEATURE_COLUMNS_TRAINING))),
            LSTM(50, return_sequences=False),
            Dense(FUTURE_TARGET_DAYS_TRAINING)
        ])
        current_model.compile(optimizer='adam', loss='mean_squared_error')
    
    if current_model is None:
        print(f"[오류] {market_name} 모델을 로드하거나 생성하지 못했습니다.")
        return

    print(f"{market_name} 모델 추가 학습 ({EPOCHS_FOR_DAILY_RETRAIN} 에포크)...")
    checkpoint_cb = ModelCheckpoint(filepath=model_file_path, save_best_only=True, monitor='val_loss', verbose=0)
    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0) 
    
    history = current_model.fit(X_train_seq, y_train_seq,
                        epochs=EPOCHS_FOR_DAILY_RETRAIN,
                        batch_size=32,
                        validation_split=0.1, 
                        callbacks=[checkpoint_cb, early_stopping_cb],
                        verbose=1)
    
    print(f"{market_name} 모델 추가 학습 완료 및 저장됨: {model_file_path}")


def run_daily_startup_tasks_main(enable_model_retraining=False): 
    print(f"일일 데이터 업데이트 및 모델 학습 작업 (startup_tasks.py) 시작... 모델 재학습 활성화: {enable_model_retraining}")
    
    ml_models_dir_main = os.path.join(settings.BASE_DIR, 'predict_info', 'ml_models') 
    os.makedirs(ml_models_dir_main, exist_ok=True)
    # print(f"모델 및 스케일러 저장/로드 기본 경로: {ml_models_dir_main}") # 로그 빈도 조절

    try:
        kosdaq_csv_folder = settings.KOSDAQ_TRAINING_DATA_DIR 
        kospi_csv_folder = settings.KOSPI_TRAINING_DATA_DIR   
        
        if not os.path.isdir(kosdaq_csv_folder):
            print(f"[오류] KOSDAQ CSV 폴더 경로가 잘못되었거나 존재하지 않습니다: {kosdaq_csv_folder}")
            os.makedirs(kosdaq_csv_folder, exist_ok=True) 
            print(f"폴더 생성 시도: {kosdaq_csv_folder}")
        if not os.path.isdir(kospi_csv_folder):
            print(f"[오류] KOSPI CSV 폴더 경로가 잘못되었거나 존재하지 않습니다: {kospi_csv_folder}")
            os.makedirs(kospi_csv_folder, exist_ok=True) 
            print(f"폴더 생성 시도: {kospi_csv_folder}")

        # print(f"KOSDAQ CSV 폴더: {kosdaq_csv_folder}") # 로그 빈도 조절
        # print(f"KOSPI CSV 폴더: {kospi_csv_folder}")
    except AttributeError:
        print("[오류] settings.py에 KOSDAQ_TRAINING_DATA_DIR 또는 KOSPI_TRAINING_DATA_DIR가 정의되지 않았습니다.")
        print("이 작업은 CSV 폴더 경로 설정이 필요합니다. settings.py를 확인하세요.")
        return

    update_csv_and_retrain_market_model(
        market_name="KOSDAQ",
        market_csv_folder=kosdaq_csv_folder, 
        market_id_original="_kosdaq_", 
        market_id_file_prefix_django="kosdaq_technical", 
        ml_models_dir_param=ml_models_dir_main, 
        is_initial_scaler_fit=not os.path.exists(os.path.join(ml_models_dir_main, "kosdaq_technical_scaler_X.joblib")), 
        retrain_model_enabled=enable_model_retraining 
    )

    update_csv_and_retrain_market_model(
        market_name="KOSPI",
        market_csv_folder=kospi_csv_folder, 
        market_id_original="_kospi_",   
        market_id_file_prefix_django="kospi_technical", 
        ml_models_dir_param=ml_models_dir_main, 
        is_initial_scaler_fit=not os.path.exists(os.path.join(ml_models_dir_main, "kospi_technical_scaler_X.joblib")), 
        retrain_model_enabled=enable_model_retraining 
    )
    
    print("일일 데이터 업데이트 및 모델 학습 작업 (startup_tasks.py) 완료.")

if __name__ == '__main__':
    print("startup_tasks.py를 직접 실행하려면 Django 환경 설정이 필요합니다.")
    print("또는 Django management command로 만들어 실행하는 것을 권장합니다.")

