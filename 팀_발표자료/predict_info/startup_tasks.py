import pandas as pd
import FinanceDataReader as fdr
from django.utils import timezone # 사용 안 함, datetime으로 대체
from datetime import timedelta, datetime, date as date_type
import holidays # utils.get_kr_holidays 사용
import time
import os
import glob # 파일 검색
import numpy as np
import traceback

# utils.py의 함수들을 정확히 임포트
from .utils import (
    calculate_all_features, get_market_macro_data, PANDAS_TA_AVAILABLE,
    get_feature_columns_for_model, get_kr_holidays,
    add_fundamental_indicator_features, # 추가
    CSV_FILENAME_SUFFIX, CSV_FILENAME_DATE_FORMAT # utils에서 가져오기
)
from django.conf import settings
from django.core.management import call_command # 다른 management command 호출

# --- CSV 파일 및 모델 입력에 사용될 컬럼명 정의 ---
# 이 목록은 utils.get_feature_columns_for_model 의 정의와 일관성을 가져야 합니다.
# startup_tasks.py에서는 CSV 생성 시 utils.get_feature_columns_for_model을 직접 사용하여 컬럼 목록을 가져옵니다.

APPROX_5YR_TRADING_DAYS = 5 * 252 # CSV 파일에 저장할 대략적인 최대 데이터 기간 (일 수)
# CSV_FILENAME_SUFFIX = "_extendedTA.csv" # utils에서 가져옴
# CSV_FILENAME_DATE_FORMAT = "%Y%m%d" # utils에서 가져옴

def get_trading_day_before(date_reference, kr_holidays_list, days_offset=1):
    """주어진 날짜로부터 특정 거래일 수 이전의 날짜를 반환합니다."""
    if isinstance(date_reference, datetime): current_check_date = date_reference.date()
    elif isinstance(date_reference, date_type): current_check_date = date_reference
    else:
        try: current_check_date = pd.to_datetime(date_reference).date()
        except: raise TypeError("date_reference must be a datetime, date, or convertible to it.")

    days_found = 0
    while days_found < days_offset:
        current_check_date -= timedelta(days=1)
        if current_check_date.weekday() < 5 and current_check_date not in kr_holidays_list: # 주말(토:5, 일:6) 아니고 공휴일 아니면
            days_found += 1
    return current_check_date


def update_stock_data_csv(market_name, stock_code, stock_name, base_dir_market_data, kr_holidays_list, today_dt):
    """
    개별 종목의 CSV 데이터를 최신 거래일까지 업데이트합니다.
    데이터는 FinanceDataReader를 통해 가져오고, 기술적 지표 등을 추가하여 저장합니다.
    """
    print(f"  Updating CSV for {stock_name} ({stock_code}) - {market_name}...")
    
    # CSV 파일 저장 경로 설정 (market_name에 따라 kospi_data 또는 kosdaq_data)
    market_folder = 'kospi_data' if market_name.upper() == 'KOSPI' else 'kosdaq_data'
    stock_data_dir = os.path.join(base_dir_market_data, market_folder)
    os.makedirs(stock_data_dir, exist_ok=True)

    # 파일명 규칙: STOCKCODE_market_daily_from_YYYYMMDD_to_YYYYMMDD_extendedTA.csv
    # 기존 파일 검색하여 마지막 날짜 확인 또는 새로 생성
    
    # 최신 데이터 종료일은 '어제' (또는 가장 최근의 거래일)
    latest_data_target_date = get_trading_day_before(today_dt, kr_holidays_list, days_offset=1)
    
    # 데이터 시작일 결정: 약 5년 전 데이터부터 시작 (또는 기존 파일이 있으면 이어받기)
    # 기존 파일명 패턴: {stock_code}_{market_name.lower()}_daily_from_{start_date_str}_to_{end_date_str}{CSV_FILENAME_SUFFIX}
    # glob으로 해당 종목의 기존 CSV 파일 검색
    existing_files = glob.glob(os.path.join(stock_data_dir, f"{stock_code}_{market_name.lower()}_daily_*{CSV_FILENAME_SUFFIX}"))
    
    current_stock_df = pd.DataFrame()
    data_fetch_start_date_fdr = None # FDR에서 데이터를 가져올 시작 날짜
    csv_file_start_date_str = (latest_data_target_date - timedelta(days=APPROX_5YR_TRADING_DAYS)).strftime(CSV_FILENAME_DATE_FORMAT) # 기본 시작일

    if existing_files:
        latest_existing_file = max(existing_files, key=os.path.getmtime) # 수정일 기준 최신 파일
        print(f"    Found existing CSV: {os.path.basename(latest_existing_file)}")
        try:
            current_stock_df = pd.read_csv(latest_existing_file, index_col='Date', parse_dates=True)
            if not current_stock_df.empty:
                last_date_in_csv = current_stock_df.index.max().date()
                # 파일명에서 시작 날짜 추출 시도
                fname_parts = os.path.basename(latest_existing_file).split('_')
                if len(fname_parts) > 3 and fname_parts[3] == "from":
                    csv_file_start_date_str = fname_parts[4] # YYYYMMDD 형식

                if last_date_in_csv < latest_data_target_date:
                    data_fetch_start_date_fdr = (last_date_in_csv + timedelta(days=1)).strftime('%Y-%m-%d')
                    print(f"    Last date in CSV is {last_date_in_csv}. Fetching new data from {data_fetch_start_date_fdr} to {latest_data_target_date}.")
                else:
                    print(f"    CSV data for {stock_code} is already up-to-date ({last_date_in_csv}). No new data to fetch from FDR.")
                    # 그래도 TA 지표 등 재계산 및 저장은 필요할 수 있으므로 current_stock_df 사용
            else: # 기존 파일이 비어있으면 새로 시작
                data_fetch_start_date_fdr = (latest_data_target_date - timedelta(days=APPROX_5YR_TRADING_DAYS)).strftime('%Y-%m-%d')
                csv_file_start_date_str = data_fetch_start_date_fdr.replace("-","")

        except Exception as e_read_csv:
            print(f"    Error reading existing CSV {latest_existing_file}: {e_read_csv}. Fetching fresh data.")
            data_fetch_start_date_fdr = (latest_data_target_date - timedelta(days=APPROX_5YR_TRADING_DAYS)).strftime('%Y-%m-%d')
            csv_file_start_date_str = data_fetch_start_date_fdr.replace("-","")
    else: # 기존 파일 없음
        print(f"    No existing CSV found for {stock_code}. Fetching fresh data.")
        data_fetch_start_date_fdr = (latest_data_target_date - timedelta(days=APPROX_5YR_TRADING_DAYS)).strftime('%Y-%m-%d')
        csv_file_start_date_str = data_fetch_start_date_fdr.replace("-","")

    new_data_df = pd.DataFrame()
    if data_fetch_start_date_fdr and pd.to_datetime(data_fetch_start_date_fdr).date() <= latest_data_target_date :
        try:
            # OHLCV 데이터 가져오기
            new_ohlcv_df = fdr.DataReader(stock_code, data_fetch_start_date_fdr, latest_data_target_date.strftime('%Y-%m-%d'))
            if not new_ohlcv_df.empty:
                new_ohlcv_df.index.name = 'Date' # 인덱스 이름 설정
                # 필요한 컬럼만 선택 (Open, High, Low, Close, Volume, Change)
                # 'Change'는 fdr에서 제공. 없으면 직접 계산.
                if 'Change' not in new_ohlcv_df.columns and 'Close' in new_ohlcv_df.columns:
                    new_ohlcv_df['Change'] = new_ohlcv_df['Close'].pct_change()
                
                new_data_df = new_ohlcv_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Change']].copy()
                print(f"    Fetched {len(new_data_df)} new OHLCV records for {stock_code} from FDR.")
            else:
                print(f"    No new OHLCV data returned by FDR for {stock_code} from {data_fetch_start_date_fdr}.")
        except Exception as e_fdr:
            print(f"    Error fetching OHLCV data for {stock_code} from FDR: {e_fdr}")
            # 오류 발생 시 기존 데이터만 사용하거나, 빈 DataFrame으로 진행
            new_data_df = pd.DataFrame() # 확실히 비워줌

    # 기존 데이터와 새 데이터 병합
    if not new_data_df.empty:
        # current_stock_df에서 new_data_df와 겹치는 인덱스 제거 (혹시 모를 중복 방지)
        if not current_stock_df.empty:
            current_stock_df = current_stock_df[~current_stock_df.index.isin(new_data_df.index)]
        
        # OHLCV 및 Change 컬럼만 먼저 합치고, 나머지 피처는 전체에 대해 재계산
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']
        if not current_stock_df.empty:
            combined_ohlcv_df = pd.concat([current_stock_df[ohlcv_cols], new_data_df[ohlcv_cols]])
        else:
            combined_ohlcv_df = new_data_df[ohlcv_cols]
        
        combined_ohlcv_df.sort_index(inplace=True) # 날짜 순 정렬
        # 중복 인덱스 제거 (첫 번째 것 유지)
        combined_ohlcv_df = combined_ohlcv_df[~combined_ohlcv_df.index.duplicated(keep='first')]

    elif not current_stock_df.empty: # 새 데이터는 없지만 기존 데이터가 있는 경우
        combined_ohlcv_df = current_stock_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Change']].copy()
    else: # 데이터가 전혀 없는 경우
        print(f"    No data available for {stock_code} to process. Skipping CSV update.")
        return

    if combined_ohlcv_df.empty:
        print(f"    Combined OHLCV data is empty for {stock_code}. Skipping further processing.")
        return

    # --- 모든 피처 계산 ---
    # 1. 시장/거시경제 데이터 로드
    #    combined_ohlcv_df의 전체 기간에 맞춰 로드
    data_start_str = combined_ohlcv_df.index.min().strftime('%Y-%m-%d')
    data_end_str = combined_ohlcv_df.index.max().strftime('%Y-%m-%d')
    
    other_market_name = 'KOSDAQ' if market_name.upper() == 'KOSPI' else 'KOSPI'
    market_macro_df = get_market_macro_data(data_start_str, data_end_str, market_name.upper(), other_market_name_for_index=other_market_name)

    # 2. 투자자별 매매동향 데이터 로드 (fdr 사용)
    #    fdr.DataReader(ticker, start, end, data_source='naver') 로 투자자별 정보 가져올 수 있음
    #    또는 pykrx 사용: stock.get_market_trading_value_by_date(start, end, ticker, on='매수금액') 등
    #    여기서는 간단히 fdr의 투자자별 거래실적(기관, 외국인 등)을 사용한다고 가정.
    #    'Indi', 'Foreign', 'Organ' 컬럼을 생성해야 함.
    #    fdr.DataReader(stock_code, start_date, end_date, data_source='INVESTOR') -> Deprecated
    #    대안: pykrx 또는 직접 크롤링. 여기서는 예시로 비워둠. 필요시 추가 구현.
    investor_df = pd.DataFrame(index=combined_ohlcv_df.index) # 빈 DF로 시작
    try:
        # pykrx를 사용한 투자자별 순매수 (예시, 실제 사용 시 pykrx 설치 및 테스트 필요)
        # from pykrx import stock as pykrx_stock
        # investor_trading_df = pykrx_stock.get_market_trading_value_by_date(data_start_str, data_end_str, stock_code)
        # if not investor_trading_df.empty:
        #     # 컬럼명 '개인', '외국인', '기관' 등을 'Indi', 'Foreign', 'Organ'으로 변경
        #     investor_rename_map = {'개인': 'Indi', '외국인': 'Foreign', '기관전체': 'Organ'} # 실제 pykrx 컬럼명 확인
        #     investor_trading_df.rename(columns=investor_rename_map, inplace=True)
        #     investor_df = investor_trading_df[['Indi', 'Foreign', 'Organ']] # 필요한 컬럼만
        # else: # 데이터 없으면 빈 컬럼 생성
        #     investor_df[['Indi', 'Foreign', 'Organ']] = np.nan
        # 임시: 투자자 데이터는 NaN으로 채움 (실제 구현 시 대체 필요)
        investor_df[['Indi', 'Foreign', 'Organ']] = np.nan

    except Exception as e_investor:
        print(f"    Could not fetch investor data for {stock_code}: {e_investor}. Filling with NaN.")
        investor_df[['Indi', 'Foreign', 'Organ']] = np.nan


    # 3. 펀더멘탈 데이터 (시가총액, PBR, PER 등)
    #    fdr.SnapStockListing(date) 등으로 특정일의 펀더멘탈 가져오거나,
    #    pykrx.stock.get_market_fundamental_by_date(date, ticker) 사용 가능.
    #    또는 연간/분기별 재무제표에서 계산.
    #    여기서는 예시로 비워둠. 필요시 추가 구현.
    fundamental_df = pd.DataFrame(index=combined_ohlcv_df.index)
    try:
        # pykrx를 사용한 일별 펀더멘탈 (예시)
        # from pykrx import stock as pykrx_stock
        # daily_fundamental_df = pykrx_stock.get_market_fundamental_by_date(data_start_str, data_end_str, stock_code)
        # if not daily_fundamental_df.empty:
        #     # 컬럼명 'PBR', 'PER', 'EPS', 'BPS' 등 사용. '시가총액'은 직접 계산하거나 다른 소스.
        #     # 여기서는 PBR, PER만 사용한다고 가정. 시가총액은 종가 * 상장주식수.
        #     # '시가총액'은 fdr.StockListing().loc[stock_code]['Marcap'] 사용 가능 (특정일 기준)
        #     # 또는 fdr.DataReader(stock_code, data_source='naver')의 'MarketCap' 컬럼 사용 가능
        #     marcap_series = fdr.DataReader(stock_code, data_start_str, data_end_str, data_source='naver')['MarketCap']
        #     fundamental_df['MarketCap'] = marcap_series
        #     fundamental_df['PBR'] = daily_fundamental_df['PBR']
        #     fundamental_df['PER'] = daily_fundamental_df['PER']
        # else:
        #     fundamental_df[['MarketCap', 'PBR', 'PER']] = np.nan
        # 임시: 펀더멘탈 데이터는 NaN으로 채움 (실제 구현 시 대체 필요)
        fundamental_df[['MarketCap', 'PBR', 'PER']] = np.nan

    except Exception as e_fundamental:
        print(f"    Could not fetch fundamental data for {stock_code}: {e_fundamental}. Filling with NaN.")
        fundamental_df[['MarketCap', 'PBR', 'PER']] = np.nan

    # 모든 피처 계산
    all_features_df = calculate_all_features(
        stock_df_ohlcv=combined_ohlcv_df,
        market_macro_data_df=market_macro_df,
        investor_df=investor_df,
        fundamental_df=fundamental_df,
        pandas_ta_available=PANDAS_TA_AVAILABLE
    )
    
    # _is_nan, _is_zero 인디케이터 컬럼 추가
    all_features_df = add_fundamental_indicator_features(all_features_df)

    # 최종적으로 모델에 사용될 피처 목록 가져오기
    # market_name_for_features는 해당 시장의 모델이 어떤 시장 기준으로 학습되었는지 명시
    # (예: KOSPI 모델은 KOSPI 시장 피처 기준)
    model_feature_columns = get_feature_columns_for_model(market_name=market_name.upper(), model_type='technical')
    
    # all_features_df에 모든 model_feature_columns이 있는지 확인하고, 없으면 NaN으로 채우기
    for col in model_feature_columns:
        if col not in all_features_df.columns:
            print(f"    [WARN] Feature column '{col}' not found in all_features_df for {stock_code} (CSV gen). Filling with NaN.")
            all_features_df[col] = np.nan
            
    # CSV 저장 시에는 모델 입력 피처만 선택하여 저장 (또는 모든 계산된 피처 저장 후 모델 학습 시 선택)
    # 여기서는 모델 입력 피처만 저장하는 것으로 가정 (Colab 학습 데이터와 동일하게)
    df_to_save_csv = all_features_df[model_feature_columns].copy()
    
    # NaN 처리 (모델 학습 데이터와 동일한 방식 적용)
    # 예: Change의 첫 NaN은 0으로, 나머지는 ffill().bfill() 후 0으로.
    if 'Change' in df_to_save_csv.columns:
        df_to_save_csv['Change'].fillna(0, inplace=True)
    df_to_save_csv.ffill(inplace=True)
    df_to_save_csv.bfill(inplace=True)
    if df_to_save_csv.isnull().values.any():
        print(f"    [WARN] NaN values still present in df_to_save_csv for {stock_code} after ffill/bfill. Filling with 0.")
        df_to_save_csv.fillna(0, inplace=True)


    # 최종 CSV 파일명 결정
    # 파일명에 실제 데이터의 시작일과 종료일 반영
    if not df_to_save_csv.empty:
        actual_start_date_str = df_to_save_csv.index.min().strftime(CSV_FILENAME_DATE_FORMAT)
        actual_end_date_str = df_to_save_csv.index.max().strftime(CSV_FILENAME_DATE_FORMAT)
        
        # 기존 파일 삭제 (덮어쓰기 위해)
        for old_file in existing_files:
            try: os.remove(old_file)
            except OSError as e_remove: print(f"    Error removing old file {old_file}: {e_remove}")

        final_csv_filename = f"{stock_code}_{market_name.lower()}_daily_from_{actual_start_date_str}_to_{actual_end_date_str}{CSV_FILENAME_SUFFIX}"
        final_csv_path = os.path.join(stock_data_dir, final_csv_filename)
        
        df_to_save_csv.to_csv(final_csv_path)
        print(f"    Successfully updated and saved CSV: {final_csv_filename} (Rows: {len(df_to_save_csv)})")
    else:
        print(f"    No data to save in CSV for {stock_code}.")


def run_daily_startup_tasks_main(retrain_models_enabled=False):
    """
    일일 데이터 업데이트 작업을 실행하는 메인 함수.
    1. KRX 주식 목록 가져오기
    2. 각 주식에 대해 CSV 데이터 업데이트
    3. (선택적) 모델 재학습 management command 호출
    """
    print("Starting daily data update task (startup_tasks.py - main entry)...")
    start_time_total = time.time()

    today_datetime = datetime.now()
    current_year = today_datetime.year
    # 공휴일 정보 로드 (향후 2년치)
    kr_holidays = get_kr_holidays([current_year, current_year + 1, current_year + 2])

    # CSV 데이터가 저장될 기본 경로 (settings.py에서 가져옴)
    # KOSPI_TRAINING_DATA_DIR, KOSDAQ_TRAINING_DATA_DIR은 settings에 정의되어 있어야 함.
    # 여기서는 settings.BASE_DIR 아래 predict_info/kospi_data 등으로 가정.
    # 실제 경로는 settings.py의 정의를 따름.
    # 이 함수는 settings가 로드된 Django 환경에서 실행되므로 settings 사용 가능.
    
    # predict_info 앱 내의 kospi_data, kosdaq_data 디렉토리를 사용한다고 가정
    # settings.BASE_DIR은 manage.py가 있는 프로젝트 루트
    base_dir_for_csv_data = os.path.join(settings.BASE_DIR, 'predict_info')

    markets_to_process = ['KOSPI', 'KOSDAQ']
    
    # fdr을 통해 전체 KRX 주식 목록 가져오기 (마켓별로)
    # from .utils import get_krx_stock_list # 이미 임포트됨
    all_stocks_krx = []
    for market in markets_to_process:
        try:
            # get_krx_stock_list는 Code, Name, Market을 반환
            stocks_in_market = get_krx_stock_list(market=market) 
            all_stocks_krx.extend(stocks_in_market)
            print(f"Fetched {len(stocks_in_market)} stocks for {market} market.")
        except Exception as e_krx_list:
            print(f"[ERROR] Failed to get stock list for {market}: {e_krx_list}")
            continue
            
    if not all_stocks_krx:
        print("[ERROR] Could not retrieve any stock lists. Aborting CSV update.")
        return

    print(f"\nTotal stocks to process for CSV update: {len(all_stocks_krx)}")
    
    processed_csv_count = 0
    error_csv_count = 0

    for stock_info in all_stocks_krx:
        stock_code = stock_info.get('Code')
        stock_name = stock_info.get('Name', stock_code) # 이름 없으면 코드로 대체
        market_name = stock_info.get('Market')

        if not stock_code or not market_name:
            print(f"  Skipping stock due to missing code or market: {stock_info}")
            error_csv_count += 1
            continue
        
        try:
            update_stock_data_csv(market_name, stock_code, stock_name, base_dir_for_csv_data, kr_holidays, today_datetime)
            processed_csv_count += 1
        except Exception as e_update_csv:
            print(f"  [ERROR] Failed to update CSV for {stock_name} ({stock_code}): {e_update_csv}")
            traceback.print_exc()
            error_csv_count += 1
        time.sleep(0.1) # API 호출 등 부하 분산 (fdr 호출 시)

    print(f"\n--- CSV Update Summary ---")
    print(f"Total stocks considered: {len(all_stocks_krx)}")
    print(f"Successfully processed for CSV: {processed_csv_count}")
    print(f"Errors during CSV processing: {error_csv_count}")

    # 모델 재학습 로직 (옵션에 따라 실행)
    if retrain_models_enabled:
        print("\nStarting model retraining process (via management command)...")
        # 여기에 모델 재학습을 위한 management command 호출 로직 추가
        # 예: call_command('retrain_stock_models', '--markets=KOSPI,KOSDAQ', '--epochs=50')
        # 실제 재학습 command 이름과 옵션에 맞춰야 함.
        # 이 프로젝트에는 'retrain_stock_models' command가 없으므로, 해당 command를 만들거나
        # Colab 노트북 실행 스크립트를 호출하는 방식으로 구현해야 함.
        print("[INFO] Model retraining logic needs to be implemented here.")
        print("       (e.g., by calling a specific management command like 'retrain_models')")
        # try:
        #     print("Executing model retraining command...")
        #     # call_command('your_retrain_command_name', '--options_if_any')
        #     # 예시: call_command('train_models', '--markets=KOSPI,KOSDAQ')
        #     print("Model retraining command finished.")
        # except Exception as e_retrain:
        #     print(f"[ERROR] Failed to execute model retraining command: {e_retrain}")
        #     traceback.print_exc()
    
    end_time_total = time.time()
    print(f"Daily data update task (startup_tasks.py - main entry) finished in {end_time_total - start_time_total:.2f} seconds.")

# 이 파일이 직접 실행될 때 (테스트용, Django settings 없이)
if __name__ == '__main__':
    print("Running startup_tasks.py directly (for testing, without full Django settings)...")
    
    # 테스트를 위해 Django settings 모킹 또는 최소한의 설정 제공 필요
    # 예: settings.configure() 사용 또는 필요한 변수 직접 설정
    # 이 부분은 실제 Django 프로젝트 외부에서 테스트할 때만 의미가 있음.
    # Django manage.py를 통해 실행될 때는 settings가 자동으로 로드됨.
    
    # --- Mock settings for testing (if run directly) ---
    # 이 부분은 Django 프로젝트의 manage.py를 통해 실행될 때는 필요 없음.
    # 직접 실행 테스트를 원할 경우에만 사용.
    
    # current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # # predict_info 앱이 project_root/predict_info 에 있다고 가정
    # project_root_dir_mock = os.path.dirname(os.path.dirname(current_script_dir)) 
    
    # class MockSettingsForTesting:
    #     BASE_DIR = project_root_dir_mock # manage.py가 있는 곳
    #     # ML_MODELS_DIR 등 다른 필요한 settings 변수도 모킹
    #     # 이 방식보다는 Django의 테스트 환경을 사용하거나, manage.py를 통해 실행하는 것이 좋음.

    # global settings # 전역 settings 변수를 사용하기 위함 (권장되지 않음)
    # settings = MockSettingsForTesting()
    
    # print(f"Mock BASE_DIR for testing: {settings.BASE_DIR}")
    
    # run_daily_startup_tasks_main(retrain_models_enabled=False)
    print("Direct execution of startup_tasks.py is intended for limited testing.")
    print("For full functionality, run as a Django management command or ensure Django settings are configured.")