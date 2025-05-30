# predict_Info/startup_tasks.py
import pandas as pd
import FinanceDataReader as fdr
from django.utils import timezone # 현재 직접 사용 안함, 필요시 추가
from datetime import timedelta, datetime, date as date_type
import holidays
import time
import os
import glob
import numpy as np
import traceback 

from .utils import calculate_all_features, get_market_macro_data, PANDAS_TA_AVAILABLE
from django.conf import settings # settings.KOSDAQ_TRAINING_DATA_DIR 등 사용

# --- CSV 파일 및 모델 입력에 사용될 컬럼명 정의 ---
# 이 목록은 views.py 및 generate_daily_predictions.py의 컬럼 정의와 일관성을 가져야 합니다.
# 사용자의 새로운 데이터셋에 맞춰 이 부분을 정확하게 정의해야 합니다.

# 1. 기본 OHLCV 및 변동률 (FDR 출력 기준 또는 DB 저장 기준)
BASE_OHLCV_COLS_FOR_CSV = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']

# 2. 기술적 지표 (pandas-ta 등으로 계산 후 CSV에 저장될 컬럼명)
EXISTING_TA_COLS_FOR_CSV = [
    'ATR_14', 
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 
    'RSI_14',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
]
NEW_TA_COLS_FOR_CSV = [
    'STOCHk_14_3_3', 'STOCHd_14_3_3', 
    'OBV', 
    'ADX_14', 'DMP_14', 'DMN_14'
]
# USER_CUSTOM_TA_COLS_FOR_CSV = ['EMA_50', 'WILLR_14'] # 예시: 새로운 기술적 지표

# 3. 시장 지수 데이터 (KOSPI 또는 KOSDAQ) - market_name_upper에 따라 동적으로 결정
# update_stock_csv_with_all_features 함수 내에서 컬럼명 생성

# 4. 거시경제 지표
MACRO_DATA_COLS_FOR_CSV = ['USD_KRW_Close', 'USD_KRW_Change']

# 5. 투자자별 매매동향 (FDR 출력 컬럼명: '개인', '외국인', '기관')
# CSV 저장 시 사용할 컬럼명 (모델 입력과 일치시킬 수 있음)
INVESTOR_COLS_FOR_CSV = ['Indi', 'Foreign', 'Organ'] 

# 6. 펀더멘털 데이터 (FDR StockListing 컬럼명 또는 사용자가 CSV에 저장할 이름)
FUNDAMENTAL_COLS_FOR_CSV = ['Marcap', 'PER', 'PBR']
# USER_CUSTOM_FUNDAMENTAL_COLS_FOR_CSV = ['EPS', 'BPS', 'DPS', 'ROE'] # 예시: 새로운 펀더멘털 지표

# --- 기타 설정 ---
APPROX_5YR_TRADING_DAYS = 5 * 252 # CSV 파일에 저장할 대략적인 최대 데이터 기간 (일 수)

def get_trading_day_before(date_reference, days_offset=1):
    """주어진 날짜로부터 특정 거래일 수 이전의 날짜를 반환합니다."""
    if isinstance(date_reference, datetime): current_check_date = date_reference.date()
    elif isinstance(date_reference, date_type): current_check_date = date_reference
    else:
        try: current_check_date = pd.to_datetime(date_reference).date()
        except Exception as e:
            print(f"[Error] get_trading_day_before: Date conversion failed ({date_reference}): {e}. Using today.")
            current_check_date = timezone.now().date() # Django timezone 사용
    
    # 휴일 정보는 충분한 범위를 포함하여 계산
    relevant_years = list(set([current_check_date.year - (days_offset // 200 + 3), current_check_date.year, current_check_date.year + 2]))
    kr_holidays = holidays.KR(years=relevant_years)
    
    trading_days_found = 0
    temp_date = current_check_date
    
    if days_offset == 0: # days_offset이 0이면 기준일 자체가 거래일인지 확인 후 반환, 아니면 그 이전 거래일
        while temp_date.weekday() >= 5 or temp_date in kr_holidays:
            temp_date -= timedelta(days=1)
        return temp_date

    while trading_days_found < days_offset:
        temp_date -= timedelta(days=1)
        if temp_date.weekday() < 5 and temp_date not in kr_holidays:
            trading_days_found += 1
    return temp_date

def update_stock_csv_with_all_features(stock_code, stock_name, market_name_upper, base_csv_folder, market_id_suffix_for_filename, df_market_wide_fundamentals):
    """
    개별 종목의 CSV 파일을 모든 피처(OHLCV, TA, 거시경제, 투자자, 펀더멘털)를 포함하여 업데이트/생성합니다.
    df_market_wide_fundamentals: 미리 가져온 전체 시장의 펀더멘털 데이터 (DataFrame 또는 Dict)
    """
    if not PANDAS_TA_AVAILABLE:
        print(f"    [CRITICAL] {stock_name}({stock_code}) pandas_ta library missing. CSV update skipped.")
        return False
        
    print(f"  - Updating CSV for {stock_name}({stock_code}) (Market: {market_name_upper})...")

    # 파일명 규칙: {stock_code}{market_id_suffix}daily_{min_date}_{max_date}_features_all.csv (파일명 변경 가능)
    # 기존 파일 검색 패턴 (예시: _features_manualTA.csv -> _features_all.csv)
    # 새 파일명 규칙을 사용한다면 패턴도 변경해야 함
    new_filename_convention_part = "_features_all.csv" # 새 파일명 규칙에 맞게 수정
    general_pattern = os.path.join(base_csv_folder, f"{stock_code}{market_id_suffix_for_filename}*{new_filename_convention_part}")
    
    all_potential_files = glob.glob(general_pattern)
    
    df_existing_features_csv = pd.DataFrame()
    last_csv_date = None
    original_csv_path_to_remove = None

    if all_potential_files:
        try:
            all_potential_files.sort(key=os.path.getmtime, reverse=True) # 가장 최신 파일
            original_csv_path_to_remove = all_potential_files[0]
            if os.path.exists(original_csv_path_to_remove) and os.path.getsize(original_csv_path_to_remove) > 0:
                df_existing_features_csv = pd.read_csv(original_csv_path_to_remove)
                if 'Date' in df_existing_features_csv.columns:
                    df_existing_features_csv['Date'] = pd.to_datetime(df_existing_features_csv['Date']).dt.date
                    last_csv_date = df_existing_features_csv['Date'].max()
                else: df_existing_features_csv = pd.DataFrame() 
            else: df_existing_features_csv = pd.DataFrame() 
        except Exception as e_read:
            print(f"    [Warning] Error reading existing CSV ({original_csv_path_to_remove}): {e_read}. Fetching fresh data.")
            df_existing_features_csv = pd.DataFrame()

    # 어제 거래일 (데이터의 최신 기준일)
    yesterday_trading_date = get_trading_day_before(datetime.now(), 1) 
    
    if last_csv_date and last_csv_date >= yesterday_trading_date:
        print(f"    {stock_name}({stock_code}) CSV is up-to-date ({last_csv_date}). Skipping update.")
        return False # 이미 최신이면 업데이트 안 함

    # FDR 데이터 가져오기 시작 날짜 결정
    # CSV가 아예 없거나 마지막 날짜가 너무 오래 전이면 충분한 기간(5년치 + TA계산용 버퍼)을 가져옴
    # 그렇지 않으면 마지막 CSV 날짜 다음날부터 가져옴
    days_buffer_for_ta = 90 # TA 계산을 위한 추가 버퍼 일 수
    start_fetch_date_fdr = (last_csv_date + timedelta(days=1)) if last_csv_date else \
                           (yesterday_trading_date - timedelta(days=APPROX_5YR_TRADING_DAYS + days_buffer_for_ta))
    
    # 시작일이 목표일(어제 거래일)보다 뒤면 안됨
    if start_fetch_date_fdr > yesterday_trading_date:
        print(f"    {stock_name}({stock_code}) No new data period to fetch (start_fetch: {start_fetch_date_fdr}, yesterday_trade: {yesterday_trading_date}). Skipping.")
        return False
        
    print(f"    Fetching FDR OHLCV/Investor data for {stock_name}({stock_code}): {start_fetch_date_fdr} to {yesterday_trading_date} (source: naver)")
    
    try:
        # FDR에서 OHLCV 및 투자자별 거래량 데이터 가져오기
        df_new_ohlcv_investor_raw = fdr.DataReader(stock_code, start=start_fetch_date_fdr, end=yesterday_trading_date, data_source='naver')
        
        if df_new_ohlcv_investor_raw.empty:
            if not last_csv_date: # 기존 CSV도 없는데 FDR 데이터도 없으면 문제
                print(f"    [Warning] No new data from FDR for {stock_name}({stock_code}) and no existing CSV. Skipping.")
                return False 
            else: # 기존 CSV는 있는데 추가 데이터가 없는 경우
                print(f"    No additional new data from FDR for {stock_name}({stock_code}). Existing CSV data maintained if up-to-date.")
                # 이 경우, 위에서 last_csv_date >= yesterday_trading_date 조건으로 이미 걸러졌어야 함.
                # 만약 여기까지 왔다면, last_csv_date가 yesterday_trading_date보다 이전인데 추가 데이터가 없는 상황.
                # 이럴 때는 기존 CSV를 그대로 유지하거나, 에러 처리할 수 있음. 여기서는 일단 False 반환.
                return False

        df_new_ohlcv_investor_raw.index = pd.to_datetime(df_new_ohlcv_investor_raw.index).date
        df_new_ohlcv_investor_raw.rename_axis('Date', inplace=True)
        
        # 투자자별 컬럼명 변경 (FDR '개인' -> CSV/모델 'Indi')
        investor_col_map_fdr_to_csv = {'개인': 'Indi', '외국인': 'Foreign', '기관': 'Organ'}
        df_new_ohlcv_investor_raw.rename(columns=investor_col_map_fdr_to_csv, inplace=True)
        
        # 필요한 기본 컬럼 목록 (OHLCV, Change, 투자자)
        cols_from_fdr_standardized = BASE_OHLCV_COLS_FOR_CSV + INVESTOR_COLS_FOR_CSV
        for col in cols_from_fdr_standardized: # 누락된 컬럼 NaN으로 추가 (Change는 pct_change로 계산되므로 제외 가능)
            if col not in df_new_ohlcv_investor_raw.columns and col != 'Change': 
                df_new_ohlcv_investor_raw[col] = np.nan
        
        # Change 컬럼 계산 (FDR에서 이미 제공하지만, 일관성을 위해 재계산 또는 확인)
        if 'Close' in df_new_ohlcv_investor_raw.columns:
            df_new_ohlcv_investor_raw['Change'] = df_new_ohlcv_investor_raw['Close'].pct_change()
        else:
            df_new_ohlcv_investor_raw['Change'] = np.nan

        # 새로 가져온 데이터에서 필요한 컬럼만 선택
        df_new_data_processed = df_new_ohlcv_investor_raw.reset_index()[['Date'] + cols_from_fdr_standardized]

        # 기존 CSV 데이터와 새로 가져온 데이터 병합
        if not df_existing_features_csv.empty and 'Date' in df_existing_features_csv.columns:
            # 기존 CSV에서 기본 컬럼만 가져와서 합치고, TA 등은 나중에 전체 데이터로 재계산
            base_cols_in_existing_csv = ['Date'] + [col for col in cols_from_fdr_standardized if col in df_existing_features_csv.columns]
            df_base_for_concat = df_existing_features_csv[base_cols_in_existing_csv]
            df_combined_base_ohlcv_investor = pd.concat([df_base_for_concat, df_new_data_processed], ignore_index=True)
        else:
            df_combined_base_ohlcv_investor = df_new_data_processed.copy()

        # 중복 제거 (최신 데이터 우선) 및 날짜 정렬
        df_combined_base_ohlcv_investor = df_combined_base_ohlcv_investor.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
        
        if df_combined_base_ohlcv_investor.empty or 'Date' not in df_combined_base_ohlcv_investor.columns:
             print(f"    {stock_name}({stock_code}) Data empty after merge or Date column missing. Skipping CSV update.")
             return False
        df_combined_base_ohlcv_investor.set_index('Date', inplace=True)

        # --- 거시경제 및 시장 지수 데이터 추가 ---
        min_date_for_others = df_combined_base_ohlcv_investor.index.min()
        max_date_for_others = df_combined_base_ohlcv_investor.index.max()
        
        market_fdr_code_param = 'KS11' if market_name_upper == 'KOSPI' else 'KQ11' # 시장 지수 FDR 티커
        df_market_idx_data, df_macro_fx_data = get_market_macro_data(min_date_for_others, max_date_for_others, market_fdr_code=market_fdr_code_param)

        df_to_calculate_ta_final = df_combined_base_ohlcv_investor.copy()
        
        # 시장 지수 컬럼명 (예: 'KOSPI_Close', 'KOSPI_Change')
        market_index_cols_to_add = [f'{market_name_upper}_Close', f'{market_name_upper}_Change']
        if not df_market_idx_data.empty: 
            df_to_calculate_ta_final = df_to_calculate_ta_final.join(df_market_idx_data, how='left')
        else: 
            for col in market_index_cols_to_add: df_to_calculate_ta_final[col] = np.nan
        
        # 거시경제 지표 컬럼명 (MACRO_DATA_COLS_FOR_CSV)
        if not df_macro_fx_data.empty: 
            df_to_calculate_ta_final = df_to_calculate_ta_final.join(df_macro_fx_data, how='left')
        else: 
            for col in MACRO_DATA_COLS_FOR_CSV: df_to_calculate_ta_final[col] = np.nan
        
        # 투자자별 매매동향 컬럼 NaN 값 0.0으로 채우기 (TA 계산 전)
        for col in INVESTOR_COLS_FOR_CSV: 
            if col not in df_to_calculate_ta_final.columns: df_to_calculate_ta_final[col] = 0.0
            else: df_to_calculate_ta_final[col].fillna(0.0, inplace=True)
        
        # --- 펀더멘털 데이터 추가 ---
        # df_market_wide_fundamentals는 {stock_code: {'Marcap': val, 'PER': val, ...}} 형태의 dict 또는 DataFrame일 수 있음
        # 여기서는 DataFrame으로 가정하고, 해당 stock_code의 데이터를 가져옴
        stock_specific_fundamentals = {}
        if isinstance(df_market_wide_fundamentals, pd.DataFrame) and stock_code in df_market_wide_fundamentals.index:
            stock_specific_fundamentals = df_market_wide_fundamentals.loc[stock_code].to_dict()
        elif isinstance(df_market_wide_fundamentals, dict) and stock_code in df_market_wide_fundamentals:
            stock_specific_fundamentals = df_market_wide_fundamentals[stock_code]

        for fund_col_csv in FUNDAMENTAL_COLS_FOR_CSV: # + USER_CUSTOM_FUNDAMENTAL_COLS_FOR_CSV (필요시)
            # 펀더멘털 데이터는 스냅샷 값이므로 모든 행에 동일하게 적용
            # PER, PBR 등 NaN 값은 여기서 0.0으로 채움 (사용자 요청)
            df_to_calculate_ta_final[fund_col_csv] = stock_specific_fundamentals.get(fund_col_csv, 0.0) 
            if pd.isna(df_to_calculate_ta_final[fund_col_csv]).all(): # 전체가 NaN이면 0.0으로
                 df_to_calculate_ta_final[fund_col_csv] = 0.0
            elif pd.isna(df_to_calculate_ta_final[fund_col_csv]).any(): # 부분 NaN도 0.0으로
                 df_to_calculate_ta_final[fund_col_csv].fillna(0.0, inplace=True)


        # 외부 데이터(시장지수, 환율, 펀더멘털) 추가 후 ffill (TA 계산 전 마지막 ffill)
        # OHLCV는 이미 위에서 ffill/bfill 되었을 수 있음. 여기서는 주로 외부 데이터의 NaN 처리.
        # 주의: 펀더멘털은 스냅샷이므로 ffill 대상이 아님. 이미 모든 행에 값이 할당됨.
        cols_to_ffill_after_join = market_index_cols_to_add + MACRO_DATA_COLS_FOR_CSV
        df_to_calculate_ta_final[cols_to_ffill_after_join] = df_to_calculate_ta_final[cols_to_ffill_after_join].ffill()
        
        # --- 기술적 지표 계산 ---
        df_final_features_for_csv = calculate_all_features(df_to_calculate_ta_final.copy(), market_name_upper=market_name_upper)
        
        # 최종 CSV 저장용 컬럼 순서 정의
        # views.py의 get_feature_columns_for_market 순서와 유사하게 맞춤
        current_market_all_feature_columns_ordered_for_csv = (
            BASE_OHLCV_COLS_FOR_CSV + 
            EXISTING_TA_COLS_FOR_CSV + NEW_TA_COLS_FOR_CSV + # USER_CUSTOM_TA_COLS_FOR_CSV +
            market_index_cols_to_add + 
            MACRO_DATA_COLS_FOR_CSV + 
            INVESTOR_COLS_FOR_CSV + 
            FUNDAMENTAL_COLS_FOR_CSV # + USER_CUSTOM_FUNDAMENTAL_COLS_FOR_CSV
        )
        
        # 최종 DataFrame에 모든 예상 컬럼이 있는지 확인하고, 없으면 NaN으로 추가
        for col_expected in current_market_all_feature_columns_ordered_for_csv:
            if col_expected not in df_final_features_for_csv.columns:
                print(f"    [Warning] Final feature set for CSV missing '{col_expected}'. Adding as NaN. ({stock_name})")
                df_final_features_for_csv[col_expected] = np.nan
        
        df_final_features_for_csv.reset_index(inplace=True) # Date 컬럼 복원
        if df_final_features_for_csv.empty or 'Date' not in df_final_features_for_csv.columns:
            print(f"    {stock_name}({stock_code}) Final features for CSV empty or Date column missing. Skipping save.")
            return False
        
        # CSV 저장용 컬럼만 선택 및 순서 맞춤
        cols_to_save_ordered_in_csv = ['Date'] + current_market_all_feature_columns_ordered_for_csv
        df_to_save_final_csv = df_final_features_for_csv[cols_to_save_ordered_in_csv].copy()
        
        # 너무 오래된 데이터는 CSV에서 제외 (예: 최근 5년치)
        if len(df_to_save_final_csv) > APPROX_5YR_TRADING_DAYS: 
            df_to_save_final_csv = df_to_save_final_csv.tail(APPROX_5YR_TRADING_DAYS)
        
        # Date 컬럼 형식 통일
        df_to_save_final_csv['Date'] = pd.to_datetime(df_to_save_final_csv['Date']).dt.strftime('%Y-%m-%d')
        
        # 최종적으로 숫자형 컬럼들의 NaN 값을 0.0으로 채움 (PER, PBR은 위에서 이미 처리됨)
        numeric_cols_for_final_fillna = [col for col in df_to_save_final_csv.columns if col != 'Date']
        df_to_save_final_csv[numeric_cols_for_final_fillna] = df_to_save_final_csv[numeric_cols_for_final_fillna].fillna(0.0)

        if df_to_save_final_csv.empty or df_to_save_final_csv.drop(columns=['Date']).isnull().all().all():
            print(f"    {stock_name}({stock_code}) No data to save in CSV or all features are NaN. Skipping save.")
            return False

        # 새 파일명 생성 (날짜 범위 포함)
        min_date_str = pd.to_datetime(df_to_save_final_csv['Date']).min().strftime('%Y%m%d')
        max_date_str = pd.to_datetime(df_to_save_final_csv['Date']).max().strftime('%Y%m%d')
        new_filename = f"{stock_code}{market_id_suffix_for_filename}daily_{min_date_str}_{max_date_str}{new_filename_convention_part}"
        new_csv_path = os.path.join(base_csv_folder, new_filename)
        
        df_to_save_final_csv.to_csv(new_csv_path, index=False, encoding='utf-8-sig')
        print(f"    {stock_name}({stock_code}) CSV saved: {new_filename} (Rows: {len(df_to_save_final_csv)})")

        # 기존 파일 삭제 (새 파일과 경로가 다르고, 존재한다면)
        if original_csv_path_to_remove and original_csv_path_to_remove != new_csv_path and os.path.exists(original_csv_path_to_remove):
            try: 
                os.remove(original_csv_path_to_remove)
                print(f"    Old CSV file removed: {os.path.basename(original_csv_path_to_remove)}")
            except Exception as e_del: 
                print(f"    [Error] Failed to remove old CSV file ({original_csv_path_to_remove}): {e_del}")
        return True
        
    except Exception as e_main_csv_update:
        print(f"    [CRITICAL ERROR] Major error processing CSV for {stock_name}({stock_code}): {e_main_csv_update}")
        traceback.print_exc()
        return False

def run_daily_csv_update_tasks(market_config_list):
    """
    지정된 시장 목록에 대해 일일 CSV 데이터 업데이트 작업을 실행합니다.
    market_config_list: 각 시장의 설정(이름, CSV 폴더 경로, 파일명 접미사)을 담은 리스트
    """
    print(f"Daily CSV data update task (startup_tasks.py) started...")
    total_updated_count = 0
    
    # 시장 전체 펀더멘털 데이터 미리 한 번 가져오기 (FDR StockListing 사용)
    print("Fetching market-wide fundamental data (Marcap, PER, PBR, etc.) via StockListing('KRX-DESC')...")
    df_all_market_fundamentals = pd.DataFrame()
    try:
        df_all_market_fundamentals_raw = fdr.StockListing('KRX-DESC') # 더 많은 펀더멘털 정보 포함 가능
        if 'Code' in df_all_market_fundamentals_raw.columns: # 컬럼명 'Symbol'로 통일
            df_all_market_fundamentals_raw.rename(columns={'Code': 'Symbol'}, inplace=True)
        
        if 'Symbol' in df_all_market_fundamentals_raw.columns:
            df_all_market_fundamentals = df_all_market_fundamentals_raw.set_index('Symbol')
            # 필요한 펀더멘털 컬럼 숫자형으로 변환 시도 (오류 시 NaN)
            fund_cols_to_convert = FUNDAMENTAL_COLS_FOR_CSV # + USER_CUSTOM_FUNDAMENTAL_COLS_FOR_CSV
            for col in fund_cols_to_convert:
                if col in df_all_market_fundamentals.columns:
                    df_all_market_fundamentals[col] = pd.to_numeric(df_all_market_fundamentals[col], errors='coerce')
            print(f"Successfully fetched and processed fundamental data for {len(df_all_market_fundamentals)} stocks.")
        else:
            print("[Warning] 'Symbol' column not found in StockListing('KRX-DESC') result. Fundamental data in CSVs might be incomplete.")
    except Exception as e_fund_fetch:
        print(f"[Error] Failed to fetch market-wide fundamental data: {e_fund_fetch}. Fundamental data in CSVs might be incomplete.")
        traceback.print_exc()
        # df_all_market_fundamentals는 빈 DataFrame으로 유지됨

    for market_info_dict in market_config_list:
        market_name_display = market_info_dict['name'] # 예: 'KOSPI'
        market_csv_base_folder_path = market_info_dict['csv_folder'] # settings에서 가져온 경로
        market_file_id_suffix_val = market_info_dict['id_suffix'] # 예: '_kospi_'

        os.makedirs(market_csv_base_folder_path, exist_ok=True) # 해당 시장 폴더 생성
        print(f"\n--- Processing {market_name_display} market CSVs ---")
        print(f"Target CSV folder: {market_csv_base_folder_path}")

        # 해당 시장의 종목 목록 가져오기
        stock_listing_fdr_raw = fdr.StockListing(market_name_display) # 예: fdr.StockListing('KOSPI')
        if 'Code' in stock_listing_fdr_raw.columns: # 컬럼명 'Symbol'로 통일
            stock_listing_fdr_raw.rename(columns={'Code': 'Symbol'}, inplace=True)
        
        if 'Symbol' not in stock_listing_fdr_raw.columns or 'Name' not in stock_listing_fdr_raw.columns:
            print(f"[Error] {market_name_display} stock list from FDR missing 'Symbol' or 'Name' column. Skipping this market.")
            continue
            
        stock_listing_fdr_df = stock_listing_fdr_raw[stock_listing_fdr_raw['Symbol'].notna() & stock_listing_fdr_raw['Name'].notna()]
        num_stocks_in_this_market = len(stock_listing_fdr_df)
        market_specific_updated_count = 0

        for i, stock_list_row in stock_listing_fdr_df.iterrows():
            stock_code_val = stock_list_row['Symbol']
            stock_name_val = stock_list_row['Name']
            
            print(f"[{market_name_display} {i+1}/{num_stocks_in_this_market}] Processing CSV for {stock_name_val}({stock_code_val})...")
            
            if update_stock_csv_with_all_features(
                stock_code_val, stock_name_val, market_name_display.upper(), 
                market_csv_base_folder_path, market_file_id_suffix_val, 
                df_all_market_fundamentals # 미리 가져온 전체 펀더멘털 데이터 전달
            ):
                market_specific_updated_count +=1
            
            if (i+1) % 20 == 0: # 진행 상황 로그 출력 빈도
                print(f"  [{market_name_display}] {i+1} stocks checked for CSV update. {market_specific_updated_count} updated/created so far in this market.")
            time.sleep(0.05) # FDR 호출 간격 조절 (필요시) - 개별 종목 데이터 가져올 때 적용됨

        print(f"--- {market_name_display} market CSV update complete: {market_specific_updated_count} out of {num_stocks_in_this_market} stock CSVs updated/created ---")
        total_updated_count += market_specific_updated_count

    print(f"\nDaily CSV data update task (startup_tasks.py) finished. Total {total_updated_count} CSV files updated/created across all processed markets.")

def run_daily_startup_tasks_main(enable_model_retraining=False): 
    """
    일일 데이터 업데이트 작업의 메인 실행 함수. CSV 업데이트를 수행하고, 모델 재학습 옵션을 처리합니다.
    """
    print(f"Daily data update task (startup_tasks.py - main entry) started... (Model retraining: {'Enabled' if enable_model_retraining else 'Disabled'})")
    
    try:
        # settings.py에 정의된 경로 사용
        kosdaq_csv_folder_from_settings = settings.KOSDAQ_TRAINING_DATA_DIR
        kospi_csv_folder_from_settings = settings.KOSPI_TRAINING_DATA_DIR
        
        # 경로 존재 확인 및 생성 (settings.py에서 처리하거나 여기서 한 번 더 확인)
        if not os.path.isdir(kosdaq_csv_folder_from_settings): 
            os.makedirs(kosdaq_csv_folder_from_settings, exist_ok=True)
            print(f"Created KOSDAQ CSV directory: {kosdaq_csv_folder_from_settings}")
        if not os.path.isdir(kospi_csv_folder_from_settings): 
            os.makedirs(kospi_csv_folder_from_settings, exist_ok=True)
            print(f"Created KOSPI CSV directory: {kospi_csv_folder_from_settings}")

    except AttributeError:
        print("[CRITICAL ERROR][startup_tasks.py] KOSDAQ_TRAINING_DATA_DIR or KOSPI_TRAINING_DATA_DIR not defined in Django settings.py. Aborting CSV update.")
        return
    except Exception as e_path:
        print(f"[CRITICAL ERROR][startup_tasks.py] Error accessing or creating CSV directories: {e_path}. Aborting CSV update.")
        traceback.print_exc()
        return

    markets_to_process_config = [
        {'name': 'KOSDAQ', 'csv_folder': kosdaq_csv_folder_from_settings, 'id_suffix': '_kosdaq_'},
        {'name': 'KOSPI', 'csv_folder': kospi_csv_folder_from_settings, 'id_suffix': '_kospi_'}
    ]
    run_daily_csv_update_tasks(markets_to_process_config) # CSV 업데이트 실행
    
    if enable_model_retraining:
        print("Model retraining logic needs to be implemented separately.")
        print("This could involve calling another management command or a dedicated script that uses the updated CSVs.")
        # 예: from django.core.management import call_command
        # try:
        #     print("Attempting to run model retraining command (e.g., 'train_models')...")
        #     call_command('train_models', '--markets=KOSPI,KOSDAQ') # 예시 커맨드
        #     print("Model retraining command finished.")
        # except Exception as e_retrain:
        #     print(f"[ERROR] Failed to execute model retraining command: {e_retrain}")
        #     traceback.print_exc()
    
    print("Daily data update task (startup_tasks.py - main entry) finished.")

if __name__ == '__main__':
    # Django 환경 외부에서 직접 실행 테스트용 (settings.py 임시 모킹)
    print("Running startup_tasks.py directly (for testing, without full Django settings)...")
    
    # 현재 파일의 디렉토리 기준 (predict_info/startup_tasks.py)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # predict_info 폴더 (current_script_dir)
    # 프로젝트 루트 폴더 (predict_info 폴더의 부모)
    project_root_dir_mock = os.path.dirname(current_script_dir) 
    
    class MockSettingsForTesting:
        # BASE_DIR 대신 직접 경로 구성
        # 사용자가 언급한 경로: ROOT/predict_info/kospi_data, ROOT/predict_info/kosdaq_data
        # 여기서 ROOT는 Django 프로젝트의 최상위 디렉토리 (manage.py가 있는 곳)
        # project_root_dir_mock 이 Django 프로젝트 루트를 가리킨다고 가정
        KOSDAQ_TRAINING_DATA_DIR = os.path.join(project_root_dir_mock, 'predict_info', 'kosdaq_data_test') # 테스트용 다른 폴더
        KOSPI_TRAINING_DATA_DIR = os.path.join(project_root_dir_mock, 'predict_info', 'kospi_data_test')
        # ML_MODELS_DIR 등 다른 필요한 설정도 모킹 가능
    
    global settings # 전역 settings 객체를 임시로 모킹
    settings = MockSettingsForTesting()
    
    print(f"Mock KOSDAQ_TRAINING_DATA_DIR for test: {settings.KOSDAQ_TRAINING_DATA_DIR}")
    print(f"Mock KOSPI_TRAINING_DATA_DIR for test: {settings.KOSPI_TRAINING_DATA_DIR}")
    
    run_daily_startup_tasks_main(enable_model_retraining=False) # 테스트 시에는 모델 재학습 비활성화
