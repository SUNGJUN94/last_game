# predict_info/management/commands/update_daily_data.py
import pandas as pd
import FinanceDataReader as fdr
from django.core.management.base import BaseCommand
from django.utils import timezone # timezone.now() 사용을 위해 유지
from datetime import timedelta, datetime, date as date_type
from predict_info.models import StockPrice, MarketIndex 
from predict_info.utils import ( 
    calculate_all_features,
    get_market_macro_data, 
    get_kr_holidays,
    add_fundamental_indicator_features,
    PANDAS_TA_AVAILABLE,
    get_krx_stock_list # <<< get_krx_stock_list 임포트 추가
)
import time
import traceback
import numpy as np
from django.db import models as django_db_models
from django.db import transaction
from django.core.exceptions import FieldDoesNotExist
import logging # 로깅 추가

logger = logging.getLogger(__name__)

try:
    from pykrx import stock as pykrx_stock
    PYKRX_AVAILABLE = True
    logger.info("[update_daily_data.py] pykrx 라이브러리가 성공적으로 로드되었습니다.")
except ImportError:
    PYKRX_AVAILABLE = False
    logger.warning("[update_daily_data.py] 'pykrx' 라이브러리를 찾을 수 없습니다. 펀더멘털 및 일부 투자자 데이터가 누락될 수 있습니다.")


class Command(BaseCommand):
    help = ('Fetches and stores daily stock data (OHLCV, Investor Trends, Fundamentals, TA features) '
            'and Market Index data into the database.')

    MIN_RECORDS_FOR_SUFFICIENT_HISTORY = 120 
    DAYS_TO_FETCH_FOR_BACKFILL = 300       
    DAYS_TO_FETCH_OHLCV_BASE = DAYS_TO_FETCH_FOR_BACKFILL + 100


    def get_previous_trading_day(self, reference_date, kr_holidays_list, days_offset=1):
        """
        주어진 날짜로부터 특정 거래일 수 이전의 날짜를 반환합니다.
        days_offset=0 이면, reference_date가 거래일이면 reference_date를, 아니면 그 이전 최신 거래일을 반환합니다.
        """
        if not isinstance(reference_date, date_type):
            try: reference_date = pd.to_datetime(reference_date).date()
            except: raise TypeError("reference_date must be a datetime.date or convertible to it.")

        current_check_date = reference_date
        
        if days_offset == 0: 
            # reference_date 자체가 거래일인지 확인, 아니면 이전 거래일 탐색
            while current_check_date.weekday() >= 5 or current_check_date in kr_holidays_list:
                current_check_date -= timedelta(days=1)
            return current_check_date

        trading_days_found = 0
        # days_offset > 0 이면, 현재 날짜는 포함하지 않고 그 이전부터 카운트
        while trading_days_found < days_offset:
            current_check_date -= timedelta(days=1)
            if current_check_date.weekday() < 5 and current_check_date not in kr_holidays_list:
                trading_days_found += 1
        return current_check_date

    def fetch_pykrx_fundamentals(self, stock_code, start_date_str, end_date_str):
        if not PYKRX_AVAILABLE:
            fund_cols = ['MarketCap', 'PBR', 'PER', 'EPS', 'BPS', 'DPS', 'CashDPS', 'ROE']
            return pd.DataFrame(columns=fund_cols)

        df_fund_combined = pd.DataFrame()
        try:
            # pykrx는 YYYYMMDD 형식의 문자열을 사용
            df_cap_raw = pykrx_stock.get_market_cap_by_date(start_date_str, end_date_str, stock_code)
            if not df_cap_raw.empty:
                df_cap = df_cap_raw.reset_index()
                df_cap.rename(columns={'날짜': 'Date', '시가총액': 'MarketCap', '상장주식수': 'ListedShares'}, inplace=True)
                if 'Date' in df_cap.columns:
                    df_cap['Date'] = pd.to_datetime(df_cap['Date']).dt.date
                    df_fund_combined = df_cap[['Date', 'MarketCap', 'ListedShares']]

            df_fundamental_raw = pykrx_stock.get_market_fundamental_by_date(start_date_str, end_date_str, stock_code)
            if not df_fundamental_raw.empty:
                df_pbr_per_eps = df_fundamental_raw.reset_index().rename(columns={'날짜': 'Date', 'DIV': 'CashDPS'}) # DIV는 현금배당률이므로 CashDPS로
                if 'Date' in df_pbr_per_eps.columns:
                    df_pbr_per_eps['Date'] = pd.to_datetime(df_pbr_per_eps['Date']).dt.date
                
                    fund_cols_to_merge = ['Date']
                    # pykrx에서 제공하는 컬럼명과 모델 필드명 매핑 (필요시 확장)
                    pykrx_to_model_fund_map = {'PBR':'PBR', 'PER':'PER', 'EPS':'EPS', 'BPS':'BPS', 'CashDPS':'CashDPS', 'DPS':'DPS'}
                    for pykrx_col, model_col in pykrx_to_model_fund_map.items():
                        if pykrx_col in df_pbr_per_eps.columns:
                            fund_cols_to_merge.append(pykrx_col) # 실제 pykrx 컬럼명 사용
                    
                    if len(fund_cols_to_merge) > 1: # 'Date' 외에 다른 컬럼이 있을 때만 병합
                        if not df_fund_combined.empty:
                            df_fund_combined = pd.merge(df_fund_combined, df_pbr_per_eps[fund_cols_to_merge], on='Date', how='outer')
                        else:
                            df_fund_combined = df_pbr_per_eps[fund_cols_to_merge]
            
            if not df_fund_combined.empty:
                # 최종적으로 모델 필드명에 맞게 컬럼명 변경 (이미 pykrx_to_model_fund_map에서 처리되었을 수 있음)
                final_fund_rename_map = {'PBR':'PBR', 'PER':'PER', 'EPS':'EPS', 'BPS':'BPS', 'CashDPS':'CashDPS', 'DPS':'DPS', 'MarketCap':'MarketCap'}
                df_fund_combined.rename(columns={k:v for k,v in final_fund_rename_map.items() if k in df_fund_combined.columns}, inplace=True)

                df_fund_combined.set_index('Date', inplace=True)
                
                # 숫자형 변환 및 ffill/bfill
                for col in ['MarketCap', 'ListedShares', 'PBR', 'PER', 'EPS', 'BPS', 'CashDPS', 'DPS']:
                    if col in df_fund_combined.columns:
                        df_fund_combined[col] = pd.to_numeric(df_fund_combined[col], errors='coerce')
                        df_fund_combined[col] = df_fund_combined[col].ffill().bfill() # 연속성 확보
            
                # ROE 계산 (EPS와 BPS가 모두 유효할 때)
                if 'EPS' in df_fund_combined.columns and 'BPS' in df_fund_combined.columns:
                    df_fund_combined['ROE'] = np.where(
                        (df_fund_combined['BPS'].notna()) & (df_fund_combined['BPS'] != 0),
                        (df_fund_combined['EPS'] / df_fund_combined['BPS']) * 100,
                        np.nan # BPS가 0이거나 NaN이면 ROE도 NaN
                    )
                    df_fund_combined['ROE'] = df_fund_combined['ROE'].ffill().bfill()
                else:
                    df_fund_combined['ROE'] = np.nan # EPS 또는 BPS가 없으면 ROE도 NaN
            return df_fund_combined.reset_index() # Date 컬럼을 다시 일반 컬럼으로
        except Exception as e:
            self.stderr.write(self.style.WARNING(f"      pykrx fundamental fetch error for {stock_code} ({start_date_str}~{end_date_str}): {e}\n{traceback.format_exc()}"))
            # 오류 발생 시 빈 DataFrame 또는 예상 컬럼을 가진 DataFrame 반환
            fund_cols = ['Date', 'MarketCap', 'PBR', 'PER', 'EPS', 'BPS', 'DPS', 'CashDPS', 'ROE'] # 예상 컬럼
            return pd.DataFrame(columns=fund_cols)

    def save_market_index_data(self, market_name_param, df_market_data_full, kr_holidays_list_for_index):
        if df_market_data_full is None or df_market_data_full.empty:
            self.stdout.write(self.style.NOTICE(f"    No data to save for MarketIndex: {market_name_param}"))
            return 0

        saved_count = 0
        
        # 인덱스가 datetime.date 타입이 아니면 변환 시도
        if not all(isinstance(i, date_type) for i in df_market_data_full.index if pd.notna(i)):
            try:
                df_market_data_full.index = pd.to_datetime(df_market_data_full.index, errors='coerce').date
            except Exception as e_idx:
                self.stderr.write(self.style.ERROR(f"MarketIndex data for {market_name_param}: Date index conversion failed ({e_idx}). Skipping save."))
                return 0
        
        df_to_save = df_market_data_full.copy()
        
        # pykrx에서 반환하는 컬럼명 기준 (예: '종가', '거래량', '거래대금')
        close_col_name_in_df = '종가' # 또는 'Close' 등 실제 FDR/pykrx 반환값 확인 필요
        volume_col_name_in_df = '거래량' # 또는 'Volume'
        value_col_name_in_df = '거래대금'   # 또는 'Value'
        
        if close_col_name_in_df not in df_to_save.columns:
            self.stderr.write(self.style.ERROR(f"MarketIndex data for {market_name_param}: Missing '{close_col_name_in_df}' column. pykrx may have changed column names. Available: {df_to_save.columns.tolist()}"))
            return 0

        with transaction.atomic():
            for date_idx, row_data in df_to_save.iterrows():
                if not isinstance(date_idx, date_type): continue # 유효한 date 객체만 처리

                current_close_price = row_data.get(close_col_name_in_df)
                if pd.isna(current_close_price): continue # 종가 없으면 저장 안 함

                defaults = {
                    'close_price': float(current_close_price),
                    'volume': float(row_data.get(volume_col_name_in_df)) if volume_col_name_in_df in row_data and pd.notna(row_data.get(volume_col_name_in_df)) else None,
                    'trade_value': float(row_data.get(value_col_name_in_df)) if value_col_name_in_df in row_data and pd.notna(row_data.get(value_col_name_in_df)) else None,
                }
                
                # 전일 종가 계산
                prev_day_for_lookup = self.get_previous_trading_day(date_idx, kr_holidays_list_for_index, 1)
                prev_market_record = MarketIndex.objects.filter(market_name=market_name_param, date=prev_day_for_lookup).first()
                
                if prev_market_record and prev_market_record.close_price is not None:
                    defaults['previous_day_close_price'] = prev_market_record.close_price
                elif prev_day_for_lookup in df_to_save.index and pd.notna(df_to_save.loc[prev_day_for_lookup, close_col_name_in_df]):
                     defaults['previous_day_close_price'] = float(df_to_save.loc[prev_day_for_lookup, close_col_name_in_df])
                else:
                    defaults['previous_day_close_price'] = None # 전일 종가 없으면 None
                
                try:
                    obj, created = MarketIndex.objects.update_or_create(
                        market_name=market_name_param,
                        date=date_idx,
                        defaults=defaults
                    )
                    if created:
                        saved_count += 1
                except Exception as e_db_save:
                    self.stderr.write(self.style.ERROR(f"  MarketIndex DB Save Error for {market_name_param} on {date_idx}: {e_db_save}"))
        
        if saved_count > 0:
            self.stdout.write(self.style.SUCCESS(f"    Successfully saved/updated {saved_count} records for MarketIndex: {market_name_param}."))
        return saved_count


    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS(f"Starting daily data update for DB (min history: {self.MIN_RECORDS_FOR_SUFFICIENT_HISTORY} days)..."))

        today_dt = timezone.now().date() # 현재 날짜 (스크립트 실행 시점)
        current_year = today_dt.year
        # 과거 10년치 + 미래 2년치 공휴일 정보 로드 (넉넉하게)
        all_years_for_holidays = list(range(current_year - 11, current_year + 3)) 
        kr_holidays_list_global = get_kr_holidays(all_years_for_holidays)
        
        # --- 데이터 수집 기준일 설정 수정 ---
        # 기존: 하루 전 거래일
        # target_date_for_data = self.get_previous_trading_day(today_dt, kr_holidays_list_global, 1)
        # 변경: 스크립트 실행일(today_dt) 자체가 거래일이면 해당 날짜, 아니면 그 이전 최신 거래일
        target_date_for_data = self.get_previous_trading_day(today_dt, kr_holidays_list_global, 0)
        self.stdout.write(f"Target date for data processing (attempting up to current day if trading day): {target_date_for_data}")

        # pykrx용 날짜 형식 (YYYYMMDD)
        overall_fetch_start_date_str_pykrx = (target_date_for_data - timedelta(days=10*365)).strftime("%Y%m%d")
        target_date_str_pykrx = target_date_for_data.strftime("%Y%m%d")

        # MarketIndex 데이터 (KOSPI, KOSDAQ 지수) 업데이트
        if PYKRX_AVAILABLE:
            self.stdout.write(f"Pre-fetching KOSPI market data from {overall_fetch_start_date_str_pykrx} to {target_date_str_pykrx} for MarketIndex model using pykrx...")
            try:
                # pykrx.stock.get_index_ohlcv_by_date(fromdate, todate, ticker) 사용
                df_kospi_index_raw = pykrx_stock.get_index_ohlcv_by_date(overall_fetch_start_date_str_pykrx, target_date_str_pykrx, "1001") # 1001: KOSPI
                if df_kospi_index_raw is not None and not df_kospi_index_raw.empty:
                    self.save_market_index_data('KOSPI', df_kospi_index_raw, kr_holidays_list_global)
            except Exception as e_pykrx_idx:
                self.stderr.write(self.style.ERROR(f"Error fetching KOSPI index data from pykrx: {e_pykrx_idx}"))
        else:
            self.stdout.write(self.style.WARNING("pykrx not available, skipping KOSPI index data fetch for MarketIndex."))

        if PYKRX_AVAILABLE:
            self.stdout.write(f"Pre-fetching KOSDAQ market data from {overall_fetch_start_date_str_pykrx} to {target_date_str_pykrx} for MarketIndex model using pykrx...")
            try:
                df_kosdaq_index_raw = pykrx_stock.get_index_ohlcv_by_date(overall_fetch_start_date_str_pykrx, target_date_str_pykrx, "2001") # 2001: KOSDAQ
                if df_kosdaq_index_raw is not None and not df_kosdaq_index_raw.empty:
                    self.save_market_index_data('KOSDAQ', df_kosdaq_index_raw, kr_holidays_list_global)
            except Exception as e_pykrx_idx:
                 self.stderr.write(self.style.ERROR(f"Error fetching KOSDAQ index data from pykrx: {e_pykrx_idx}"))
        else:
            self.stdout.write(self.style.WARNING("pykrx not available, skipping KOSDAQ index data fetch for MarketIndex."))

        # StockPrice 모델에 사용될 시장/거시경제 데이터 (utils.get_market_macro_data 사용)
        # 이 함수는 내부적으로 FDR을 사용하며, YYYY-MM-DD 형식의 날짜를 사용
        overall_fetch_start_date_str_util = (target_date_for_data - timedelta(days=10*365)).strftime("%Y-%m-%d") # YYYY-MM-DD
        target_date_str_util = target_date_for_data.strftime("%Y-%m-%d") # YYYY-MM-DD

        self.stdout.write(f"Pre-fetching KOSPI market & macro data (for StockPrice) from {overall_fetch_start_date_str_util} to {target_date_str_util}...")
        df_kospi_macro_full_for_stockprice = get_market_macro_data(overall_fetch_start_date_str_util, target_date_str_util, market_name='KOSPI', other_market_name_for_index='KOSDAQ')
        
        self.stdout.write(f"Pre-fetching KOSDAQ market & macro data (for StockPrice) from {overall_fetch_start_date_str_util} to {target_date_str_util}...")
        df_kosdaq_macro_full_for_stockprice = get_market_macro_data(overall_fetch_start_date_str_util, target_date_str_util, market_name='KOSDAQ', other_market_name_for_index='KOSPI')

        # --- KRX 주식 목록 가져오기 방식 변경 ---
        # 기존: krx_listing_df = fdr.StockListing('KRX')
        #       krx_listing_df = krx_listing_df[krx_listing_df['Market'].isin(['KOSPI', 'KOSDAQ'])]
        # 변경: utils.get_krx_stock_list 사용
        self.stdout.write(f"Fetching KOSPI and KOSDAQ stock list using standardized method...")
        all_stocks_to_process_list = get_krx_stock_list(market='KOSPI,KOSDAQ')
        
        if not all_stocks_to_process_list:
            self.stderr.write(self.style.ERROR("Failed to fetch any stocks from KOSPI/KOSDAQ using get_krx_stock_list. Exiting."))
            return

        total_stocks_to_process = len(all_stocks_to_process_list)
        self.stdout.write(f"Found {total_stocks_to_process} stocks in KOSPI/KOSDAQ (standardized). Starting data update/backfill for StockPrice DB...")
        processed_count, skipped_count, error_count = 0, 0, 0

        # --- 반복문 변경: DataFrame 순회에서 List of Dicts 순회로 ---
        for idx, stock_info in enumerate(all_stocks_to_process_list): # row_listing 대신 stock_info 사용
            stock_code = stock_info['Code']
            stock_name = stock_info['Name']
            market_name_str = stock_info['Market'].upper() # 이미 'KOSPI' 또는 'KOSDAQ'으로 표준화됨

            if (idx + 1) % 20 == 0: # 20개 종목마다 진행 상황 출력
                self.stdout.write(f"   [{market_name_str}] Checked {(idx + 1)}/{total_stocks_to_process} stocks... (OK: {processed_count}, Skip: {skipped_count}, Err: {error_count})")

            # DB에서 해당 종목의 마지막 데이터 날짜 확인
            last_stock_date_in_db_obj = StockPrice.objects.filter(stock_code=stock_code).order_by('-date').first()
            num_records_in_db = StockPrice.objects.filter(stock_code=stock_code).count()

            fetch_start_date_stock_specific_dt = None
            if last_stock_date_in_db_obj: # DB에 데이터가 있는 경우
                # 이미 최신 데이터(target_date_for_data)까지 있고, 충분한 기록이 있으면 건너뛰기
                if last_stock_date_in_db_obj.date >= target_date_for_data and num_records_in_db >= self.MIN_RECORDS_FOR_SUFFICIENT_HISTORY:
                    # self.stdout.write(self.style.NOTICE(f"    Data for {stock_name} ({stock_code}) is up-to-date ({last_stock_date_in_db_obj.date}). Skipping."))
                    skipped_count += 1
                    continue
                
                # 새 데이터 가져올 시작 날짜: DB 마지막 날짜 + 1일
                fetch_start_date_stock_specific_dt = last_stock_date_in_db_obj.date + timedelta(days=1)
                
                # 만약 기록이 부족하거나, 업데이트할 기간이 너무 짧으면 과거 데이터부터 다시 가져와서 채움 (백필)
                if num_records_in_db < self.MIN_RECORDS_FOR_SUFFICIENT_HISTORY or \
                   (target_date_for_data - fetch_start_date_stock_specific_dt).days < self.DAYS_TO_FETCH_FOR_BACKFILL: # 최소 백필 기간
                    fetch_start_date_stock_specific_dt = self.get_previous_trading_day(target_date_for_data, kr_holidays_list_global, self.DAYS_TO_FETCH_OHLCV_BASE)
            else: # DB에 데이터가 전혀 없는 경우 (신규 상장 등)
                # 과거 데이터부터 가져옴
                fetch_start_date_stock_specific_dt = self.get_previous_trading_day(target_date_for_data, kr_holidays_list_global, self.DAYS_TO_FETCH_OHLCV_BASE)

            # 가져올 데이터의 시작일이 목표일보다 미래면 건너뛰기 (이미 처리된 것으로 간주)
            if fetch_start_date_stock_specific_dt > target_date_for_data:
                skipped_count += 1
                continue

            # FDR 및 pykrx용 날짜 문자열 형식 변환
            fetch_start_date_stock_specific_str_fdr = fetch_start_date_stock_specific_dt.strftime("%Y-%m-%d") # FDR용
            target_date_stock_specific_str_fdr = target_date_for_data.strftime("%Y-%m-%d") # FDR용
            
            fetch_start_date_stock_specific_str_pykrx = fetch_start_date_stock_specific_dt.strftime("%Y%m%d") # pykrx용
            target_date_stock_specific_str_pykrx = target_date_for_data.strftime("%Y%m%d") # pykrx용


            try:
                # 1. FinanceDataReader로 OHLCV 데이터 가져오기
                df_stock_ohlcv_fdr = fdr.DataReader(stock_code, fetch_start_date_stock_specific_str_fdr, target_date_stock_specific_str_fdr)
                if df_stock_ohlcv_fdr.empty:
                    self.stdout.write(self.style.NOTICE(f"    No data from FDR for {stock_name} ({stock_code}) for period {fetch_start_date_stock_specific_str_fdr}-{target_date_stock_specific_str_fdr}. Skipping."))
                    skipped_count +=1
                    continue

                df_stock_ohlcv_fdr.index = pd.to_datetime(df_stock_ohlcv_fdr.index).date # 인덱스를 date 객체로
                df_stock_ohlcv_fdr.sort_index(inplace=True)

                # calculate_all_features 함수에 전달할 기본 OHLCV DataFrame 준비
                stock_df_ohlcv_arg = df_stock_ohlcv_fdr[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                if 'Change' in df_stock_ohlcv_fdr.columns: # FDR이 Change를 제공하면 사용
                    stock_df_ohlcv_arg['Change'] = pd.to_numeric(df_stock_ohlcv_fdr['Change'], errors='coerce')
                else: # 제공하지 않으면 직접 계산
                    stock_df_ohlcv_arg['Change'] = stock_df_ohlcv_fdr['Close'].pct_change(fill_method=None) # 첫날은 NaN

                # 2. 투자자별 매매동향 데이터 (fdr 또는 pykrx)
                investor_df_arg = pd.DataFrame(index=df_stock_ohlcv_fdr.index.copy()) # OHLCV와 동일한 인덱스
                
                # FDR에서 투자자별 거래량(금액) 가져오기 시도 (컬럼명 확인 필요)
                possible_investor_cols_fdr = {
                    'Indi': ['개인', 'Individual', 'Retail'], # 개인 순매수 관련 컬럼명 후보
                    'Foreign': ['외국인', 'Foreigner', 'Forn'], # 외국인 순매수
                    'Organ': ['기관', '기관계', 'Institution', 'Inst'] # 기관 순매수
                }
                
                fdr_investor_data_found_at_least_one_type = False
                for model_col_prefix, fdr_name_options in possible_investor_cols_fdr.items():
                    found_this_type = False
                    for fdr_name in fdr_name_options:
                        if fdr_name in df_stock_ohlcv_fdr.columns: # fdr.DataReader가 반환한 DataFrame에 해당 컬럼이 있는지 확인
                            investor_df_arg[model_col_prefix] = pd.to_numeric(df_stock_ohlcv_fdr[fdr_name], errors='coerce')
                            fdr_investor_data_found_at_least_one_type = True 
                            found_this_type = True
                            break # 해당 타입의 첫 번째 유효한 컬럼 사용
                    if not found_this_type: # FDR에서 해당 타입의 컬럼을 못 찾았으면 NaN으로 채움
                         investor_df_arg[model_col_prefix] = np.nan
                
                # 만약 FDR에서 충분한 투자자 데이터를 얻지 못했다면 pykrx 시도
                should_try_pykrx_investor = PYKRX_AVAILABLE and \
                                           (not fdr_investor_data_found_at_least_one_type or \
                                            (investor_df_arg['Indi'].isnull().all() if 'Indi' in investor_df_arg else True) or \
                                            (investor_df_arg['Foreign'].isnull().all() if 'Foreign' in investor_df_arg else True) or \
                                            (investor_df_arg['Organ'].isnull().all() if 'Organ' in investor_df_arg else True) )


                if should_try_pykrx_investor:
                    self.stdout.write(self.style.NOTICE(f"    FDR for {stock_name} ({stock_code}) did not provide sufficient investor data. Trying pykrx for trading value..."))
                    try:
                        # pykrx.stock.get_market_trading_value_by_date 사용 (순매수 금액 기준)
                        df_investor_pykrx_raw = pykrx_stock.get_market_trading_value_by_date(
                            fromdate=fetch_start_date_stock_specific_str_pykrx, 
                            todate=target_date_stock_specific_str_pykrx, 
                            ticker=stock_code,
                            on='순매수' # 또는 '매수', '매도' 등 상세 옵션 확인
                        )
                        if not df_investor_pykrx_raw.empty:
                            df_investor_pykrx_temp = df_investor_pykrx_raw.reset_index()
                            # '날짜' 컬럼명 확인 및 변환
                            if '날짜' in df_investor_pykrx_temp.columns:
                                df_investor_pykrx_temp.rename(columns={'날짜':'Date'}, inplace=True)
                            elif df_investor_pykrx_temp.columns[0] != 'Date' and pd.api.types.is_datetime64_any_dtype(df_investor_pykrx_temp.iloc[:, 0]): # 첫번째 컬럼이 날짜형이면 Date로
                                df_investor_pykrx_temp.rename(columns={df_investor_pykrx_temp.columns[0]:'Date'}, inplace=True) 
                            
                            df_investor_pykrx_temp['Date'] = pd.to_datetime(df_investor_pykrx_temp['Date']).dt.date # date 객체로
                            
                            # investor_df_arg (FDR 기반)와 pykrx 데이터 병합 준비
                            if investor_df_arg.index.name == 'Date' and isinstance(investor_df_arg.index, pd.DatetimeIndex): # 인덱스가 DatetimeIndex면
                                investor_df_arg_for_merge = investor_df_arg.reset_index()
                                investor_df_arg_for_merge['Date'] = pd.to_datetime(investor_df_arg_for_merge['Date']).dt.date
                            elif 'Date' in investor_df_arg.columns and pd.api.types.is_datetime64_any_dtype(investor_df_arg['Date']): # 'Date' 컬럼이 있으면
                                investor_df_arg_for_merge = investor_df_arg.copy()
                                investor_df_arg_for_merge['Date'] = pd.to_datetime(investor_df_arg_for_merge['Date']).dt.date
                            else: # 그 외 (예: 인덱스가 이미 date 객체인 경우)
                                investor_df_arg_for_merge = investor_df_arg.copy()
                                if isinstance(investor_df_arg.index, pd.Index) and all(isinstance(x, date_type) for x in investor_df_arg.index): # 인덱스가 date 객체면
                                     investor_df_arg_for_merge.reset_index(inplace=True) # 'Date' 컬럼 생성
                                     if 'index' in investor_df_arg_for_merge.columns and 'Date' not in investor_df_arg_for_merge.columns: # 컬럼명 변경
                                         investor_df_arg_for_merge.rename(columns={'index':'Date'}, inplace=True)
                                elif 'Date' not in investor_df_arg_for_merge.columns: # 'Date' 컬럼이 아예 없으면
                                     investor_df_arg_for_merge['Date'] = investor_df_arg.index # 인덱스를 'Date'로 사용

                            if 'Date' in investor_df_arg_for_merge.columns: # Date 컬럼 타입 최종 확인
                                investor_df_arg_for_merge['Date'] = pd.to_datetime(investor_df_arg_for_merge['Date']).dt.date
                            else:
                                self.stderr.write(self.style.WARNING(f"      'Date' column missing in investor_df_arg_for_merge for {stock_name} before pykrx merge. Using OHLCV index."))
                                investor_df_arg_for_merge['Date'] = df_stock_ohlcv_fdr.index # 최후의 수단

                            # pykrx 컬럼명과 모델 컬럼명 매핑 (순매수 금액 기준)
                            target_model_investor_cols = {
                                'Indi': ['개인'],
                                'Foreign': ['외국인합계', '외국인'], # 우선순위: 외국인합계 > 외국인
                                'Organ': ['기관합계', '기관']      # 우선순위: 기관합계 > 기관
                            }

                            for model_c, pykrx_source_options in target_model_investor_cols.items():
                                pykrx_actual_source_col = None
                                for source_opt in pykrx_source_options: # pykrx가 반환하는 컬럼 중 우선순위 높은 것 선택
                                    if source_opt in df_investor_pykrx_temp.columns:
                                        pykrx_actual_source_col = source_opt
                                        break 
                                if pykrx_actual_source_col:
                                    df_pykrx_specific_investor = df_investor_pykrx_temp[['Date', pykrx_actual_source_col]].rename(
                                        columns={pykrx_actual_source_col: model_c} # 모델 컬럼명으로 변경
                                    )
                                    if model_c not in investor_df_arg_for_merge.columns: # FDR 데이터에 없던 컬럼이면 추가
                                        investor_df_arg_for_merge[model_c] = np.nan

                                    # pykrx 데이터를 FDR 데이터에 병합 (pykrx 우선)
                                    investor_df_arg_for_merge = pd.merge(
                                        investor_df_arg_for_merge,
                                        df_pykrx_specific_investor,
                                        on='Date',
                                        how='left',
                                        suffixes=('_fdr', '_pykrx') # 중복 컬럼명 처리
                                    )
                                    
                                    pykrx_suffixed_col = f"{model_c}_pykrx"
                                    fdr_suffixed_col = f"{model_c}_fdr"
                                    original_col_to_update = fdr_suffixed_col if fdr_suffixed_col in investor_df_arg_for_merge.columns else model_c
                                    
                                    if pykrx_suffixed_col in investor_df_arg_for_merge.columns:
                                        # pykrx 데이터가 있으면 사용, 없으면 FDR 데이터 사용 (combine_first)
                                        investor_df_arg_for_merge[model_c] = investor_df_arg_for_merge[pykrx_suffixed_col].combine_first(
                                            investor_df_arg_for_merge[original_col_to_update]
                                        )
                                        # 임시 컬럼 삭제
                                        cols_to_drop_after_merge = [pykrx_suffixed_col]
                                        if fdr_suffixed_col in investor_df_arg_for_merge.columns and fdr_suffixed_col != model_c:
                                            cols_to_drop_after_merge.append(fdr_suffixed_col)
                                        investor_df_arg_for_merge.drop(columns=cols_to_drop_after_merge, errors='ignore', inplace=True)
                                    elif fdr_suffixed_col in investor_df_arg_for_merge.columns and fdr_suffixed_col != model_c : # pykrx는 없지만 fdr suffix가 붙은 경우
                                         investor_df_arg_for_merge.rename(columns={fdr_suffixed_col: model_c}, inplace=True, errors='ignore')
                            
                            # 병합 후 인덱스 재설정
                            if 'Date' in investor_df_arg_for_merge.columns:
                                investor_df_arg = investor_df_arg_for_merge.set_index('Date')
                            else:
                                self.stderr.write(self.style.ERROR(f"      Investor data merge for {stock_name} resulted in missing 'Date' column after pykrx attempt."))
                                # 오류 시 기존 investor_df_arg 유지 (pykrx 데이터 손실)

                            # 최종적으로 모델에서 사용할 컬럼이 없으면 NaN으로 채움
                            for model_col_final in ['Indi', 'Foreign', 'Organ']:
                                if model_col_final not in investor_df_arg.columns:
                                    investor_df_arg[model_col_final] = np.nan

                            self.stdout.write(self.style.SUCCESS(f"    Successfully attempted to update investor data for {stock_name} using pykrx (value based)."))
                        else:
                            self.stdout.write(self.style.NOTICE(f"    pykrx (value based) returned no investor data for {stock_name} ({stock_code})."))
                    except AttributeError as e_attr: # pykrx 함수가 없을 경우 (버전 등)
                         self.stderr.write(self.style.ERROR(f"      AttributeError for pykrx investor data for {stock_name}: {e_attr}. pykrx function 'get_market_trading_value_by_date' might not be available or was misspelled."))
                    except Exception as e_pykrx_inv:
                        self.stderr.write(self.style.ERROR(f"      Error fetching/processing investor data from pykrx for {stock_name}: {e_pykrx_inv}\n{traceback.format_exc()}"))

                
                # 3. 펀더멘털 데이터 (pykrx 사용)
                #    실제 데이터 가져온 기간에 맞춰서 요청
                actual_stock_data_start_date_str = df_stock_ohlcv_fdr.index.min().strftime("%Y%m%d")
                actual_stock_data_end_date_str = df_stock_ohlcv_fdr.index.max().strftime("%Y%m%d")
                
                fundamental_df_from_pykrx = self.fetch_pykrx_fundamentals(stock_code, actual_stock_data_start_date_str, actual_stock_data_end_date_str)
                fundamental_df_arg = pd.DataFrame() # 기본값은 빈 DataFrame
                if not fundamental_df_from_pykrx.empty and 'Date' in fundamental_df_from_pykrx.columns:
                    fundamental_df_arg = fundamental_df_from_pykrx.set_index('Date') # Date를 인덱스로
                    if isinstance(fundamental_df_arg.index, pd.DatetimeIndex): # 인덱스가 DatetimeIndex면 date 객체로
                         fundamental_df_arg.index = fundamental_df_arg.index.date

                # 4. 시장/거시경제 데이터 (미리 로드한 데이터에서 현재 종목 기간에 맞게 필터링/조인)
                market_macro_data_df_arg = pd.DataFrame(index=df_stock_ohlcv_fdr.index) # OHLCV와 동일한 인덱스
                # <<< market_name_str은 이제 표준화된 "KOSPI" 또는 "KOSDAQ" 임 >>>
                current_stock_market_macro_df_full = df_kospi_macro_full_for_stockprice if market_name_str == 'KOSPI' else df_kosdaq_macro_full_for_stockprice
                
                if not current_stock_market_macro_df_full.empty:
                    # 인덱스 타입 통일 (date 객체)
                    if not isinstance(current_stock_market_macro_df_full.index, pd.DatetimeIndex) and \
                       not all(isinstance(i, date_type) for i in current_stock_market_macro_df_full.index if pd.notna(i)):
                        current_stock_market_macro_df_full.index = pd.to_datetime(current_stock_market_macro_df_full.index, errors='coerce').date

                    market_macro_data_df_arg = market_macro_data_df_arg.join(current_stock_market_macro_df_full, how='left')
                
                # 5. 모든 피처 계산 (utils.calculate_all_features 호출)
                df_features_from_calc = calculate_all_features(
                    stock_df_ohlcv=stock_df_ohlcv_arg,
                    market_macro_data_df=market_macro_data_df_arg,
                    investor_df=investor_df_arg,
                    fundamental_df=fundamental_df_arg, # pykrx에서 가져온 펀더멘털 데이터 전달
                    pandas_ta_available=PANDAS_TA_AVAILABLE
                )
                
                if df_features_from_calc is None or df_features_from_calc.empty:
                    self.stdout.write(self.style.WARNING(f"    No features calculated for {stock_name} ({stock_code}). Skipping DB save for this stock."))
                    skipped_count +=1
                    continue

                # 펀더멘털 기반 파생 플래그 컬럼 추가 (예: PER_is_nan 등)
                df_final_for_db = add_fundamental_indicator_features(df_features_from_calc.copy())
                df_final_for_db.replace([np.inf, -np.inf], np.nan, inplace=True) # 무한대 값 NaN으로

                # DB 저장을 위한 컬럼 매핑 (모델 필드명 <-> DataFrame 컬럼명)
                model_to_df_col_map = {
                    # OHLCV 및 기본
                    'open_price': 'Open', 'high_price': 'High', 'low_price': 'Low', 'close_price': 'Close',
                    'volume': 'Volume', 'change': 'Change', # 'change'는 소수점 등락률
                    
                    # 투자자별 (컬럼명이 Indi, Foreign, Organ 등으로 통일되어 있다고 가정)
                    'indi_volume': 'Indi', 'foreign_volume': 'Foreign', 'organ_volume': 'Organ',
                    
                    # 펀더멘털 (MarketCap, PBR, PER 등은 calculate_all_features 결과에 포함 가정)
                    'market_cap': 'MarketCap', 'per': 'PER', 'pbr': 'PBR',
                    'eps': 'EPS', 'bps': 'BPS', 'dps': 'DPS', # utils.py에서 계산 또는 pykrx에서 가져옴
                    'roe': 'ROE', # utils.py에서 계산 또는 pykrx에서 가져옴

                    # 기술적 지표 (utils.calculate_technical_indicators에서 생성된 컬럼명 그대로 사용)
                    'MA5': 'MA_5', 'MA10': 'MA_10', 'MA20': 'MA_20', 'MA60': 'MA_60', 'MA120': 'MA_120',
                    'EMA5': 'EMA_5', 'EMA10': 'EMA_10', 'EMA20': 'EMA_20', 'EMA60': 'EMA_60', 'EMA120': 'EMA_120',
                    
                    'BB_Upper': 'BBU_20_2.0', # pandas_ta 기본 컬럼명
                    'BB_Middle': 'BBM_20_2.0',
                    'BB_Lower': 'BBL_20_2.0',
                    'BB_Width': 'BBB_20_2.0', # 볼린저 밴드 폭
                    'BB_PercentB': 'BBP_20_2.0', # %B
                    # ... (views.py의 KOSPI_TECH_LSTM_FEATURES 목록과 유사하게 모든 TA 지표 컬럼 추가) ...
                    'MACD': 'MACD_12_26_9', 'MACD_Signal': 'MACDs_12_26_9', 'MACD_Hist': 'MACDh_12_26_9',
                    'RSI6': 'RSI_6', 'RSI14': 'RSI_14', 'RSI28': 'RSI_28',
                    'STOCH_K': 'STOCHk_14_3_3', 'STOCH_D': 'STOCHd_14_3_3', 
                    'STOCH_SLOW_K': 'STOCHk_fast_14_3_1', # pandas_ta 컬럼명 확인 필요 (STOCHk_14_3_1 등)
                    'STOCH_SLOW_D': 'STOCHd_fast_14_3_1', # pandas_ta 컬럼명 확인 필요 (STOCHd_14_3_1 등)

                    'ATR14': 'ATR_14', 'ADX14': 'ADX_14', 'DMP14': 'DMP_14', 'DMN14': 'DMN_14', 
                    'CCI14': 'CCI_14_0.015', 'MFI14': 'MFI_14', 'OBV': 'OBV', 
                    'WilliamsR14': 'WILLR_14', 'Momentum': 'MOM_10', 'ROC': 'ROC_10', 
                    'TRIX': 'TRIX_14_9', 'VR': 'VR_20', 'PSY': 'PSL_12',

                    # 시장/거시경제 (utils.get_market_macro_data 결과 컬럼명)
                    # <<< market_name_str은 이제 표준화된 "KOSPI" 또는 "KOSDAQ" 임 >>>
                    'Market_Index_Close': f"{market_name_str}_Close", 
                    'Market_Index_Change': f"{market_name_str}_Change",
                    'USD_KRW_Close': 'USD_KRW_Close',
                    'USD_KRW_Change': 'USD_KRW_Change',

                    # 로그 변환 값
                    'log_close_price': 'Log_Close',
                    'log_volume': 'Log_Volume',
                    
                    # 펀더멘털 파생 플래그 (add_fundamental_indicator_features에서 생성)
                    'per_is_high': 'PER_is_high', 
                    'per_is_low': 'PER_is_low',   
                    'per_is_zero': 'PER_is_zero',
                    'per_is_nan': 'PER_is_nan',
                    'pbr_is_high': 'PBR_is_high', 
                    'pbr_is_low': 'PBR_is_low',   
                    'pbr_is_zero': 'PBR_is_zero', 
                    'pbr_is_nan': 'PBR_is_nan',
                    'market_cap_is_nan': 'MarketCap_is_nan',
                }

                saved_count_for_stock_this_fetch = 0
                with transaction.atomic():
                    # df_final_for_db의 인덱스가 date 객체가 아니면 변환
                    if not isinstance(df_final_for_db.index, pd.DatetimeIndex) and \
                       not all(isinstance(i, date_type) for i in df_final_for_db.index if pd.notna(i)):
                        try:
                            df_final_for_db.index = pd.to_datetime(df_final_for_db.index, errors='coerce').date
                        except: 
                             self.stderr.write(self.style.ERROR(f"      Date index conversion failed for {stock_name} ({stock_code}). Skipping DB save."))
                             error_count +=1
                             continue # 다음 종목으로


                    for date_iter_dt, row_final_data in df_final_for_db.iterrows():
                        if not isinstance(date_iter_dt, date_type): # 유효한 date 객체가 아니면 건너뛰기
                            # self.stderr.write(self.style.WARNING(f"Invalid date index found: {date_iter_dt} for {stock_name}. Skipping this row."))
                            continue
                        
                        current_iter_date_for_db = date_iter_dt # date 객체
                        # <<< market_name_str은 이제 표준화된 "KOSPI" 또는 "KOSDAQ" 임 >>>
                        defaults_for_db = {'stock_name': stock_name, 'market_name': market_name_str}

                        # 모델 필드에 맞춰 데이터 할당
                        for model_field_name, df_col_name in model_to_df_col_map.items():
                            try:
                                model_field = StockPrice._meta.get_field(model_field_name)
                            except FieldDoesNotExist:
                                # self.stdout.write(self.style.NOTICE(f"Model field '{model_field_name}' does not exist. Skipping."))
                                continue # 모델에 없는 필드면 건너뜀

                            if df_col_name in row_final_data:
                                val_from_df = row_final_data[df_col_name]
                                # NaN, Inf 값은 None으로 변환하여 DB에 NULL로 저장되도록 함
                                if pd.isna(val_from_df) or (isinstance(val_from_df, float) and (np.isinf(val_from_df) or np.isnan(val_from_df))): 
                                    defaults_for_db[model_field_name] = None
                                else:
                                    # 타입에 맞게 변환
                                    if isinstance(model_field, (django_db_models.FloatField, django_db_models.DecimalField)):
                                        defaults_for_db[model_field_name] = float(val_from_df)
                                    elif isinstance(model_field, (django_db_models.IntegerField, django_db_models.BigIntegerField)):
                                        # 거래량, 시가총액 등 큰 숫자는 float으로 온 후 int로 변환될 수 있음
                                        if model_field_name in ['indi_volume', 'foreign_volume', 'organ_volume', 'market_cap', 'volume']: 
                                            try:
                                                defaults_for_db[model_field_name] = int(float(val_from_df)) # 소수점 버림
                                            except (ValueError, TypeError):
                                                defaults_for_db[model_field_name] = None # 변환 실패 시 None
                                        else:
                                            defaults_for_db[model_field_name] = int(round(float(val_from_df))) # 반올림 후 정수
                                    elif isinstance(model_field, django_db_models.BooleanField):
                                        defaults_for_db[model_field_name] = bool(val_from_df)
                                    else: # CharField 등 기타
                                        defaults_for_db[model_field_name] = val_from_df
                            else:
                                defaults_for_db[model_field_name] = None # DataFrame에 해당 컬럼 없으면 None
                        
                        # 전일 종가 계산 (DB 또는 현재 DataFrame에서 조회)
                        prev_day_for_db_lookup = self.get_previous_trading_day(current_iter_date_for_db, kr_holidays_list_global, 1)
                        prev_day_stock_record_db = StockPrice.objects.filter(stock_code=stock_code, date=prev_day_for_db_lookup).first()
                        
                        if prev_day_stock_record_db and prev_day_stock_record_db.close_price is not None:
                            defaults_for_db['previous_day_close_price'] = prev_day_stock_record_db.close_price
                        else: # DB에 없으면 현재 처리중인 DataFrame에서 찾기
                            if prev_day_for_db_lookup in df_final_for_db.index: # 인덱스에 해당 날짜가 있는지 확인
                                prev_close_in_df = df_final_for_db.loc[prev_day_for_db_lookup, 'Close'] # 'Close' 컬럼 가정
                                if pd.notna(prev_close_in_df):
                                    defaults_for_db['previous_day_close_price'] = float(prev_close_in_df)
                                else:
                                    defaults_for_db['previous_day_close_price'] = None
                            else:
                                defaults_for_db['previous_day_close_price'] = None
                        
                        # 필수 OHLCV 값 중 하나라도 None이면 저장하지 않음 (데이터 품질 관리)
                        required_ohlcv_fields = ['open_price', 'high_price', 'low_price', 'close_price'] # volume은 제외 가능
                        can_save = True
                        for req_fld in required_ohlcv_fields:
                            if defaults_for_db.get(req_fld) is None: 
                                # self.stdout.write(self.style.WARNING(f"      Skipping save for {stock_name} on {current_iter_date_for_db} due to missing {req_fld}."))
                                can_save = False
                                break
                        if not can_save:
                            continue

                        try:
                            obj, created = StockPrice.objects.update_or_create(
                                stock_code=stock_code, date=current_iter_date_for_db,
                                defaults=defaults_for_db
                            )
                            if created:
                                saved_count_for_stock_this_fetch += 1
                        except Exception as e_db_save:
                            self.stderr.write(self.style.ERROR(f"      DB Save Error for {stock_name} ({stock_code}) on {current_iter_date_for_db}: {e_db_save}"))
                            problematic_data_log = {k: (type(v), v) for k, v in defaults_for_db.items()}
                            self.stderr.write(self.style.ERROR(f"        Problematic Defaults (type, value): {problematic_data_log}"))

                if saved_count_for_stock_this_fetch > 0:
                    self.stdout.write(self.style.SUCCESS(f"    Successfully saved/updated {saved_count_for_stock_this_fetch} records for {stock_name} ({stock_code}) up to {target_date_for_data}."))
                    processed_count +=1
                elif not df_final_for_db.empty : # 저장된 건 없지만 데이터는 있었던 경우 (이미 최신이거나, 업데이트할 내용이 없었음)
                    # self.stdout.write(self.style.NOTICE(f"    No new records saved for {stock_name} ({stock_code}), data might be current or no changes to update."))
                    skipped_count += 1 # 이미 최신이거나 업데이트할 내용이 없는 경우도 skipped로 처리
                else: # df_final_for_db가 비어있었던 경우 (FDR에서 데이터 못가져옴 등)
                    pass # 이미 위에서 skipped_count 처리됨

            except Exception as e_stock_fetch:
                self.stderr.write(self.style.ERROR(f"      Error processing/fetching data for {stock_name} ({stock_code}): {e_stock_fetch}\n{traceback.format_exc()}"))
                error_count +=1
            time.sleep(0.1) # API 호출 부하 분산

        self.stdout.write(self.style.SUCCESS(f"\nFinished processing all markets. Total Stocks: {total_stocks_to_process}, Successfully Processed/NewData: {processed_count}, Skipped (No new data/Already up-to-date/No FDR data/No features): {skipped_count}, Errors in processing: {error_count}"))
        self.stdout.write(self.style.SUCCESS(f'Daily DB data update/backfill process finished at {timezone.now()}.'))
