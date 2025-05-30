# predict_info/management/commands/update_daily_data.py
import pandas as pd
import FinanceDataReader as fdr
from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta, datetime, date # 'date'를 명시적으로 임포트
from predict_info.models import MarketIndex, StockPrice 
# from pandas.tseries.offsets import BDay # get_previous_trading_day에서 사용 안함
import holidays 
import time
import traceback
import numpy as np # np.nan 사용

class Command(BaseCommand):
    help = ('Fetches and stores daily stock data (OHLCV, Investor Trends, Fundamentals) '
            'into the database. Ensures a minimum history and updates up to the previous trading day.')

    MIN_RECORDS_FOR_SUFFICIENT_HISTORY = 60 
    DAYS_TO_FETCH_FOR_BACKFILL = 80 # MIN_RECORDS_FOR_SUFFICIENT_HISTORY 보다 크게

    def get_previous_trading_day(self, reference_date, days_offset=1):
        """지정된 날짜로부터 특정 거래일 수 이전의 날짜를 반환합니다."""
        if not isinstance(reference_date, date):
            try: reference_date = pd.to_datetime(reference_date).date()
            except: raise TypeError("reference_date must be a datetime.date or convertible to it.")
        
        # 휴일 정보는 매번 다시 계산 (연도가 바뀔 수 있으므로)
        # 충분한 과거/미래 연도를 포함하여 휴일 계산
        relevant_years = list(set([reference_date.year - (days_offset // 200 + 3), reference_date.year, reference_date.year + 2]))
        kr_holidays = holidays.KR(years=relevant_years)
        
        current_check_date = reference_date
        trading_days_found = 0
        
        # days_offset이 0이면 reference_date가 거래일인지 확인 후 반환 (또는 그 이전 거래일)
        if days_offset == 0:
            while current_check_date.weekday() >= 5 or current_check_date in kr_holidays:
                current_check_date -= timedelta(days=1)
            return current_check_date

        while trading_days_found < days_offset:
            current_check_date -= timedelta(days=1)
            if current_check_date.weekday() < 5 and current_check_date not in kr_holidays:
                trading_days_found += 1
        return current_check_date

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS(f"Starting daily data update for DB (min history: {self.MIN_RECORDS_FOR_SUFFICIENT_HISTORY} days)..."))

        today_dt = timezone.now()
        # 데이터 가져올 최종 목표 날짜 (어제 거래일)
        # 만약 오늘이 월요일이면, target_date는 금요일이 됨.
        target_date_for_data = self.get_previous_trading_day(today_dt, 1)
        
        self.stdout.write(f"Target date for data processing: {target_date_for_data}")

        # 1. 시장 지수 업데이트 (target_date_for_data 및 그 이전 거래일 데이터 필요)
        day_before_target_date_for_index = self.get_previous_trading_day(target_date_for_data, 1)
        indices_to_fetch = {'KOSPI': 'KS11', 'KOSDAQ': 'KQ11'}

        for market_readable_name, fdr_ticker in indices_to_fetch.items():
            if MarketIndex.objects.filter(market_name=market_readable_name, date=target_date_for_data).exists():
                self.stdout.write(self.style.SUCCESS(f"{market_readable_name} index data for {target_date_for_data} already exists. Skipping."))
                continue
            
            self.stdout.write(f"Fetching data for {market_readable_name} index ({fdr_ticker})...")
            try:
                # 시작일을 하루 더 이전으로 하여 전일 종가를 확실히 가져옴
                df_index = fdr.DataReader(fdr_ticker, start=self.get_previous_trading_day(day_before_target_date_for_index,1), end=target_date_for_data)
                if df_index.empty:
                    self.stdout.write(self.style.WARNING(f"No data for {market_readable_name} for dates up to {target_date_for_data}."))
                    continue
                
                df_index.index = pd.to_datetime(df_index.index).date
                
                data_target_date_row = df_index[df_index.index == target_date_for_data]
                data_day_before_row = df_index[df_index.index == day_before_target_date_for_index]

                if data_target_date_row.empty:
                    self.stdout.write(self.style.WARNING(f"No data for {market_readable_name} on target date {target_date_for_data}."))
                    continue
                
                current_data = data_target_date_row.iloc[0]
                prev_close_val = data_day_before_row.iloc[0]['Close'] if not data_day_before_row.empty else None
                
                MarketIndex.objects.update_or_create(
                    market_name=market_readable_name, 
                    date=target_date_for_data,
                    defaults={
                        'close_price': current_data['Close'],
                        'previous_day_close_price': prev_close_val, 
                        'volume': current_data.get('Volume'),
                        'trade_value': current_data.get('Amount'), # FDR에서 Amount가 거래대금
                        # change_value, change_percent는 save() 메소드에서 자동 계산됨
                    }
                )
                self.stdout.write(self.style.SUCCESS(f"Updated/Created {market_readable_name} index data for {target_date_for_data}."))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error fetching/saving {market_readable_name} index: {e}"))
                traceback.print_exc()
            time.sleep(0.5) # API 호출 간격

        # 2. 전체 상장 종목의 최신 펀더멘털 정보 일괄 가져오기 (하루 한 번이면 충분)
        self.stdout.write("Fetching latest fundamental data for all KRX stocks (Marcap, PER, PBR, etc.)...")
        df_all_fundamentals = pd.DataFrame()
        try:
            # KRX-DESC는 더 많은 펀더멘털 정보를 포함할 수 있음 (EPS, BPS 등)
            df_all_fundamentals = fdr.StockListing('KRX-DESC') 
            if 'Code' in df_all_fundamentals.columns: 
                df_all_fundamentals.rename(columns={'Code': 'Symbol'}, inplace=True)
            
            if 'Symbol' in df_all_fundamentals.columns:
                df_all_fundamentals.set_index('Symbol', inplace=True)
                # PER, PBR 등의 컬럼이 object 타입일 경우 float으로 변환 시도
                for col in ['PER', 'PBR', 'EPS', 'BPS', 'DPS', 'ROE']: # 필요한 펀더멘털 컬럼들
                    if col in df_all_fundamentals.columns:
                        df_all_fundamentals[col] = pd.to_numeric(df_all_fundamentals[col], errors='coerce')
                self.stdout.write(self.style.SUCCESS(f"Fetched and processed fundamental data for {len(df_all_fundamentals)} stocks."))
            else:
                self.stderr.write(self.style.ERROR("Could not find 'Symbol' column in KRX-DESC fundamental data. Fundamentals might be incomplete."))
                df_all_fundamentals = pd.DataFrame() 
        except Exception as e_fund:
            self.stderr.write(self.style.ERROR(f"Error fetching KRX fundamental data: {e_fund}"))
            traceback.print_exc()
            df_all_fundamentals = pd.DataFrame() 

        # 3. 개별 종목 데이터 처리 (가격, 투자자, 펀더멘털)
        markets_to_list = ['KOSPI', 'KOSDAQ'] # 처리할 시장 목록
        for market_name in markets_to_list:
            self.stdout.write(f"\nProcessing {market_name} stocks up to {target_date_for_data}...")
            try:
                stocks_df_listing = fdr.StockListing(market_name)
                if 'Code' in stocks_df_listing.columns: 
                    stocks_df_listing.rename(columns={'Code': 'Symbol'}, inplace=True)
                
                if 'Symbol' not in stocks_df_listing.columns or 'Name' not in stocks_df_listing.columns:
                    self.stderr.write(self.style.ERROR(f"Stock listing for {market_name} missing 'Symbol' or 'Name'. Skipping market."))
                    continue
                
                stocks_df_listing = stocks_df_listing[stocks_df_listing['Symbol'].notna() & stocks_df_listing['Name'].notna()]
                total_stocks_in_market = len(stocks_df_listing)
                self.stdout.write(f"Found {total_stocks_in_market} stocks in {market_name}. Starting data update/backfill...")

                processed_count, skipped_count, error_count = 0, 0, 0

                for idx, row_listing in stocks_df_listing.iterrows():
                    stock_code = row_listing['Symbol']
                    stock_name = row_listing['Name']
                    
                    if (idx + 1) % 50 == 0: # 로그 출력 빈도 조정
                        self.stdout.write(f"  [{market_name}] Checked {(idx + 1)}/{total_stocks_in_market} stocks...")

                    # DB에 해당 종목의 target_date_for_data가 이미 있는지 확인
                    target_date_exists_in_db = StockPrice.objects.filter(stock_code=stock_code, date=target_date_for_data).exists()
                    
                    # DB에 저장된 해당 종목의 총 레코드 수 (히스토리 충분 여부 판단용)
                    num_records_in_db = StockPrice.objects.filter(stock_code=stock_code).count()

                    fetch_start_date_for_fdr = None
                    
                    if not target_date_exists_in_db: # 목표일 데이터가 없으면 무조건 가져옴
                        if num_records_in_db < self.MIN_RECORDS_FOR_SUFFICIENT_HISTORY:
                            # 목표일 데이터도 없고, 히스토리도 부족하면 많이 가져옴
                            fetch_start_date_for_fdr = self.get_previous_trading_day(target_date_for_data, self.DAYS_TO_FETCH_FOR_BACKFILL)
                            self.stdout.write(f"    [BACKFILL] {stock_name}({stock_code}): Fetching from {fetch_start_date_for_fdr} to {target_date_for_data} (DB records: {num_records_in_db})")
                        else:
                            # 목표일 데이터는 없지만, 히스토리는 충분하면 목표일 근처만 가져옴 (예: 최근 5일치)
                            fetch_start_date_for_fdr = self.get_previous_trading_day(target_date_for_data, 5) 
                            self.stdout.write(f"    [UPDATE_RECENT] {stock_name}({stock_code}): Fetching from {fetch_start_date_for_fdr} to {target_date_for_data} (DB records: {num_records_in_db})")
                    elif num_records_in_db < self.MIN_RECORDS_FOR_SUFFICIENT_HISTORY : # 목표일 데이터는 있지만, 히스토리가 부족한 경우 (이 경우는 거의 없어야 함)
                        fetch_start_date_for_fdr = self.get_previous_trading_day(target_date_for_data, self.DAYS_TO_FETCH_FOR_BACKFILL)
                        self.stdout.write(f"    [FILL_HISTORY] {stock_name}({stock_code}): Target date exists, but history insufficient. Fetching from {fetch_start_date_for_fdr} to {target_date_for_data} (DB records: {num_records_in_db})")
                    else: # 목표일 데이터도 있고, 히스토리도 충분하면 스킵
                        self.stdout.write(f"    [SKIP] {stock_name}({stock_code}) is up-to-date with sufficient history ({num_records_in_db} records).")
                        skipped_count +=1
                        continue
                    
                    if fetch_start_date_for_fdr:
                        try:
                            # data_source='naver'는 투자자별 매매동향 포함
                            df_new_data_fdr = fdr.DataReader(stock_code, start=fetch_start_date_for_fdr, end=target_date_for_data, data_source='naver')
                            if df_new_data_fdr.empty:
                                self.stdout.write(self.style.WARNING(f"      No data from FDR for {stock_name} in range {fetch_start_date_for_fdr}-{target_date_for_data}."))
                                error_count +=1
                                continue
                            
                            df_new_data_fdr.index = pd.to_datetime(df_new_data_fdr.index).date
                            df_new_data_fdr.sort_index(inplace=True)

                            saved_count_for_stock_this_fetch = 0
                            for current_iter_date, row_fdr_data in df_new_data_fdr.iterrows():
                                # 이미 DB에 있는 날짜는 update_or_create로 처리하거나, 여기서 건너뛸 수 있음
                                # 여기서는 update_or_create를 사용하므로 중복 저장 시 업데이트됨

                                # 이전 거래일 종가 계산
                                prev_close_val_for_db = None
                                current_df_idx = df_new_data_fdr.index.get_loc(current_iter_date)
                                if current_df_idx > 0:
                                    prev_row_date_in_df = df_new_data_fdr.index[current_df_idx - 1]
                                    prev_close_val_for_db = df_new_data_fdr.loc[prev_row_date_in_df, 'Close']
                                else: 
                                    prev_trading_day_of_current_iter = self.get_previous_trading_day(current_iter_date, 1)
                                    prev_day_record_in_db = StockPrice.objects.filter(stock_code=stock_code, date=prev_trading_day_of_current_iter).first()
                                    if prev_day_record_in_db:
                                        prev_close_val_for_db = prev_day_record_in_db.close_price
                                
                                # 펀더멘털 정보 (스크립트 실행 시점의 최신 값 사용)
                                # PER, PBR 등 NaN 값은 0.0으로 처리
                                fund_defaults = {'market_cap': None, 'per': 0.0, 'pbr': 0.0, 'eps': None, 'bps': None, 'dps': None, 'roe': None}
                                if not df_all_fundamentals.empty and stock_code in df_all_fundamentals.index:
                                    fund_data_row_series = df_all_fundamentals.loc[stock_code]
                                    fund_defaults['market_cap'] = fund_data_row_series.get('Marcap') if pd.notna(fund_data_row_series.get('Marcap')) else None
                                    fund_defaults['per'] = fund_data_row_series.get('PER') if pd.notna(fund_data_row_series.get('PER')) else 0.0
                                    fund_defaults['pbr'] = fund_data_row_series.get('PBR') if pd.notna(fund_data_row_series.get('PBR')) else 0.0
                                    # 새로운 펀더멘털 컬럼 추가 시 여기에 반영
                                    fund_defaults['eps'] = fund_data_row_series.get('EPS') if pd.notna(fund_data_row_series.get('EPS')) else None
                                    fund_defaults['bps'] = fund_data_row_series.get('BPS') if pd.notna(fund_data_row_series.get('BPS')) else None
                                    fund_defaults['dps'] = fund_data_row_series.get('DPS') if pd.notna(fund_data_row_series.get('DPS')) else None
                                    fund_defaults['roe'] = fund_data_row_series.get('ROE') if pd.notna(fund_data_row_series.get('ROE')) else None

                                defaults_for_db = {
                                    'stock_name': stock_name, 
                                    'market_name': market_name,
                                    'open_price': row_fdr_data.get('Open'), 
                                    'high_price': row_fdr_data.get('High'),
                                    'low_price': row_fdr_data.get('Low'), 
                                    'close_price': row_fdr_data['Close'],
                                    'previous_day_close_price': prev_close_val_for_db, 
                                    'volume': row_fdr_data.get('Volume'),
                                    'trade_value': row_fdr_data.get('거래대금'), # FDR '거래대금' 컬럼 사용
                                    'indi_volume': row_fdr_data.get('개인') if pd.notna(row_fdr_data.get('개인')) else None,
                                    'foreign_volume': row_fdr_data.get('외국인') if pd.notna(row_fdr_data.get('외국인')) else None,
                                    'organ_volume': row_fdr_data.get('기관') if pd.notna(row_fdr_data.get('기관')) else None,
                                    **fund_defaults # 펀더멘털 정보 병합
                                }
                                # change_value, change_percent는 StockPrice 모델의 save() 메소드에서 자동 계산

                                obj, created = StockPrice.objects.update_or_create(
                                    stock_code=stock_code, 
                                    date=current_iter_date,
                                    defaults=defaults_for_db
                                )
                                if created: saved_count_for_stock_this_fetch += 1
                            
                            if saved_count_for_stock_this_fetch > 0:
                                self.stdout.write(self.style.SUCCESS(f"      Saved/Updated {saved_count_for_stock_this_fetch} daily records for {stock_name}."))
                                processed_count +=1
                            elif not target_date_exists_in_db : 
                                self.stdout.write(self.style.WARNING(f"      Fetched data for {stock_name} but no new records were saved (target date {target_date_for_data} still missing)."))
                                error_count +=1
                        except Exception as e_stock_fetch:
                            self.stderr.write(self.style.ERROR(f"    Error processing/fetching data for {stock_name} ({stock_code}): {e_stock_fetch}"))
                            traceback.print_exc()
                            error_count +=1
                        time.sleep(0.15) # FDR 호출 간격 (조금 늘림)
                
                self.stdout.write(self.style.SUCCESS(f"Finished processing {market_name}. Total: {total_stocks_in_market}, Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}"))
            except Exception as e_market_listing:
                self.stderr.write(self.style.ERROR(f"Error fetching stock list for {market_name}: {e_market_listing}"))
                traceback.print_exc()
        
        self.stdout.write(self.style.SUCCESS(f'\nDaily DB data update/backfill process finished at {timezone.now()}.'))
