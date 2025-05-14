import pandas as pd
import FinanceDataReader as fdr
from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta, datetime, date # 'date'를 명시적으로 임포트
from predict_info.models import MarketIndex, StockPrice # 앱 이름 및 모델 이름 확인
from pandas.tseries.offsets import BDay # Business Day
import holidays # 한국 공휴일 처리
import time

class Command(BaseCommand):
    help = 'Fetches and stores the previous trading day\'s market and stock data from FinanceDataReader.'

    def get_previous_trading_day(self, date_reference, days_offset=1):
        """
        주어진 기준일로부터 이전 N번째 거래일을 찾습니다. (주말 및 한국 공휴일 제외)
        date_reference는 datetime.datetime 또는 datetime.date 객체일 수 있습니다.
        항상 datetime.date 객체를 반환합니다.
        """
        if isinstance(date_reference, datetime):
            current_check_date = date_reference.date()
        elif isinstance(date_reference, date):
            current_check_date = date_reference
        else:
            raise TypeError("date_reference must be a datetime.datetime or datetime.date object.")

        kr_holidays = holidays.KR(years=[current_check_date.year -1, current_check_date.year, current_check_date.year + 1])
        trading_days_found = 0

        while trading_days_found < days_offset:
            current_check_date -= timedelta(days=1)
            if current_check_date.weekday() < 5 and current_check_date not in kr_holidays:
                trading_days_found += 1
        return current_check_date

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting daily market data update...'))

        today_dt = timezone.now()
        target_date = self.get_previous_trading_day(today_dt, 1)
        day_before_target_date = self.get_previous_trading_day(target_date, 1)

        self.stdout.write(f"Target date for data fetching: {target_date}")
        self.stdout.write(f"Day before target date (for change calculation): {day_before_target_date}")

        # 1. 시장 지수 업데이트 (KOSPI, KOSDAQ)
        # 이미 해당 날짜의 데이터가 있으면 건너뛰도록 수정
        indices_to_fetch = {
            'KOSPI': 'KS11',
            'KOSDAQ': 'KQ11'
        }

        for market_readable_name, fdr_ticker in indices_to_fetch.items():
            if MarketIndex.objects.filter(market_name=market_readable_name, date=target_date).exists():
                self.stdout.write(self.style.SUCCESS(f"{market_readable_name} index data for {target_date} already exists. Skipping."))
                continue # 이미 데이터가 있으면 다음 지수로 넘어감

            self.stdout.write(f"Fetching data for {market_readable_name} index ({fdr_ticker})...")
            try:
                df_index = fdr.DataReader(fdr_ticker, start=day_before_target_date, end=target_date)
                
                if df_index.empty:
                    self.stdout.write(self.style.WARNING(f"No data returned for {market_readable_name} for dates {day_before_target_date} to {target_date}."))
                    continue

                df_index.index = pd.to_datetime(df_index.index).date

                data_target_date_rows = df_index[df_index.index == target_date]
                data_day_before_rows = df_index[df_index.index == day_before_target_date]

                if data_target_date_rows.empty:
                    self.stdout.write(self.style.WARNING(f"No data for {market_readable_name} on target date {target_date} in the fetched range."))
                    continue
                
                current_data = data_target_date_rows.iloc[0]
                
                prev_close_val = None
                if not data_day_before_rows.empty:
                    prev_close_val = data_day_before_rows.iloc[0]['Close']
                else:
                    self.stdout.write(self.style.WARNING(f"No data for {market_readable_name} on {day_before_target_date}, previous_day_close_price will be None."))

                # update_or_create 대신 create 사용 (위에서 이미 존재 여부 확인)
                MarketIndex.objects.create(
                    market_name=market_readable_name,
                    date=target_date,
                    close_price=current_data['Close'],
                    previous_day_close_price=prev_close_val,
                    volume=current_data.get('Volume'),
                    trade_value=current_data.get('Amount'),
                )
                self.stdout.write(self.style.SUCCESS(f"Successfully created {market_readable_name} index data for {target_date}."))

            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error fetching {market_readable_name} index: {e}"))
            time.sleep(1) # API 호출 간격

        # 2. 개별 종목 가격 업데이트
        markets_to_list = ['KOSPI', 'KOSDAQ']
        for market in markets_to_list:
            self.stdout.write(f"Fetching stock list for {market}...")
            try:
                stocks_df = fdr.StockListing(market)
                if 'Code' in stocks_df.columns and 'Symbol' not in stocks_df.columns:
                    stocks_df.rename(columns={'Code': 'Symbol'}, inplace=True)
                
                if 'Symbol' not in stocks_df.columns or 'Name' not in stocks_df.columns:
                    self.stderr.write(self.style.ERROR(f"Stock listing for {market} is missing 'Symbol' or 'Name' column."))
                    continue
                
                stocks_df = stocks_df[stocks_df['Symbol'].notna() & stocks_df['Name'].notna()]
                total_stocks = len(stocks_df)
                self.stdout.write(f"Found {total_stocks} stocks in {market}. Processing prices if not exist for {target_date}...")

                processed_count = 0
                skipped_count = 0

                for index, row in stocks_df.iterrows():
                    stock_code = row['Symbol']
                    stock_name = row['Name']
                    
                    # DB에서 해당 종목, 해당 날짜의 데이터가 이미 있는지 확인
                    if StockPrice.objects.filter(stock_code=stock_code, date=target_date).exists():
                        # self.stdout.write(f"Data for {stock_name} ({stock_code}) on {target_date} already exists. Skipping.")
                        skipped_count += 1
                        if (skipped_count + processed_count) % 200 == 0: # 일정 간격으로 스킵/처리 현황 표시
                             self.stdout.write(f"[{market}] Progress: {(skipped_count + processed_count)}/{total_stocks} (Skipped: {skipped_count}, Processed: {processed_count})")
                        continue # 이미 데이터가 있으면 다음 종목으로

                    # 100개 처리마다 로그 대신, 실제 처리된 종목 기준으로 로그 남기도록 변경
                    if processed_count > 0 and processed_count % 50 == 0: # 실제 API 호출한 종목 기준
                         self.stdout.write(f"[{market}] Processed {processed_count} new stocks so far (Current: {stock_name} ({stock_code})). Total checked: {(skipped_count + processed_count)}/{total_stocks}")

                    try:
                        df_stock_price = fdr.DataReader(stock_code, start=day_before_target_date, end=target_date)
                        
                        if df_stock_price.empty:
                            # self.stdout.write(self.style.WARNING(f"No data for {stock_name} ({stock_code}) from FDR."))
                            continue
                        
                        df_stock_price.index = pd.to_datetime(df_stock_price.index).date

                        data_target_date_stock_rows = df_stock_price[df_stock_price.index == target_date]
                        data_day_before_stock_rows = df_stock_price[df_stock_price.index == day_before_target_date]

                        if data_target_date_stock_rows.empty:
                            # self.stdout.write(self.style.WARNING(f"No data for {stock_name} ({stock_code}) on target_date {target_date} in FDR response."))
                            continue
                        
                        current_stock_data = data_target_date_stock_rows.iloc[0]
                        prev_close_val_stock = None
                        if not data_day_before_stock_rows.empty:
                            prev_close_val_stock = data_day_before_stock_rows.iloc[0]['Close']

                        # update_or_create 대신 create 사용 (이미 존재 여부 확인)
                        StockPrice.objects.create(
                            stock_code=stock_code,
                            date=target_date,
                            stock_name=stock_name,
                            market_name=market,
                            open_price=current_stock_data.get('Open'),
                            high_price=current_stock_data.get('High'),
                            low_price=current_stock_data.get('Low'),
                            close_price=current_stock_data['Close'],
                            previous_day_close_price=prev_close_val_stock,
                            volume=current_stock_data.get('Volume'),
                            trade_value=current_stock_data.get('Amount'),
                        )
                        processed_count += 1
                    except Exception as e_stock:
                        self.stderr.write(self.style.ERROR(f"  Error processing new data for {stock_name} ({stock_code}): {e_stock}"))
                    time.sleep(0.1) # API 호출 간격은 유지
                
                self.stdout.write(self.style.SUCCESS(f"Finished processing {market} stocks. Total: {total_stocks}, Processed new: {processed_count}, Skipped existing: {skipped_count}"))

            except Exception as e_market_list:
                self.stderr.write(self.style.ERROR(f"Error fetching stock list for {market}: {e_market_list}"))
        
        self.stdout.write(self.style.SUCCESS('Daily market data update finished.'))
