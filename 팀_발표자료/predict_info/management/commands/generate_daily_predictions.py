import time
import traceback
from datetime import datetime, timedelta, date as date_type

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db import transaction

from predict_info.models import PredictedStockPrice, StockPrice
from predict_info.utils import (
    PANDAS_TA_AVAILABLE,
    add_fundamental_indicator_features, get_market_macro_data,
    calculate_all_features, get_kr_holidays, get_future_trading_dates_list
)
from predict_info.views import (
    ML_MODELS_DIR, FUTURE_TARGET_DAYS,
    load_model_and_scalers, get_krx_stock_list
)

DAYS_TO_KEEP_PREDICTIONS_DEFAULT = 14

class Command(BaseCommand):
    help = ('Generates daily stock price predictions using pre-trained models, '
            'based on data from StockPrice DB, ensuring correct feature set and preprocessing.')

    def add_arguments(self, parser):
        parser.add_argument('--markets', type=str, default='KOSPI,KOSDAQ', 
                            help="Comma-separated list of markets to process (e.g., KOSPI,KOSDAQ or KOSPI or KOSDAQ). "
                                 "A corresponding model configuration must exist in views.py for each market and model_type combination.")
        parser.add_argument('--stock-codes', type=str, default=None, 
                            help="Comma-separated list of specific stock codes to predict. If provided, --markets is used to filter these specific stocks.")
        parser.add_argument('--model-type-suffix', type=str, default='technical_lstm', 
                            help="Suffix for model type (e.g., 'technical_lstm', 'lstm'). Must form a valid key with market_name in views.py DEFAULT_MODEL_PARAMS.")
        parser.add_argument('--prediction-base-date', type=str, default=None, 
                            help="YYYY-MM-DD. Base date for prediction. Default is latest DB date for each stock.")
        parser.add_argument('--delete-old-days', type=int, default=DAYS_TO_KEEP_PREDICTIONS_DEFAULT, 
                            help="Delete predictions older than this many days. Set to 0 to disable.")

    def handle(self, *args, **options):
        start_time_script = time.time()
        markets_to_process_arg = [m.strip().upper() for m in options['markets'].split(',') if m.strip()]
        specific_stock_codes_str = options['stock_codes']
        model_type_suffix_arg = options['model_type_suffix'] 
        prediction_base_date_str = options['prediction_base_date']
        delete_predictions_older_than_days = options['delete_old_days']

        if not markets_to_process_arg:
            self.stderr.write(self.style.ERROR("No markets specified. Use --markets KOSPI,KOSDAQ or similar."))
            return

        self.stdout.write(self.style.SUCCESS(f"--- Starting Daily Prediction Generation for Markets: {', '.join(markets_to_process_arg)} ---"))
        self.stdout.write(f"Model type suffix to be used: {model_type_suffix_arg}")

        all_stocks_to_attempt = []
        if specific_stock_codes_str:
            codes_to_fetch = [code.strip() for code in specific_stock_codes_str.split(',') if code.strip()]
            full_list_for_specific_check = get_krx_stock_list(market='KOSPI,KOSDAQ')
            for code in codes_to_fetch:
                found_stock_info = next((s for s in full_list_for_specific_check if s['Code'] == code), None)
                if found_stock_info:
                    if found_stock_info['Market'].upper() in markets_to_process_arg:
                        all_stocks_to_attempt.append(found_stock_info)
                    else:
                        self.stdout.write(self.style.WARNING(f"Specified stock {code} (Market: {found_stock_info['Market']}) is not in the target markets (--markets={options['markets']}). Skipping."))
                else:
                    self.stderr.write(self.style.WARNING(f"Specified stock code {code} not found in KRX list. Skipping."))
        else:
            for market_name_iter in markets_to_process_arg:
                self.stdout.write(f"Fetching stock list for market: {market_name_iter}")
                stocks_for_this_market = get_krx_stock_list(market=market_name_iter)
                if not stocks_for_this_market:
                    self.stdout.write(self.style.WARNING(f"No stocks found for market: {market_name_iter} from get_krx_stock_list."))
                all_stocks_to_attempt.extend(stocks_for_this_market)


        if not all_stocks_to_attempt:
            self.stdout.write(self.style.WARNING("No stocks found to predict based on the specified criteria (after filtering by --markets)."))
            return
        
        self.stdout.write(f"Total stocks to attempt processing: {len(all_stocks_to_attempt)}")

        total_processed_ok_count = 0
        total_skipped_count = 0
        total_error_count = 0
        
        current_year = timezone.now().year
        kr_holidays_list_global = get_kr_holidays([current_year, current_year + 1, current_year + 2, current_year +3])

        for stock_idx, stock_info in enumerate(all_stocks_to_attempt):
            stock_code = stock_info['Code']
            stock_name = stock_info['Name']
            actual_stock_market = stock_info['Market'].upper() 
            
            model_market_context_for_load = actual_stock_market 
            
            if model_market_context_for_load not in markets_to_process_arg:
                self.stdout.write(self.style.NOTICE(f"[{stock_idx+1}/{len(all_stocks_to_attempt)}] Skipping {stock_name} ({stock_code}) as its market '{actual_stock_market}' is not in the target processing list: {markets_to_process_arg}."))
                total_skipped_count +=1 
                continue

            self.stdout.write(f"\n[{stock_idx+1}/{len(all_stocks_to_attempt)}] Processing {stock_name} ({stock_code}) - Market: {actual_stock_market} using {model_market_context_for_load} model (Type: {model_type_suffix_arg})...")

            try:
                model, scaler_X, scaler_y, model_info = load_model_and_scalers(model_market_context_for_load, model_type_key=model_type_suffix_arg)
                
                if not model or not scaler_X or not scaler_y or not model_info:
                    self.stderr.write(self.style.ERROR(f"  LOAD FAIL: Could not load model/scalers for market '{model_market_context_for_load}' (type: {model_type_suffix_arg}). Skipping {stock_code}."))
                    total_error_count += 1
                    continue
                
                trained_feature_list = model_info.get('trained_feature_list')
                log_transformed_input_features = model_info.get('log_transformed_input_features', [])
                
                if not trained_feature_list: 
                    self.stderr.write(self.style.ERROR(f"  CONFIG FAIL: 'trained_feature_list' not in model_info for {model_market_context_for_load}_{model_type_suffix_arg}. Skipping {stock_code}."))
                    total_error_count += 1
                    continue

                time_steps = model_info['time_steps']
                model_was_log_trained_target = model_info['model_was_log_trained']
                market_name_for_feature_calc = model_info.get('market_name_for_features', model_market_context_for_load)

                db_query_end_date = None
                if prediction_base_date_str:
                    try:
                        db_query_end_date = datetime.strptime(prediction_base_date_str, '%Y-%m-%d').date()
                    except ValueError:
                        self.stderr.write(self.style.ERROR(f"  DATE FAIL: Invalid prediction_base_date format: {prediction_base_date_str}. Skipping {stock_code}."))
                        total_skipped_count += 1
                        continue
                else:
                    latest_stock_price_obj = StockPrice.objects.filter(stock_code=stock_code).order_by('-date').first()
                    if not latest_stock_price_obj:
                        self.stdout.write(self.style.WARNING(f"  DB SKIP: No data in DB for {stock_code}. Skipping."))
                        total_skipped_count += 1
                        continue
                    db_query_end_date = latest_stock_price_obj.date
                
                min_ta_window = 120 
                min_records_needed_for_sequence_and_ta = min_ta_window + time_steps
                
                calendar_day_fetch_multiplier = 1.8 
                fetch_buffer_days = 60 
                required_history_calendar_days = int(min_records_needed_for_sequence_and_ta * calendar_day_fetch_multiplier) + fetch_buffer_days
                
                db_query_start_date = db_query_end_date - timedelta(days=required_history_calendar_days)

                stock_price_qs = StockPrice.objects.filter(
                    stock_code=stock_code,
                    date__gte=db_query_start_date,
                    date__lte=db_query_end_date
                ).order_by('date')

                num_records_fetched = stock_price_qs.count()
                if num_records_fetched < min_records_needed_for_sequence_and_ta:
                    self.stdout.write(self.style.WARNING(f"  DATA SKIP: Insufficient historical data for {stock_code}. "
                                                        f"Fetched {num_records_fetched} (range: {db_query_start_date} to {db_query_end_date}), "
                                                        f"Need {min_records_needed_for_sequence_and_ta}. Skipping."))
                    total_skipped_count += 1
                    continue
                
                raw_df_from_db = pd.DataFrame(list(stock_price_qs.values()))
                raw_df_from_db['date'] = pd.to_datetime(raw_df_from_db['date'])
                raw_df_from_db.set_index('date', inplace=True)

                base_cols_map_for_calc = {
                    'Open': 'open_price', 'High': 'high_price', 'Low': 'low_price', 'Close': 'close_price', 'Volume': 'volume',
                    'MarketCap': 'market_cap', 'PBR': 'pbr', 'PER': 'per',
                    'Indi': 'indi_volume', 'Foreign': 'foreign_volume', 'Organ': 'organ_volume'
                }
                df_for_feature_calc = pd.DataFrame(index=raw_df_from_db.index)
                for calc_col, db_field_name in base_cols_map_for_calc.items():
                    if db_field_name in raw_df_from_db:
                        df_for_feature_calc[calc_col] = raw_df_from_db[db_field_name]
                    else:
                        if calc_col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                             self.stderr.write(self.style.ERROR(f"  PREP FAIL: DB missing essential base column '{db_field_name}' for '{calc_col}' for {stock_code}. Skipping."))
                             total_error_count += 1; raise ValueError(f"DB missing essential base column {db_field_name}")
                        df_for_feature_calc[calc_col] = np.nan

                if 'Close' in df_for_feature_calc.columns:
                    df_for_feature_calc['Change'] = df_for_feature_calc['Close'].pct_change()
                else: df_for_feature_calc['Change'] = np.nan

                calc_data_start_date_log = df_for_feature_calc.index.min().strftime('%Y-%m-%d')
                calc_data_end_date_log = df_for_feature_calc.index.max().strftime('%Y-%m-%d')
                
                other_market_for_mm_calc = 'KOSDAQ' if market_name_for_feature_calc.upper() == 'KOSPI' else 'KOSPI'
                market_macro_df_calc = get_market_macro_data(
                    calc_data_start_date_log, calc_data_end_date_log, 
                    market_name_for_feature_calc.upper(),
                    other_market_name_for_index=other_market_for_mm_calc
                )

                processed_df_superset = calculate_all_features(
                    stock_df_ohlcv=df_for_feature_calc[['Open', 'High', 'Low', 'Close', 'Volume', 'Change']],
                    market_macro_data_df=market_macro_df_calc,
                    investor_df=df_for_feature_calc[['Indi', 'Foreign', 'Organ']],
                    fundamental_df=df_for_feature_calc[['MarketCap', 'PBR', 'PER']],
                    pandas_ta_available=PANDAS_TA_AVAILABLE
                )
                processed_df_superset = add_fundamental_indicator_features(processed_df_superset) 
                
                final_features_df = pd.DataFrame(index=processed_df_superset.index)
                missing_features_in_superset = []
                for feature_name in trained_feature_list:
                    if feature_name in processed_df_superset:
                        final_features_df[feature_name] = processed_df_superset[feature_name]
                    else:
                        missing_features_in_superset.append(feature_name)
                
                if missing_features_in_superset:
                    self.stderr.write(self.style.ERROR(f"  PREP FAIL: Required features not found after calculation for {stock_code}: {missing_features_in_superset}. Skipping."))
                    total_error_count += 1
                    continue
                
                for col in final_features_df.columns:
                    if final_features_df[col].dtype == 'object':
                        try: final_features_df[col] = pd.to_numeric(final_features_df[col], errors='coerce')
                        except Exception as e_conv:
                            print(f"Warning: Col {col} to numeric failed for {stock_code}. Err: {e_conv}")
                            final_features_df[col] = np.nan 
                    if not pd.api.types.is_float_dtype(final_features_df[col]) and pd.api.types.is_numeric_dtype(final_features_df[col]):
                         if col not in ['MarketCap_is_nan', 'PBR_is_nan', 'PER_is_nan', 'PER_is_zero']:
                            try: final_features_df[col] = final_features_df[col].astype(float)
                            except ValueError: final_features_df[col] = pd.to_numeric(final_features_df[col], errors='coerce')

                if 'Change' in final_features_df.columns:
                    final_features_df['Change'] = pd.to_numeric(final_features_df['Change'], errors='coerce').fillna(0)
                
                final_features_df = final_features_df.ffill().bfill()

                if log_transformed_input_features:
                    for col_to_log in log_transformed_input_features:
                        if col_to_log in final_features_df.columns:
                            numeric_col = pd.to_numeric(final_features_df[col_to_log], errors='coerce')
                            final_features_df[col_to_log] = np.log1p(numeric_col.clip(lower=0)) if not numeric_col.isnull().all() else np.nan
                        else:
                            self.stdout.write(self.style.WARNING(f"   Column '{col_to_log}' for log transform not found for {stock_code}."))
                    final_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    final_features_df = final_features_df.ffill().bfill()

                if final_features_df.isnull().values.any():
                    nan_counts = final_features_df.isnull().sum()
                    nan_cols_with_counts = nan_counts[nan_counts > 0]
                    cols_to_warn = [c for c in nan_cols_with_counts.index if not c.endswith(('_is_nan', '_is_zero'))]
                    if cols_to_warn:
                        self.stdout.write(self.style.WARNING(f"    NaNs WARNING: NaNs present before scaling for {stock_code} in non-indicator columns: {nan_cols_with_counts[cols_to_warn]}"))

                if final_features_df.shape[0] < time_steps:
                    self.stdout.write(self.style.WARNING(f"  DATA SKIP: Final features df too short for {stock_code} (rows: {final_features_df.shape[0]}, need {time_steps}). Skipping."))
                    total_skipped_count += 1
                    continue
                
                if hasattr(scaler_X, 'n_features_in_') and scaler_X.n_features_in_ != final_features_df.shape[1]:
                    msg = (f"SCALER MISMATCH: Scaler X expected {scaler_X.n_features_in_}, got {final_features_df.shape[1]} for {stock_code}. "
                           f"Features: {final_features_df.columns.tolist()}")
                    self.stderr.write(self.style.ERROR(f"  {msg}"))
                    total_error_count += 1
                    continue

                try:
                    # Ensure all data is float before scaling, as scaler expects float
                    final_features_df_float = final_features_df.astype(float)
                    scaled_features = scaler_X.transform(final_features_df_float.values)
                except ValueError as e_scale:
                    if 'Input contains NaN' in str(e_scale):
                        self.stderr.write(self.style.ERROR(f"  SCALER FAIL (NaN): Scaler X failed due to NaNs for {stock_code}. Error: {e_scale}"))
                        self.stdout.write(self.style.WARNING(f"  Attempting fillna(0) as fallback for {stock_code}..."))
                        final_features_df_filled = final_features_df.fillna(0).astype(float) # Ensure float after fillna
                        if final_features_df_filled.isnull().values.any(): # Should not happen
                             self.stderr.write(self.style.ERROR(f"  FILLNA FAIL: NaNs STILL present after fillna(0) for {stock_code}. Skipping."))
                             total_error_count += 1
                             continue
                        scaled_features = scaler_X.transform(final_features_df_filled.values)
                    else: raise e_scale

                last_sequence = scaled_features[-time_steps:]
                last_sequence_reshaped = np.reshape(last_sequence, (1, time_steps, last_sequence.shape[1]))
                predicted_scaled_values = model.predict(last_sequence_reshaped, verbose=0)
                
                try:
                    predicted_actual_values = scaler_y.inverse_transform(predicted_scaled_values)
                except ValueError as ve_scaler_y: 
                    self.stderr.write(self.style.ERROR(f"  SCALER Y FAIL: Scaler Y Error for {stock_code}: {ve_scaler_y}"))
                    if predicted_scaled_values.shape[1] == FUTURE_TARGET_DAYS and hasattr(scaler_y, 'n_features_in_') and scaler_y.n_features_in_ == 1:
                        self.stdout.write(self.style.WARNING(f"    Fallback: scaler_y expects 1 feature. Using first output for all days for {stock_code}."))
                        single_day_pred_scaled = predicted_scaled_values[:, 0].reshape(-1,1)
                        single_day_pred_actual = scaler_y.inverse_transform(single_day_pred_scaled)
                        predicted_actual_values = np.full((1, FUTURE_TARGET_DAYS), single_day_pred_actual[0,0])
                    else:
                        total_error_count += 1; continue

                if model_was_log_trained_target:
                    predicted_actual_values = np.expm1(predicted_actual_values)

                prediction_base_date_for_saving_result = final_features_df.index[-1].date()
                future_dates_for_save = get_future_trading_dates_list(prediction_base_date_for_saving_result, FUTURE_TARGET_DAYS, kr_holidays_list_global)
                last_actual_close_price_for_clip = df_for_feature_calc['Close'].iloc[-1]
                
                predictions_to_save = []
                current_reference_price_for_clipping = last_actual_close_price_for_clip
                for i in range(FUTURE_TARGET_DAYS):
                    predicted_price_val = predicted_actual_values[0, i]
                    
                    # Convert np.nan to None before saving to DB
                    if pd.isna(predicted_price_val): # Checks for np.nan
                        db_price_to_save = None
                    else:
                        # Clip only if not None/NaN
                        price_change_limit_factor = 0.30 
                        upper_limit = current_reference_price_for_clipping * (1 + price_change_limit_factor)
                        lower_limit = current_reference_price_for_clipping * (1 - price_change_limit_factor)
                        clipped_price = np.clip(predicted_price_val, lower_limit, upper_limit)
                        db_price_to_save = round(float(clipped_price), 2)
                    
                    pred_obj = PredictedStockPrice(
                        stock_code=stock_code, stock_name=stock_name, market_name=actual_stock_market,
                        prediction_base_date=prediction_base_date_for_saving_result,
                        predicted_date=future_dates_for_save[i],
                        predicted_price=db_price_to_save, # This can be None
                        analysis_type=f"{model_market_context_for_load.lower()}_{model_type_suffix_arg}",
                    )
                    predictions_to_save.append(pred_obj)
                    
                    # Update clipping reference only if current prediction was valid
                    if db_price_to_save is not None:
                        current_reference_price_for_clipping = db_price_to_save 
                    # else, it remains the last valid price (either actual or previous day's clipped prediction)

                with transaction.atomic():
                    for p_obj_to_save in predictions_to_save:
                        PredictedStockPrice.objects.update_or_create(
                            stock_code=p_obj_to_save.stock_code,
                            prediction_base_date=p_obj_to_save.prediction_base_date,
                            predicted_date=p_obj_to_save.predicted_date,
                            analysis_type=p_obj_to_save.analysis_type,
                            defaults={
                                'stock_name': p_obj_to_save.stock_name,
                                'market_name': p_obj_to_save.market_name,
                                'predicted_price': p_obj_to_save.predicted_price, # Will be NULL if None
                            }
                        )
                self.stdout.write(self.style.SUCCESS(f"  OK: Predicted and saved for {stock_code} based on {prediction_base_date_for_saving_result}"))
                total_processed_ok_count += 1

            except ValueError as ve: 
                 self.stderr.write(self.style.ERROR(f"  PREP FAIL (ValueError): {ve} for {stock_code}. Skipping."))
                 total_error_count +=1
            except Exception as e_pred:
                self.stderr.write(self.style.ERROR(f"  PREDICTION FAIL: Error predicting for {stock_code}: {e_pred}\n{traceback.format_exc()}"))
                total_error_count += 1
            
            time.sleep(0.03)

        self.stdout.write(self.style.SUCCESS(f"\n--- Daily Prediction Generation Summary ---"))
        self.stdout.write(f"Total stocks attempted: {len(all_stocks_to_attempt)}")
        self.stdout.write(f"Markets processed: {', '.join(markets_to_process_arg)}")
        self.stdout.write(f"Model type suffix used: {model_type_suffix_arg}")
        self.stdout.write(f"Successfully predicted/saved: {total_processed_ok_count}")
        self.stdout.write(f"Skipped (DB data issue, insufficient history, etc.): {total_skipped_count}")
        self.stdout.write(f"Errors (model load, data prep, scaler, prediction exception, etc.): {total_error_count}")

        if delete_predictions_older_than_days > 0:
            cutoff_date_for_deletion = timezone.now().date() - timedelta(days=delete_predictions_older_than_days)
            self.stdout.write(f"\nDeleting predictions with prediction_base_date older than {cutoff_date_for_deletion}...")
            try:
                num_deleted, _ = PredictedStockPrice.objects.filter(prediction_base_date__lt=cutoff_date_for_deletion).delete()
                self.stdout.write(self.style.SUCCESS(f"Successfully deleted {num_deleted} old prediction records."))
            except Exception as e_delete_old:
                self.stderr.write(self.style.ERROR(f"Error deleting old predictions: {e_delete_old}"))
        else:
            self.stdout.write(f"Old prediction deletion skipped (delete_old_days set to 0 or less).")

        end_time_script = time.time()
        self.stdout.write(f"Total execution time: {end_time_script - start_time_script:.2f} seconds.")

