import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta, date as date_type
import traceback
import holidays
from django.core.cache import cache
import os
import glob
from django.conf import settings
import logging # 로깅 모듈 추가

# 로거 설정 (필요에 따라 Django 로깅 설정과 통합 가능)
logger = logging.getLogger(__name__)
# 기본 로깅 레벨 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
# 실제 운영 환경에서는 INFO 또는 WARNING으로 설정하는 것이 일반적입니다.
# Django 프로젝트의 settings.py에서 LOGGING 설정을 통해 관리하는 것이 더 좋습니다.
if not logger.handlers: # 핸들러가 이미 설정되어 있는지 확인
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # 기본 로깅 레벨

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
    logger.info("[predict_info/utils.py] pandas_ta 라이브러리가 성공적으로 로드되었습니다.")
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.critical("[predict_info/utils.py] 'pandas_ta' 라이브러리를 찾을 수 없습니다. 기술적 지표 계산이 불가능합니다. 서버 환경에 'pip install pandas-ta'로 설치해주세요.")

CSV_BASE_DIR = "data_csv_for_models"
CSV_FILENAME_SUFFIX = "_TA_extended.csv"
CSV_FILENAME_DATE_FORMAT = "%Y%m%d"

KOSPI_TRAINED_FEATURES_FALLBACK = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'ATR_14', 'BBL_20_2.0',
    'BBM_20_2.0', 'BBU_20_2.0', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9',
    'MACDs_12_26_9', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14',
    'DMP_14', 'DMN_14', 'KOSPI_Close', 'KOSPI_Change', 'USD_KRW_Close',
    'USD_KRW_Change', 'Indi', 'Foreign', 'Organ', 'MarketCap', 'PBR', 'PER',
    'MarketCap_is_nan', 'PBR_is_nan', 'PER_is_nan', 'PER_is_zero'
]


def get_kr_holidays(years):
    """지정된 연도들에 대한 한국 공휴일 목록을 반환합니다."""
    kr_holidays_list = []
    if not isinstance(years, list):
        years = [years]
    for year_val in years:
        try:
            kr_holidays_list.extend(holidays.KR(years=year_val).keys())
        except KeyError:
            logger.warning(f"[utils.get_kr_holidays] No holiday data found for year {year_val}. Skipping.")
    return [pd.to_datetime(d).date() for d in kr_holidays_list]


def get_market_macro_data(start_date_str, end_date_str, market_name='KOSPI', other_market_name_for_index=None):
    """지정된 기간 동안의 주요 시장 지수 및 환율 데이터를 가져옵니다."""
    df_market_macro = pd.DataFrame()
    main_market_symbol, other_market_symbol = None, None
    primary_market_context = market_name.upper() # 일관성을 위해 대문자로 처리

    if primary_market_context == 'KOSPI':
        main_market_symbol = 'KS11'
        # other_market_name_for_index가 명시적으로 KOSDAQ이거나, 명시되지 않았을 때 기본으로 KOSDAQ 설정
        if (other_market_name_for_index and other_market_name_for_index.upper() == 'KOSDAQ') or not other_market_name_for_index:
            other_market_symbol = 'KQ11'
            if not other_market_name_for_index: other_market_name_for_index = 'KOSDAQ'
    elif primary_market_context == 'KOSDAQ':
        main_market_symbol = 'KQ11'
        # other_market_name_for_index가 명시적으로 KOSPI거나, 명시되지 않았을 때 기본으로 KOSPI 설정
        if (other_market_name_for_index and other_market_name_for_index.upper() == 'KOSPI') or not other_market_name_for_index:
            other_market_symbol = 'KS11'
            if not other_market_name_for_index: other_market_name_for_index = 'KOSPI'
    else:
        logger.warning(f"[utils.get_market_macro_data] Unexpected market_name '{market_name}'. Defaulting to KOSPI context.")
        main_market_symbol, primary_market_context = 'KS11', 'KOSPI'
        other_market_symbol, other_market_name_for_index = 'KQ11', 'KOSDAQ'
        
    try:
        if main_market_symbol:
            df_main = fdr.DataReader(main_market_symbol, start_date_str, end_date_str)
            if not df_main.empty:
                df_main['Change'] = df_main['Close'].pct_change() if 'Change' not in df_main.columns and 'Close' in df_main.columns else df_main.get('Change')
                df_market_macro = df_main[['Close', 'Change']].copy().rename(columns={'Close': f"{primary_market_context}_Close", 'Change': f"{primary_market_context}_Change"})

        if other_market_symbol and other_market_name_for_index:
            df_other = fdr.DataReader(other_market_symbol, start_date_str, end_date_str)
            if not df_other.empty:
                df_other['Change'] = df_other['Close'].pct_change() if 'Change' not in df_other.columns and 'Close' in df_other.columns else df_other.get('Change')
                df_other_processed = df_other[['Close', 'Change']].copy().rename(columns={'Close': f"{other_market_name_for_index.upper()}_Close", 'Change': f"{other_market_name_for_index.upper()}_Change"})
                df_market_macro = df_market_macro.join(df_other_processed, how='outer') if not df_market_macro.empty else df_other_processed
        
        df_usd = fdr.DataReader('USD/KRW', start_date_str, end_date_str)
        if not df_usd.empty:
            df_usd['Change'] = df_usd['Close'].pct_change() if 'Change' not in df_usd.columns and 'Close' in df_usd.columns else df_usd.get('Change')
            cols_usd = [col for col in ['Close', 'Change'] if col in df_usd.columns]
            if cols_usd:
                df_usd_p = df_usd[cols_usd].copy().rename(columns={c: f"USD_KRW_{c}" for c in cols_usd})
                df_market_macro = df_market_macro.join(df_usd_p, how='outer') if not df_market_macro.empty else df_usd_p
        
        if not df_market_macro.empty:
            if isinstance(df_market_macro.index, pd.DatetimeIndex):
                df_market_macro.index = df_market_macro.index.date
            else:
                df_market_macro.index = pd.to_datetime(df_market_macro.index, errors='coerce').date
            df_market_macro = df_market_macro.ffill().bfill()
        return df_market_macro

    except Exception as e:
        logger.error(f"[utils.get_market_macro_data] Error fetching market/macro data: {e}\n{traceback.format_exc()}")
        expected_cols = []
        if primary_market_context: expected_cols.extend([f"{primary_market_context}_Close", f"{primary_market_context}_Change"])
        if other_market_name_for_index: expected_cols.extend([f"{other_market_name_for_index.upper()}_Close", f"{other_market_name_for_index.upper()}_Change"])
        expected_cols.extend(['USD_KRW_Close', 'USD_KRW_Change'])
        return pd.DataFrame(columns=expected_cols)


def calculate_technical_indicators(df_ohlcv, pandas_ta_available=PANDAS_TA_AVAILABLE):
    """OHLCV DataFrame에 기술적 지표를 계산하여 추가합니다."""
    if not pandas_ta_available:
        logger.warning("[calculate_technical_indicators] pandas_ta is not available. Skipping TA calculation.")
        return df_ohlcv.copy()

    df_ta_input = df_ohlcv.copy()
    std_col_names = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
    df_ta_input.rename(columns=std_col_names, inplace=True, errors='ignore')

    required_cols_lower = ['open', 'high', 'low', 'close']
    if not all(col in df_ta_input.columns for col in required_cols_lower):
        missing_cols = [col for col in required_cols_lower if col not in df_ta_input.columns]
        logger.error(f"[calculate_technical_indicators] Required OHLC columns (lower case: {missing_cols}) are missing. Cannot calculate TA.")
        return df_ohlcv.copy()

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df_ta_input.columns:
            df_ta_input[col] = pd.to_numeric(df_ta_input[col], errors='coerce')
    
    if 'volume' in df_ta_input.columns and df_ta_input['volume'].notna().any():
        df_ta_input['volume'] = df_ta_input['volume'].astype('float64')
    else:
        df_ta_input['volume'] = pd.Series(np.nan, index=df_ta_input.index, dtype='float64')

    if 'close' not in df_ta_input or df_ta_input['close'].isnull().all():
        logger.warning("[calculate_technical_indicators] All 'close' prices are NaN or column missing. TA calculations will result in NaNs.")
        return df_ohlcv.copy()

    df_with_ta_features = df_ohlcv.copy()

    try:
        df_with_ta_features['ATR_14'] = ta.atr(df_ta_input['high'], df_ta_input['low'], df_ta_input['close'], length=14)
        bbands = ta.bbands(df_ta_input['close'], length=20, std=2.0)
        if bbands is not None and not bbands.empty:
            for col_name in ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0']:
                df_with_ta_features[col_name] = bbands.get(col_name)
        for length in [14, 6, 28]:
            df_with_ta_features[f'RSI_{length}'] = ta.rsi(df_ta_input['close'], length=length)
        macd = ta.macd(df_ta_input['close'], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            for col_name in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
                df_with_ta_features[col_name] = macd.get(col_name)
        stoch_slow = ta.stoch(df_ta_input['high'], df_ta_input['low'], df_ta_input['close'], k=14, d=3, smooth_k=3)
        if stoch_slow is not None and not stoch_slow.empty:
            df_with_ta_features['STOCHk_14_3_3'] = stoch_slow.get('STOCHk_14_3_3')
            df_with_ta_features['STOCHd_14_3_3'] = stoch_slow.get('STOCHd_14_3_3')
        stoch_fast = ta.stoch(df_ta_input['high'], df_ta_input['low'], df_ta_input['close'], k=14, d=3, smooth_k=1)
        if stoch_fast is not None and not stoch_fast.empty:
            df_with_ta_features['STOCHk_fast_14_3_1'] = stoch_fast.get(stoch_fast.columns[0]) 
            df_with_ta_features['STOCHd_fast_14_3_1'] = stoch_fast.get(stoch_fast.columns[1]) 

        if df_ta_input['volume'].notna().any():
            df_with_ta_features['OBV'] = ta.obv(df_ta_input['close'], df_ta_input['volume'])
        else:
            df_with_ta_features['OBV'] = np.nan
        adx_df = ta.adx(df_ta_input['high'], df_ta_input['low'], df_ta_input['close'], length=14)
        if adx_df is not None and not adx_df.empty:
            for col_name in ['ADX_14', 'DMP_14', 'DMN_14']:
                df_with_ta_features[col_name] = adx_df.get(col_name)
        for length in [5, 10, 20, 60, 120]:
            df_with_ta_features[f'MA_{length}'] = ta.sma(df_ta_input['close'], length=length)
            df_with_ta_features[f'EMA_{length}'] = ta.ema(df_ta_input['close'], length=length)
        df_with_ta_features['CCI_14_0.015'] = ta.cci(df_ta_input['high'], df_ta_input['low'], df_ta_input['close'], length=14, c=0.015)
        if df_ta_input['volume'].notna().any():
            mfi_series = ta.mfi(df_ta_input['high'], df_ta_input['low'], df_ta_input['close'], df_ta_input['volume'].astype('float64'), length=14)
            df_with_ta_features['MFI_14'] = mfi_series.astype('float64')
        else:
            df_with_ta_features['MFI_14'] = np.nan
        df_with_ta_features['WILLR_14'] = ta.willr(df_ta_input['high'], df_ta_input['low'], df_ta_input['close'], length=14)
        df_with_ta_features['MOM_10'] = ta.mom(df_ta_input['close'], length=10)
        df_with_ta_features['ROC_10'] = ta.roc(df_ta_input['close'], length=10)
        trix_df = ta.trix(df_ta_input['close'], length=14, signal=9)
        if trix_df is not None and not trix_df.empty:
            df_with_ta_features['TRIX_14_9'] = trix_df.iloc[:, 0]

        if hasattr(ta, 'vr') and df_ta_input['volume'].notna().any():
            vr_series = ta.vr(df_ta_input['close'], df_ta_input['volume'].astype('float64'), length=20)
            df_with_ta_features['VR_20'] = vr_series.astype('float64')
        else: df_with_ta_features['VR_20'] = np.nan
        if hasattr(ta, 'psl'):
            df_with_ta_features['PSL_12'] = ta.psl(df_ta_input['close'], length=12)
        else: df_with_ta_features['PSL_12'] = np.nan
        
        df_with_ta_features['Log_Close'] = np.log1p(df_ta_input['close'].clip(lower=0)).astype('float64')
        if df_ta_input['volume'].notna().any():
            log_volume_series = np.log1p(df_ta_input['volume'].fillna(0).clip(lower=0))
            df_with_ta_features['Log_Volume'] = log_volume_series.astype('float64')
        else:
            df_with_ta_features['Log_Volume'] = np.nan
        
        df_with_ta_features.replace([np.inf, -np.inf], np.nan, inplace=True)

    except Exception as e_ta:
        logger.error(f"[calculate_technical_indicators] Error during TA calculation: {e_ta}\n{traceback.format_exc()}")
    return df_with_ta_features


def calculate_all_features(stock_df_ohlcv, market_macro_data_df, investor_df, fundamental_df, pandas_ta_available=PANDAS_TA_AVAILABLE):
    """모든 피처를 계산하고 병합합니다."""
    if stock_df_ohlcv is None or stock_df_ohlcv.empty:
        logger.warning("[calculate_all_features] stock_df_ohlcv is empty. Cannot calculate features.")
        return pd.DataFrame()

    def ensure_date_index(df_to_check, df_name="DataFrame"):
        if df_to_check is not None and not df_to_check.empty:
            is_already_date_objects = isinstance(df_to_check.index, pd.DatetimeIndex) and \
                                      all(isinstance(i, date_type) for i in df_to_check.index if pd.notna(i))
            
            if not is_already_date_objects:
                try:
                    current_index_type = type(df_to_check.index)
                    if isinstance(df_to_check.index, pd.DatetimeIndex):
                        df_to_check.index = df_to_check.index.date
                    elif pd.api.types.is_object_dtype(df_to_check.index) or \
                         pd.api.types.is_string_dtype(df_to_check.index) or \
                         not all(isinstance(i, date_type) for i in df_to_check.index if pd.notna(i)):
                        df_to_check.index = pd.to_datetime(df_to_check.index, errors='coerce').date
                    elif not isinstance(df_to_check.index, pd.DatetimeIndex) and all(isinstance(i, date_type) for i in df_to_check.index if pd.notna(i)):
                         df_to_check.index = pd.DatetimeIndex(df_to_check.index).date
                except Exception as e_idx_convert:
                    logger.warning(f"[ensure_date_index] Failed to convert index of {df_name} to date objects: {e_idx_convert}. Original index type: {current_index_type}, Index head: {df_to_check.index[:2]}")
        return df_to_check

    stock_df_ohlcv = ensure_date_index(stock_df_ohlcv, "stock_df_ohlcv")
    market_macro_data_df = ensure_date_index(market_macro_data_df, "market_macro_data_df")
    investor_df = ensure_date_index(investor_df, "investor_df")
    fundamental_df = ensure_date_index(fundamental_df, "fundamental_df")

    df_with_ta = calculate_technical_indicators(stock_df_ohlcv, pandas_ta_available)
    final_df = df_with_ta.copy()

    for df_to_join, suffix, cols_to_ensure_if_empty in [
        (market_macro_data_df, '_macro_duplicate', ['KOSPI_Close', 'KOSPI_Change', 'KOSDAQ_Close', 'KOSDAQ_Change', 'USD_KRW_Close', 'USD_KRW_Change']),
        (investor_df, '_investor_duplicate', ['Indi', 'Foreign', 'Organ']),
        (fundamental_df, '_fundamental_duplicate', ['MarketCap', 'PBR', 'PER', 'EPS', 'BPS', 'DPS', 'ROE'])
    ]:
        if df_to_join is not None and not df_to_join.empty:
            if final_df.index.has_duplicates:
                final_df = final_df[~final_df.index.duplicated(keep='first')]
            if df_to_join.index.has_duplicates:
                df_to_join_unique = df_to_join[~df_to_join.index.duplicated(keep='first')]
                final_df = final_df.join(df_to_join_unique, how='left', rsuffix=suffix)
            else:
                final_df = final_df.join(df_to_join, how='left', rsuffix=suffix)
        else:
            for col_name in cols_to_ensure_if_empty:
                if col_name not in final_df.columns:
                    final_df[col_name] = np.nan
            
    cols_to_drop = [col for col in final_df.columns if '_duplicate' in col]
    final_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            
    if final_df.index.has_duplicates:
        final_df = final_df[~final_df.index.duplicated(keep='first')]

    for col in final_df.columns:
        if final_df[col].dtype == 'object':
            try:
                final_df[col] = final_df[col].apply(lambda x: np.nan if x is None else x)
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
            except Exception as e_final_conv:
                logger.warning(f"Warning: Final conversion of column {col} to numeric failed. Error: {e_final_conv}. Column head: {final_df[col].head()}")
    return final_df


def add_fundamental_indicator_features(df):
    """펀더멘털 기반 파생 플래그 피처를 추가합니다."""
    df_processed = df.copy()
    
    if 'PER' in df_processed.columns:
        per_numeric = pd.to_numeric(df_processed['PER'], errors='coerce')
        df_processed['PER_is_nan'] = per_numeric.isna().astype(int)
        df_processed['PER_is_zero'] = (per_numeric == 0).astype(int)
    else: 
        df_processed['PER_is_nan'] = 1
        df_processed['PER_is_zero'] = 0

    if 'PBR' in df_processed.columns:
        pbr_numeric = pd.to_numeric(df_processed['PBR'], errors='coerce')
        df_processed['PBR_is_nan'] = pbr_numeric.isna().astype(int)
    else: 
        df_processed['PBR_is_nan'] = 1
        
    if 'MarketCap' in df_processed.columns:
        mc_numeric = pd.to_numeric(df_processed['MarketCap'], errors='coerce')
        df_processed['MarketCap_is_nan'] = mc_numeric.isna().astype(int)
    else: 
        df_processed['MarketCap_is_nan'] = 1
        
    return df_processed


def get_feature_columns_for_model(market_name, model_type='technical'):
    """ 모델 학습에 사용될 피처 컬럼 목록을 반환합니다. """
    market_upper = market_name.upper()
    if market_upper == 'KOSPI' and model_type == 'technical':
        return KOSPI_TRAINED_FEATURES_FALLBACK.copy()

    base_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']
    technical_indicator_features = [
        'MA_5', 'MA_10', 'MA_20', 'MA_60', 'MA_120', 'EMA_5', 'EMA_10', 'EMA_20', 'EMA_60', 'EMA_120',
        'BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
        'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'RSI_6', 'RSI_14', 'RSI_28',
        'STOCHk_14_3_3', 'STOCHd_14_3_3', 'STOCHk_fast_14_3_1', 'STOCHd_fast_14_3_1', 
        'ATR_14', 'ADX_14', 'DMP_14', 'DMN_14', 'CCI_14_0.015', 'MFI_14', 'OBV', 'WILLR_14',
        'MOM_10', 'ROC_10', 'TRIX_14_9', 'VR_20', 'PSL_12', 'Log_Close', 'Log_Volume'
    ]
    
    main_market_context = market_upper
    other_market_context = 'KOSDAQ' if main_market_context == 'KOSPI' else 'KOSPI'

    market_macro_features = [
        f"{main_market_context}_Close", f"{main_market_context}_Change",
        f"{other_market_context}_Close", f"{other_market_context}_Change", 
        'USD_KRW_Close', 'USD_KRW_Change',
    ]

    feature_columns = []
    if model_type == 'technical':
        feature_columns = base_features + technical_indicator_features + market_macro_features
    elif model_type == 'comprehensive':
        investor_features = ['Indi', 'Foreign', 'Organ'] 
        fundamental_value_features = ['MarketCap', 'PBR', 'PER', 'EPS', 'BPS', 'DPS', 'ROE'] 
        fundamental_flag_features = [
            'PER_is_nan', 'PER_is_zero', 
            'PBR_is_nan', 
            'MarketCap_is_nan'
        ]
        feature_columns = (base_features + technical_indicator_features + market_macro_features +
                           investor_features + fundamental_value_features + fundamental_flag_features)
    else:
        logger.warning(f"Unknown model_type '{model_type}' in get_feature_columns_for_model. Defaulting to technical features for {market_name}.")
        feature_columns = base_features + technical_indicator_features + market_macro_features
    
    return list(dict.fromkeys(feature_columns)) # 중복 제거


def get_krx_stock_list(market='KOSPI,KOSDAQ', cache_ttl_seconds=3600):
    """KRX 전체 또는 특정 시장의 종목 목록을 반환합니다. Market 컬럼 값과 비교 시 소문자로 통일하고, 포함 여부로 확인하며, 최종 Market 값을 표준화합니다."""
    market_key_parts = sorted([m.strip().lower() for m in market.split(',') if m.strip()])
    cache_key_market_str = "_".join(market_key_parts) if market_key_parts else "all"
    cache_key = f"krx_stock_list_normalized_{cache_key_market_str}" # 캐시 키 변경 반영
    
    cached_data = cache.get(cache_key)
    if cached_data:
        logger.debug(f"Cache hit for KRX stock list (normalized): {cache_key}")
        return cached_data
    
    logger.debug(f"Cache miss for KRX stock list (normalized): {cache_key}. Fetching from FDR.")
    try:
        krx_df = fdr.StockListing('KRX')
        if krx_df.empty:
            logger.warning("fdr.StockListing('KRX') returned an empty DataFrame.")
            return []
        
        if 'Market' not in krx_df.columns:
            logger.error("[get_krx_stock_list] 'Market' column not found in FDR StockListing.")
            return []
        
        # Market 컬럼을 소문자로 변환하여 'Market_lower' 컬럼 생성 (필터링 및 표준화에 사용)
        krx_df['Market_lower'] = krx_df['Market'].astype(str).str.lower()
            
        markets_to_include_list = [m.strip().lower() for m in market.split(',') if m.strip()]
        
        krx_df_filtered = pd.DataFrame() # 필터링된 결과를 담을 DataFrame 초기화

        if not markets_to_include_list or 'all' in markets_to_include_list:
            krx_df_filtered = krx_df.copy() # 전체 시장 요청 시 원본 복사
        else:
            combined_mask = pd.Series([False] * len(krx_df), index=krx_df.index)
            for m_filter in markets_to_include_list:
                combined_mask = combined_mask | krx_df['Market_lower'].str.contains(m_filter, case=False, na=False)
            krx_df_filtered = krx_df[combined_mask].copy() # 필터링된 결과 복사

            if krx_df_filtered.empty and markets_to_include_list:
                 logger.warning(f"No stocks found for markets (using contains logic): {markets_to_include_list}. Available unique market names (lowercase): {krx_df['Market_lower'].unique()}")
        
        if not krx_df_filtered.empty:
            # --- 시장명 표준화 로직 ---
            STANDARD_KOSDAQ_NAME = "KOSDAQ"
            STANDARD_KOSPI_NAME = "KOSPI"
            # KONEX 등 다른 시장도 필요시 표준화 이름 정의 가능
            # STANDARD_KONEX_NAME = "KONEX"

            # 사용자가 요청한 시장 키워드 확인
            kosdaq_requested = any("kosdaq" in m_keyword for m_keyword in markets_to_include_list)
            kospi_requested = any("kospi" in m_keyword for m_keyword in markets_to_include_list)
            # konex_requested = any("konex" in m_keyword for m_keyword in markets_to_include_list)

            # 'Market' 컬럼 값 표준화 (원본 Market_lower 컬럼 기준으로)
            if kosdaq_requested:
                krx_df_filtered.loc[krx_df_filtered['Market_lower'].str.contains("kosdaq", case=False, na=False), 'Market'] = STANDARD_KOSDAQ_NAME
            
            if kospi_requested:
                krx_df_filtered.loc[krx_df_filtered['Market_lower'].str.contains("kospi", case=False, na=False), 'Market'] = STANDARD_KOSPI_NAME
            
            # if konex_requested: # 코넥스도 표준화가 필요하다면
            #     krx_df_filtered.loc[krx_df_filtered['Market_lower'].str.contains("konex", case=False, na=False), 'Market'] = STANDARD_KONEX_NAME
            # --- 표준화 로직 끝 ---

        # 'Market_lower' 임시 컬럼 삭제
        if 'Market_lower' in krx_df_filtered.columns:
             krx_df_filtered.drop(columns=['Market_lower'], inplace=True)

        # 필수 컬럼 존재 여부 재확인 (Market_lower 삭제 후)
        if not krx_df_filtered.empty and not all(col in krx_df_filtered.columns for col in ['Code', 'Name', 'Market']):
            logger.error(f"[get_krx_stock_list] Filtered StockListing missing required columns ['Code', 'Name', 'Market'] after processing for market: {market}.")
            return []
        
        stock_list = krx_df_filtered[['Code', 'Name', 'Market']].to_dict('records') if not krx_df_filtered.empty else []
        
        cache.set(cache_key, stock_list, cache_ttl_seconds)
        logger.info(f"Fetched and cached {len(stock_list)} stocks for markets: {market_key_parts if market_key_parts else 'all'} (Market names normalized)")
        return stock_list
    except Exception as e:
        logger.error(f"[get_krx_stock_list] Failed to fetch stock list from FDR for market '{market}': {e}\n{traceback.format_exc()}")
        return []

def get_csv_path_for_stock(stock_code, market_name, base_dir=None):
    """특정 종목의 최신 CSV 파일 경로를 찾습니다. market_name은 표준화된 이름(예: KOSDAQ, KOSPI)을 기대합니다."""
    if base_dir is None:
        base_dir = getattr(settings, 'CSV_BASE_DIR_PATH', CSV_BASE_DIR)
    
    # market_name은 이제 "KOSDAQ", "KOSPI" 등 표준화된 대문자 형태를 가정
    market_name_processed = market_name.upper() 
    
    market_path_segment = ""
    if market_name_processed == "KOSPI":
        market_path_segment = "KOSPI_Market_Data_CSV"
    elif market_name_processed == "KOSDAQ": 
        market_path_segment = "KOSDAQ_Market_Data_CSV"
    # elif market_name_processed == "KONEX": # 코넥스 등 다른 시장 추가 시
    #     market_path_segment = "KONEX_Market_Data_CSV"
    else:
        # 표준화되지 않았거나 알 수 없는 market_name 처리
        logger.warning(f"Unsupported or non-standardized market_name '{market_name}' for CSV path. Attempting to use it directly as path segment or a generic segment.")
        # market_path_segment = market_name # 또는 기본 경로
        # 이 경우, 파일명 패턴의 market 부분도 일관성 있게 처리해야 함
        # 안전하게는, 이 함수를 호출할 때 표준화된 market_name을 전달하도록 강제하는 것이 좋음
        # 여기서는 일단 market_name.lower()를 파일명에 사용
        market_path_segment = market_name_processed # 대문자 그대로 사용 또는 특정 기본값
        
    stock_specific_dir = os.path.join(base_dir, market_path_segment)
    
    # 파일명 패턴의 market 부분은 소문자 표준 이름 (예: 'kosdaq')을 사용하도록 통일
    # get_krx_stock_list에서 반환된 표준화된 Market 명(대문자)을 소문자로 변환하여 사용
    filename_market_part = market_name_processed.lower()
    filename_pattern = f"{stock_code}_{filename_market_part}_daily_from_*_to_*{CSV_FILENAME_SUFFIX}"
    search_path = os.path.join(stock_specific_dir, filename_pattern)
    
    logger.debug(f"Searching for CSV files with pattern: {search_path}")
    matching_files = glob.glob(search_path)
    
    if not matching_files:
        logger.warning(f"No CSV files found for stock {stock_code} in market {market_name} with pattern: {search_path}")
        return None

    latest_file = None
    latest_to_date_in_filename = None

    for f_path in matching_files:
        try:
            filename_only = os.path.basename(f_path)
            parts = filename_only.replace(CSV_FILENAME_SUFFIX, "").split('_')
            if len(parts) >= 6 and parts[-3] == "to": 
                date_str_part = parts[-2] 
                current_file_date = datetime.strptime(date_str_part, CSV_FILENAME_DATE_FORMAT).date()
                if latest_to_date_in_filename is None or current_file_date > latest_to_date_in_filename:
                    latest_to_date_in_filename = current_file_date
                    latest_file = f_path
            elif latest_file is None: 
                latest_file = f_path
                logger.debug(f"Filename {filename_only} does not match expected date pattern, but selected as fallback.")
        except (ValueError, IndexError) as e_parse:
            logger.warning(f"Error parsing filename {f_path} for date: {e_parse}. Skipping this file for date comparison.")
            if latest_file is None: latest_file = f_path 
            continue
            
    if latest_file:
        logger.info(f"Latest CSV file found for {stock_code} ({market_name}): {latest_file} (data up to {latest_to_date_in_filename})")
    elif matching_files: 
        latest_file = matching_files[0] 
        logger.warning(f"Could not determine latest file by date for {stock_code} ({market_name}). Returning first match: {latest_file}")
        
    return latest_file


def get_future_trading_dates_list(base_date, num_days, kr_holidays_list=None):
    """주어진 기준일로부터 미래 num_days 만큼의 거래일을 리스트로 반환합니다 (기준일 미포함)."""
    if not isinstance(base_date, (datetime, date_type)):
        try: base_date = pd.to_datetime(base_date).date()
        except Exception: raise ValueError("base_date must be a datetime object or convertible to one.")

    if kr_holidays_list is None:
        current_year = base_date.year
        years_to_fetch_holidays = list(range(current_year, current_year + int(num_days / 200) + 3))
        kr_holidays_list = get_kr_holidays(years_to_fetch_holidays)
    
    future_dates = []
    current_date_iter = base_date
    while len(future_dates) < num_days:
        current_date_iter += timedelta(days=1)
        if current_date_iter.weekday() < 5 and current_date_iter not in kr_holidays_list:
            future_dates.append(current_date_iter)
    return future_dates

def get_past_trading_dates_list(base_date, num_days, kr_holidays_list=None):
    """주어진 기준일로부터 과거 num_days 만큼의 거래일을 리스트로 반환합니다 (기준일 미포함, 날짜 오름차순)."""
    if not isinstance(base_date, (datetime, date_type)):
        try: base_date = pd.to_datetime(base_date).date()
        except Exception: raise ValueError("base_date must be a datetime object or convertible to one.")

    if kr_holidays_list is None:
        years_range = list(range(base_date.year - int(num_days / 200) - 2, base_date.year + 2)) 
        kr_holidays_list = get_kr_holidays(years_range)
    
    past_dates = []
    current_dt_iter = base_date
    while len(past_dates) < num_days:
        current_dt_iter -= timedelta(days=1)
        if current_dt_iter.weekday() < 5 and current_dt_iter not in kr_holidays_list:
            past_dates.append(current_dt_iter)
    return sorted(past_dates)

if __name__ == '__main__':
    logger.setLevel(logging.DEBUG) 
    logger.info("utils.py is being run directly for testing.")
    
    # market 파라미터에 'KOSDAQ' 또는 'kosdaq'을 전달하면 내부적으로 'kosdaq' 키워드로 처리됨
    kosdaq_stocks = get_krx_stock_list(market='KOSDAQ') 
    logger.info(f"Found {len(kosdaq_stocks)} KOSDAQ (and related, normalized) stocks.")
    if kosdaq_stocks:
        logger.info(f"First 5 KOSDAQ (and related, normalized) stocks: {kosdaq_stocks[:5]}")
        
        # 알테오젠 (196170)이 포함되어 있고, Market이 "KOSDAQ"으로 표준화되었는지 확인
        alteogen_stock_info = next((stock for stock in kosdaq_stocks if stock['Code'] == '196170'), None)
        if alteogen_stock_info:
            logger.info(f"알테오젠 (196170) 정보: {alteogen_stock_info}")
            logger.info(f"알테오젠 (196170) Market 표준화 확인: {alteogen_stock_info['Market'] == 'KOSDAQ'}")
        else:
            logger.warning("알테오젠 (196170)이 목록에 없습니다.")

        # 휴젤 (145020) 정보 확인
        hugel_stock_info = next((stock for stock in kosdaq_stocks if stock['Code'] == '145020'), None)
        if hugel_stock_info:
            logger.info(f"휴젤 (145020) 정보: {hugel_stock_info}")
            logger.info(f"휴젤 (145020) Market 표준화 확인: {hugel_stock_info['Market'] == 'KOSDAQ'}")
        else:
            logger.warning("휴젤 (145020)이 목록에 없습니다.")

    # get_csv_path_for_stock 함수 테스트 시, market_name으로 표준화된 "KOSDAQ" 사용
    # 예: csv_path = get_csv_path_for_stock("196170", "KOSDAQ", base_dir="YOUR_CSV_DATA_PATH_HERE")
    # 이 때, CSV 파일명 자체도 "196170_kosdaq_daily_..." 형태로 저장되어 있어야 함.
