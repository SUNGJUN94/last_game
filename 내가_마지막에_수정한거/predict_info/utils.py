# predict_info/utils.py
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import traceback # 상세한 오류 로깅을 위해 추가

# pandas-ta 라이브러리 임포트
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
    print("[INFO][predict_info/utils.py] pandas_ta 라이브러리가 성공적으로 로드되었습니다.")
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("[CRITICAL ERROR][predict_info/utils.py] 'pandas_ta' 라이브러리를 찾을 수 없습니다. 기술적 지표 계산이 불가능합니다. 서버 환경에 'pip install pandas-ta'로 설치해주세요.")

def get_market_macro_data(start_date, end_date, market_fdr_code='KS11', currency_pair='USD/KRW'):
    """
    지정된 기간의 시장 지수 및 환율 데이터를 가져옵니다.
    market_fdr_code: FDR에서 사용하는 시장 티커 (예: 'KS11' for KOSPI, 'KQ11' for KOSDAQ)
    반환되는 DataFrame의 시장 지표 컬럼명은 'KOSPI_Close', 'KOSDAQ_Change' 등으로 표준화됩니다.
    """
    df_market_processed = pd.DataFrame()
    df_macro_processed = pd.DataFrame()
    
    actual_market_prefix = "KOSPI" if market_fdr_code == 'KS11' else \
                           "KOSDAQ" if market_fdr_code == 'KQ11' else \
                           market_fdr_code 

    try:
        # 시장 지수 데이터
        df_market_raw = fdr.DataReader(market_fdr_code, start_date, end_date)
        if not df_market_raw.empty:
            df_market_processed[f'{actual_market_prefix}_Close'] = df_market_raw['Close']
            df_market_processed[f'{actual_market_prefix}_Change'] = df_market_raw['Change'] 
            df_market_processed.index = pd.to_datetime(df_market_raw.index)
        else:
            print(f"[WARNING][utils.get_market_macro_data] {actual_market_prefix} 지수 데이터를 찾을 수 없습니다. ({start_date} ~ {end_date})")

        # 환율 데이터 (USD/KRW)
        df_macro_raw = fdr.DataReader(currency_pair, start_date, end_date)
        if not df_macro_raw.empty:
            df_macro_processed['USD_KRW_Close'] = df_macro_raw['Close']
            df_macro_processed['USD_KRW_Change'] = df_macro_raw['Close'].pct_change(fill_method=None)
            df_macro_processed.index = pd.to_datetime(df_macro_raw.index) # 인덱스 통일
        else:
            print(f"[WARNING][utils.get_market_macro_data] 환율 데이터({currency_pair})를 찾을 수 없습니다. ({start_date} ~ {end_date})")

    except Exception as e:
        print(f"[ERROR][utils.get_market_macro_data] 시장/거시 데이터 가져오기 중 오류 발생: {e}")
        # traceback.print_exc() # 디버깅 시 상세 오류 확인
    
    return df_market_processed, df_macro_processed


def calculate_all_features(df_input_with_ohlcv_extras, market_name_upper="KOSPI"):
    """
    주어진 DataFrame (OHLCV + 외부 데이터)에 pandas-ta를 사용하여 모든 기술적 지표를 계산합니다.
    새로운 피처셋에 맞게 기술적 지표 목록을 검토하고, 필요시 추가/수정합니다.
    사용자의 새로운 데이터셋에 포함된 모든 기술적 지표가 여기서 계산되거나,
    또는 이미 계산된 값이 df_input_with_ohlcv_extras에 포함되어 있어야 합니다.
    여기서는 기존과 같이 pandas_ta를 활용하여 주요 지표들을 계산합니다.
    """
    if not PANDAS_TA_AVAILABLE:
        print("[CRITICAL ERROR][utils.calculate_all_features] pandas_ta 라이브러리가 없어 기술적 지표를 계산할 수 없습니다. 피처가 누락됩니다.")
        return df_input_with_ohlcv_extras

    if df_input_with_ohlcv_extras.empty:
        print("[WARNING][utils.calculate_all_features] 입력 DataFrame이 비어있어 TA 계산 불가.")
        return df_input_with_ohlcv_extras

    df = df_input_with_ohlcv_extras.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e_idx:
            print(f"[ERROR][utils.calculate_all_features] DataFrame 인덱스 Datetime 변환 오류: {e_idx}. TA 계산 불가.")
            return df_input_with_ohlcv_extras

    required_ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_ohlcv = [col for col in required_ohlcv if col not in df.columns]
    if missing_ohlcv:
        print(f"[ERROR][utils.calculate_all_features] TA 계산에 필요한 기본 OHLCV 컬럼 누락: {missing_ohlcv}. TA 계산 불가.")
        for col in missing_ohlcv: df[col] = np.nan # 누락 컬럼 NaN으로 추가

    for col in required_ohlcv:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    existing_ohlcv_for_fill = [col for col in required_ohlcv if col in df.columns]
    if existing_ohlcv_for_fill:
        df[existing_ohlcv_for_fill] = df[existing_ohlcv_for_fill].ffill().bfill()

    if 'Close' not in df.columns or df['Close'].isnull().all():
        print(f"[WARNING][utils.calculate_all_features] 'Close' 컬럼이 없거나 ffill/bfill 후에도 전체 NaN입니다. TA 계산을 건너뜁니다.")
        # 모델 입력 형태 유지를 위해 모든 TA 컬럼을 NaN으로 초기화
        # (실제 사용하는 피처 목록에 따라 이 부분은 달라질 수 있음)
        ta_cols_to_add_nan = [ 
            'ATR_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'RSI_14',
            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
            'STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 
            'ADX_14', 'DMP_14', 'DMN_14',
            # 사용자의 새로운 피처셋에 포함된 다른 기술적 지표들도 여기에 추가
        ]
        for ta_col in ta_cols_to_add_nan: df[ta_col] = np.nan
        return df

    # 기술적 지표 계산 최소 데이터 길이 정의
    MIN_LENGTH_DEFAULT = 1 
    MIN_LENGTH_ATR = 15 
    MIN_LENGTH_BBANDS = 20
    MIN_LENGTH_RSI = 15 
    MIN_LENGTH_MACD = 35 
    MIN_LENGTH_STOCH = 18
    MIN_LENGTH_ADX = 27 
    # 새로운 피처에 대한 최소 길이도 필요시 정의

    def safe_ta_call(df_ref, method_name, min_len, ta_output_cols, **kwargs):
        if len(df_ref) >= min_len:
            try:
                getattr(df_ref.ta, method_name)(**kwargs)
            except TypeError as e_type: 
                 if "unsupported operand type(s) for -" in str(e_type) and ("NoneType" in str(e_type) or "float" in str(e_type)):
                    print(f"    [WARNING][utils.safe_ta_call] {method_name.upper()} 계산 중 TypeError (데이터 길이: {len(df_ref)}): {e_type}. 컬럼을 NaN으로 설정.")
                    for col in ta_output_cols: df_ref[col] = np.nan
                 else: 
                    print(f"    [ERROR][utils.safe_ta_call] {method_name.upper()} 계산 중 예상치 못한 TypeError: {e_type}")
                    traceback.print_exc()
                    for col in ta_output_cols: df_ref[col] = np.nan
            except Exception as e_general: 
                print(f"    [ERROR][utils.safe_ta_call] {method_name.upper()} 계산 중 오류: {e_general}")
                traceback.print_exc()
                for col in ta_output_cols: df_ref[col] = np.nan
        else:
            for col in ta_output_cols: df_ref[col] = np.nan

    try:
        # 기존 기술적 지표 계산 (컬럼명은 views.py/generate_daily_predictions.py의 피처 목록과 일치해야 함)
        safe_ta_call(df, 'atr', MIN_LENGTH_ATR, ['ATR_14'], length=14, append=True)
        safe_ta_call(df, 'bbands', MIN_LENGTH_BBANDS, ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0'], length=20, std=2, append=True)
        safe_ta_call(df, 'rsi', MIN_LENGTH_RSI, ['RSI_14'], length=14, append=True)
        safe_ta_call(df, 'macd', MIN_LENGTH_MACD, ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'], fast=12, slow=26, signal=9, append=True)
        
        stoch_cols_expected = ['STOCHk_14_3_3', 'STOCHd_14_3_3']
        if len(df) >= MIN_LENGTH_STOCH:
            stoch_df_result = df.ta.stoch(k=14, d=3, smooth_k=3, append=False) # append=False로 하고 직접 할당
            if stoch_df_result is not None and not stoch_df_result.empty:
                # pandas-ta 버전에 따라 컬럼명이 STOCHk_14_3_3 또는 STOCHk_14_3_3_3_3 등으로 나올 수 있음. 유연하게 대처.
                actual_k_col = next((col for col in stoch_df_result.columns if col.startswith('STOCHk_')), None)
                actual_d_col = next((col for col in stoch_df_result.columns if col.startswith('STOCHd_')), None)
                
                if actual_k_col: df[stoch_cols_expected[0]] = stoch_df_result[actual_k_col]
                else: print(f"    [WARNING] Stochastic K 컬럼({stoch_cols_expected[0]})을 찾을 수 없음. 결과: {stoch_df_result.columns.tolist()}"); df[stoch_cols_expected[0]] = np.nan
                
                if actual_d_col: df[stoch_cols_expected[1]] = stoch_df_result[actual_d_col]
                else: print(f"    [WARNING] Stochastic D 컬럼({stoch_cols_expected[1]})을 찾을 수 없음. 결과: {stoch_df_result.columns.tolist()}"); df[stoch_cols_expected[1]] = np.nan
            else: # stoch_df_result가 None이거나 비어있는 경우
                for col in stoch_cols_expected: df[col] = np.nan
        else: # 데이터 길이 부족
            for col in stoch_cols_expected: df[col] = np.nan

        safe_ta_call(df, 'obv', MIN_LENGTH_DEFAULT, ['OBV'], append=True)
        # OBV 컬럼명 일관성 유지 (가끔 OBVe_ 등으로 생성될 수 있음)
        if 'OBV' not in df.columns and any(col.startswith('OBV') for col in df.columns):
            obv_col_found = [col for col in df.columns if col.startswith('OBV')][0]
            df.rename(columns={obv_col_found: 'OBV'}, inplace=True)
        elif 'OBV' not in df.columns: # OBV 계산 실패 또는 누락 시
            df['OBV'] = np.nan


        adx_cols_expected = ['ADX_14', 'DMP_14', 'DMN_14']
        if len(df) >= MIN_LENGTH_ADX:
            adx_df_result = df.ta.adx(length=14, append=False) # append=False로 하고 직접 할당
            if adx_df_result is not None and not adx_df_result.empty:
                for col_name_expected in adx_cols_expected:
                    if col_name_expected in adx_df_result.columns:
                        df[col_name_expected] = adx_df_result[col_name_expected]
                    else: 
                        alt_col_name = None
                        if col_name_expected == 'DMP_14': alt_col_name = 'PDI_14' # pandas-ta 구버전 호환성
                        elif col_name_expected == 'DMN_14': alt_col_name = 'MDI_14'
                        
                        if alt_col_name and alt_col_name in adx_df_result.columns:
                             df[col_name_expected] = adx_df_result[alt_col_name]
                        else:
                            print(f"    [WARNING] ADX 관련 예상 컬럼 '{col_name_expected}' 누락. 결과: {adx_df_result.columns.tolist()}. NaN으로 설정.")
                            df[col_name_expected] = np.nan
            else: # adx_df_result가 None이거나 비어있는 경우
                for col in adx_cols_expected: df[col] = np.nan
        else: # 데이터 길이 부족
            for col in adx_cols_expected: df[col] = np.nan
        
        # --- 여기에 사용자의 새로운 데이터셋에 포함된 추가 기술적 지표 계산 로직 추가 ---
        # 예시: 만약 'EMA_50' 이라는 피처가 필요하다면
        # safe_ta_call(df, 'ema', 50, ['EMA_50'], length=50, append=True)
        # 사용자의 코랩 코드에서 사용된 모든 피처가 여기서 생성되도록 해야 합니다.
        # 또는, 이미 계산된 피처가 df_input_with_ohlcv_extras에 포함되어 있다면 이 함수에서는 추가 계산이 필요 없을 수 있습니다.

    except Exception as e_outer: 
        print(f"[ERROR][utils.calculate_all_features] pandas_ta 지표 계산 중 외부 루프에서 예외 발생: {e_outer}")
        traceback.print_exc() 
    
    return df
