# predict_info/utils.py
import pandas as pd
import numpy as np

def calculate_manual_features(df, atr_length=14, bbands_length=20, bbands_std=2, rsi_length=14, macd_fast=12, macd_slow=26, macd_signal=9):
    """
    주어진 DataFrame에 수동 기술적 지표를 계산하여 추가합니다.
    모델 학습 시 사용된 피처 생성 로직과 동일해야 합니다.
    """
    if df.empty:
        return df

    df_out = df.copy()

    # 거래대금 (Value)
    if 'Close' in df_out.columns and 'Volume' in df_out.columns:
        df_out['Value'] = df_out['Close'] * df_out['Volume']
        df_out.loc[df_out['Volume'] == 0, 'Value'] = 0
    else: df_out['Value'] = np.nan

    # 등락률 (Change)
    if 'Close' in df_out.columns:
        df_out['Change'] = df_out['Close'].pct_change() * 100
    else: df_out['Change'] = np.nan

    # ATR
    try:
        if all(col in df_out.columns for col in ['High', 'Low', 'Close']) and len(df_out) >= atr_length:
            high_low = df_out['High'] - df_out['Low']
            high_close = np.abs(df_out['High'] - df_out['Close'].shift())
            low_close = np.abs(df_out['Low'] - df_out['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
            df_out['ATR'] = tr.ewm(alpha=1/atr_length, adjust=False).mean()
        else: df_out['ATR'] = np.nan
    except Exception as e: print(f"ATR 계산 중 오류: {e}"); df_out['ATR'] = np.nan

    # 볼린저 밴드
    try:
        if 'Close' in df_out.columns and len(df_out) >= bbands_length:
            df_out['BB_Mid'] = df_out['Close'].rolling(window=bbands_length).mean()
            std_dev = df_out['Close'].rolling(window=bbands_length).std()
            df_out['BB_Upper'] = df_out['BB_Mid'] + (std_dev * bbands_std)
            df_out['BB_Lower'] = df_out['BB_Mid'] - (std_dev * bbands_std)
        else: df_out['BB_Mid'], df_out['BB_Upper'], df_out['BB_Lower'] = np.nan, np.nan, np.nan
    except Exception as e: print(f"볼린저 밴드 계산 중 오류: {e}"); df_out['BB_Mid'], df_out['BB_Upper'], df_out['BB_Lower'] = np.nan, np.nan, np.nan

    # RSI 계산 (수정된 로직)
    try:
        if 'Close' in df_out.columns and len(df_out) > rsi_length:
            delta = df_out['Close'].diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/rsi_length, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/rsi_length, adjust=False).mean()

            # <<< RSI 계산 오류 수정 >>>
            # loss가 0일 때 gain 값에 따라 RSI 결정 (Series 비교 대신 요소별 비교)
            rs = gain / loss
            # loss가 0이고 gain도 0이면 rs는 0 (RSI 50 아님, 100 / (1+0) -> 100 아님) -> RSI 0이 적절할 수 있음 (상승도 하락도 아님)
            # loss가 0이고 gain > 0 이면 rs는 무한대 (RSI 100)
            rs[loss == 0] = np.where(gain[loss == 0] > 0, np.inf, 0)

            df_out['RSI'] = 100 - (100 / (1 + rs))
            df_out['RSI'] = df_out['RSI'].replace([np.inf, -np.inf], 100) # 무한대는 100으로
            df_out['RSI'] = df_out['RSI'].fillna(50) # 계산 불가능한 초기 NaN은 50으로
            # <<< 수정 끝 >>>
        else:
            df_out['RSI'] = np.nan
    except Exception as e:
        print(f"RSI 계산 중 오류: {e}"); df_out['RSI'] = np.nan

    # MACD
    try:
        if 'Close' in df_out.columns and len(df_out) >= macd_slow:
            ema_fast = df_out['Close'].ewm(span=macd_fast, adjust=False).mean()
            ema_slow = df_out['Close'].ewm(span=macd_slow, adjust=False).mean()
            df_out['MACD'] = ema_fast - ema_slow
            df_out['MACD_Signal'] = df_out['MACD'].ewm(span=macd_signal, adjust=False).mean()
            df_out['MACD_Hist'] = df_out['MACD'] - df_out['MACD_Signal']
        else: df_out['MACD'], df_out['MACD_Signal'], df_out['MACD_Hist'] = np.nan, np.nan, np.nan
    except Exception as e: print(f"MACD 계산 중 오류: {e}"); df_out['MACD'], df_out['MACD_Signal'], df_out['MACD_Hist'] = np.nan, np.nan, np.nan

    return df_out
