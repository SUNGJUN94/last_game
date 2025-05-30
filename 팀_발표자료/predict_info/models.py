# predict_info/models.py
from django.db import models
from django.utils import timezone
from django.conf import settings

class MarketIndex(models.Model):
    # ... (기존과 동일)
    market_choices = [
        ('KOSPI', '코스피'),
        ('KOSDAQ', '코스닥'),
    ]
    market_name = models.CharField(max_length=10, choices=market_choices, verbose_name="시장 구분")
    date = models.DateField(verbose_name="날짜")
    close_price = models.FloatField(verbose_name="종가")
    previous_day_close_price = models.FloatField(verbose_name="전일 종가", null=True, blank=True)
    change_value = models.FloatField(verbose_name="대비", null=True, blank=True)
    change_percent = models.FloatField(verbose_name="등락률 (%)", null=True, blank=True)
    volume = models.BigIntegerField(verbose_name="거래량", null=True, blank=True)
    trade_value = models.BigIntegerField(verbose_name="거래대금", null=True, blank=True)

    class Meta:
        verbose_name = "일별 시장 지수"
        verbose_name_plural = "일별 시장 지수 목록"
        unique_together = ('market_name', 'date')
        ordering = ['-date', 'market_name']

    def save(self, *args, **kwargs):
        if self.close_price is not None and self.previous_day_close_price is not None:
            self.change_value = self.close_price - self.previous_day_close_price
            if self.previous_day_close_price != 0:
                self.change_percent = (self.change_value / self.previous_day_close_price) * 100
            else:
                self.change_percent = 0.0 # 0으로 나눌 수 없을 때 등락률 0으로 처리
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.date} {self.get_market_name_display()} 종가: {self.close_price}"

class StockPrice(models.Model):
    """
    일별 개별 종목의 가격 정보 및 다양한 피처를 저장하는 모델 (ipynb 기준 확장)
    """
    stock_code = models.CharField(max_length=20, verbose_name="종목 코드")
    stock_name = models.CharField(max_length=100, verbose_name="종목명")
    market_name = models.CharField(max_length=10, verbose_name="시장 구분", blank=True, null=True) # KOSPI, KOSDAQ
    date = models.DateField(verbose_name="날짜")

    # 기본 OHLCV
    open_price = models.FloatField(verbose_name="시가", null=True, blank=True)
    high_price = models.FloatField(verbose_name="고가", null=True, blank=True)
    low_price = models.FloatField(verbose_name="저가", null=True, blank=True)
    close_price = models.FloatField(verbose_name="종가", null=True, blank=True)
    volume = models.BigIntegerField(verbose_name="거래량", null=True, blank=True)
    
    # 전일 종가 및 변동률 (계산 필드)
    previous_day_close_price = models.FloatField(verbose_name="전일 종가", null=True, blank=True)
    change = models.FloatField(verbose_name="등락률", null=True, blank=True) # 기존 Change 컬럼 (소수점)
    change_value = models.FloatField(verbose_name="대비 가격", null=True, blank=True) # 가격 변동폭

    # 투자자별 거래량 (순매수량 또는 총량, ipynb 기준에 따라 조정)
    # 컬럼명을 좀 더 명확하게 변경 (예: indi_net_buy -> indi_net_purchase_amount)
    # 여기서는 기존 컬럼명 유지
    indi_volume = models.BigIntegerField(verbose_name="개인 순매수(금액 또는 수량)", null=True, blank=True)
    foreign_volume = models.BigIntegerField(verbose_name="외국인 순매수(금액 또는 수량)", null=True, blank=True)
    organ_volume = models.BigIntegerField(verbose_name="기관 순매수(금액 또는 수량)", null=True, blank=True)

    # 펀더멘털 데이터 (ipynb에서 사용 시)
    market_cap = models.BigIntegerField(verbose_name="시가총액", null=True, blank=True)
    per = models.FloatField(verbose_name="PER", null=True, blank=True) 
    pbr = models.FloatField(verbose_name="PBR", null=True, blank=True) 
    eps = models.FloatField(verbose_name="EPS", null=True, blank=True)
    bps = models.FloatField(verbose_name="BPS", null=True, blank=True)
    dps = models.FloatField(verbose_name="DPS", null=True, blank=True) # Dividend Per Share
    roe = models.FloatField(verbose_name="ROE", null=True, blank=True)

    # 기술적 지표 (ipynb에서 사용하는 모든 지표 추가 - 예시)
    # 이동평균
    MA5 = models.FloatField(verbose_name="MA (5일)", null=True, blank=True)
    MA10 = models.FloatField(verbose_name="MA (10일)", null=True, blank=True)
    MA20 = models.FloatField(verbose_name="MA (20일)", null=True, blank=True)
    MA60 = models.FloatField(verbose_name="MA (60일)", null=True, blank=True)
    MA120 = models.FloatField(verbose_name="MA (120일)", null=True, blank=True)
    EMA5 = models.FloatField(verbose_name="EMA (5일)", null=True, blank=True) # 지수이동평균
    EMA10 = models.FloatField(verbose_name="EMA (10일)", null=True, blank=True)
    EMA20 = models.FloatField(verbose_name="EMA (20일)", null=True, blank=True)
    EMA60 = models.FloatField(verbose_name="EMA (60일)", null=True, blank=True)
    EMA120 = models.FloatField(verbose_name="EMA (120일)", null=True, blank=True)

    # 볼린저 밴드
    BB_Upper = models.FloatField(verbose_name="볼린저밴드 상단", null=True, blank=True)
    BB_Middle = models.FloatField(verbose_name="볼린저밴드 중간", null=True, blank=True)
    BB_Lower = models.FloatField(verbose_name="볼린저밴드 하단", null=True, blank=True)
    BB_Width = models.FloatField(verbose_name="볼린저밴드 폭", null=True, blank=True)
    BB_PercentB = models.FloatField(verbose_name="볼린저밴드 %B", null=True, blank=True)

    # MACD
    MACD = models.FloatField(verbose_name="MACD", null=True, blank=True)
    MACD_Signal = models.FloatField(verbose_name="MACD Signal", null=True, blank=True)
    MACD_Hist = models.FloatField(verbose_name="MACD Histogram", null=True, blank=True)

    # RSI
    RSI6 = models.FloatField(verbose_name="RSI (6일)", null=True, blank=True)
    RSI14 = models.FloatField(verbose_name="RSI (14일)", null=True, blank=True)
    RSI28 = models.FloatField(verbose_name="RSI (28일)", null=True, blank=True)


    # 스토캐스틱 (%K, %D)
    STOCH_K = models.FloatField(verbose_name="Stochastic %K (Slow)", null=True, blank=True) # Slow %K
    STOCH_D = models.FloatField(verbose_name="Stochastic %D (Slow)", null=True, blank=True) # Slow %D
    STOCH_SLOW_K = models.FloatField(verbose_name="Stochastic %K (Fast)", null=True, blank=True) # Fast %K (기존 필드명 유지, 의미상 Fast로 사용)
    STOCH_SLOW_D = models.FloatField(verbose_name="Stochastic %D (Fast)", null=True, blank=True) # Fast %D (기존 필드명 유지, 의미상 Fast로 사용)


    # 기타 지표 (ipynb에서 사용하는 것들)
    ATR14 = models.FloatField(verbose_name="ATR (14일)", null=True, blank=True) 
    ADX14 = models.FloatField(verbose_name="ADX (14일)", null=True, blank=True) 
    DMP14 = models.FloatField(verbose_name="DI+ (14일)", null=True, blank=True) 
    DMN14 = models.FloatField(verbose_name="DI- (14일)", null=True, blank=True) 
    CCI14 = models.FloatField(verbose_name="CCI (14일)", null=True, blank=True) 
    MFI14 = models.FloatField(verbose_name="MFI (14일)", null=True, blank=True) 
    OBV = models.FloatField(verbose_name="OBV", null=True, blank=True) 
    WilliamsR14 = models.FloatField(verbose_name="Williams %R (14일)", null=True, blank=True)
    Momentum = models.FloatField(verbose_name="Momentum (10일)", null=True, blank=True) # 기간 명시
    ROC = models.FloatField(verbose_name="ROC (10일)", null=True, blank=True) 
    TRIX = models.FloatField(verbose_name="TRIX (14일, Signal 9일)", null=True, blank=True) # 기간 명시
    VR = models.FloatField(verbose_name="VR (20일)", null=True, blank=True) 
    PSY = models.FloatField(verbose_name="PSY (12일)", null=True, blank=True) 

    # 시장 데이터 (해당 종목이 속한 시장의 지수 및 변동률)
    Market_Index_Close = models.FloatField(verbose_name="해당일 시장 지수 종가", null=True, blank=True)
    Market_Index_Change = models.FloatField(verbose_name="해당일 시장 지수 등락률", null=True, blank=True)
    
    # 거시 경제 데이터 (예: 환율)
    USD_KRW_Close = models.FloatField(verbose_name="해당일 원/달러 환율 종가", null=True, blank=True)
    USD_KRW_Change = models.FloatField(verbose_name="해당일 원/달러 환율 등락률", null=True, blank=True)

    # 로그 변환된 값 (ipynb에서 사용 시)
    log_close_price = models.FloatField(verbose_name="로그 변환 종가", null=True, blank=True)
    log_volume = models.FloatField(verbose_name="로그 변환 거래량", null=True, blank=True)
    
    # 기타 파생 변수 (예: 고가-저가, 시가-종가 등)
    HL_spread = models.FloatField(verbose_name="고가-저가 Spread", null=True, blank=True)
    OC_spread = models.FloatField(verbose_name="시가-종가 Spread", null=True, blank=True)

    # 펀더멘털 파생 플래그 컬럼 추가
    per_is_high = models.BooleanField(verbose_name="PER 고평가 여부", null=True, blank=True)
    per_is_low = models.BooleanField(verbose_name="PER 저평가 여부", null=True, blank=True)
    per_is_zero = models.BooleanField(verbose_name="PER 0 여부", null=True, blank=True)
    per_is_nan = models.BooleanField(verbose_name="PER NaN 여부", null=True, blank=True)
    pbr_is_high = models.BooleanField(verbose_name="PBR 고평가 여부", null=True, blank=True)
    pbr_is_low = models.BooleanField(verbose_name="PBR 저평가 여부", null=True, blank=True)
    pbr_is_zero = models.BooleanField(verbose_name="PBR 0 여부", null=True, blank=True)
    pbr_is_nan = models.BooleanField(verbose_name="PBR NaN 여부", null=True, blank=True)
    market_cap_is_nan = models.BooleanField(verbose_name="시가총액 NaN 여부", null=True, blank=True)


    # 데이터 생성/수정 시간
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="생성 시간")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="수정 시간")

    class Meta:
        verbose_name = "일별 주가 및 피처"
        verbose_name_plural = "일별 주가 및 피처 목록"
        unique_together = ('stock_code', 'date') 
        ordering = ['-date', 'stock_code']
        indexes = [
            models.Index(fields=['stock_code', 'date'], name='stock_date_idx'),
            models.Index(fields=['stock_name']),
            models.Index(fields=['market_name']),
        ]
        
    def save(self, *args, **kwargs):
        # 등락률 및 대비 가격 계산
        if self.close_price is not None and self.previous_day_close_price is not None:
            self.change_value = self.close_price - self.previous_day_close_price
            if self.previous_day_close_price != 0:
                self.change = (self.change_value / self.previous_day_close_price) 
            else:
                self.change = 0.0
        elif self.close_price is not None and self.previous_day_close_price is None: # 전일 종가가 없는 첫날 데이터
            self.change = 0.0
            self.change_value = 0.0
        
        # HL_spread, OC_spread 계산
        if self.high_price is not None and self.low_price is not None:
            self.HL_spread = self.high_price - self.low_price
        if self.open_price is not None and self.close_price is not None:
            self.OC_spread = self.close_price - self.open_price

        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.date} {self.stock_name}({self.stock_code}) 종가: {self.close_price}"

class PredictedStockPrice(models.Model):
    # ... (기존과 거의 동일, analysis_type에 'ipynb_lstm' 등 추가 가능)
    ANALYSIS_TYPE_CHOICES = [
        ('technical', '기술적 분석 기반 LSTM'),
        ('comprehensive', '종합 분석 기반 LSTM'), 
        ('ipynb_kospi_lstm', 'ipynb KOSPI LSTM'), 
        ('ipynb_kosdaq_lstm', 'ipynb KOSDAQ LSTM'), 
    ]
    stock_code = models.CharField(max_length=20, verbose_name="종목 코드")
    stock_name = models.CharField(max_length=100, verbose_name="종목명")
    market_name = models.CharField(max_length=10, verbose_name="시장 구분", blank=True, null=True)
    prediction_base_date = models.DateField(verbose_name="예측 기준일")
    predicted_date = models.DateField(verbose_name="예측 대상일")
    predicted_price = models.FloatField(verbose_name="예측 종가")
    analysis_type = models.CharField(
        max_length=100,
        choices=ANALYSIS_TYPE_CHOICES,
        default='technical',
        verbose_name="분석 유형"
    )
    model_version = models.CharField(max_length=50, verbose_name="사용 모델 버전", blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="생성 시간")

    class Meta:
        verbose_name = "주가 예측 결과"
        verbose_name_plural = "주가 예측 결과 목록"
        unique_together = ('stock_code', 'prediction_base_date', 'predicted_date', 'analysis_type')
        ordering = ['-prediction_base_date', 'stock_code', 'predicted_date']
        indexes = [
            models.Index(fields=['stock_code', 'prediction_base_date', 'analysis_type', 'predicted_date'], name='pred_price_lookup_idx'),
            models.Index(fields=['prediction_base_date']),
        ]

    def __str__(self):
        return f"{self.stock_name}({self.stock_code}) - 기준일:{self.prediction_base_date} -> 예측일:{self.predicted_date} = {self.predicted_price} ({self.get_analysis_type_display()})"


class FavoriteStock(models.Model):
    # ... (기존과 동일)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='favorite_stocks',
        verbose_name="사용자"
    )
    stock_code = models.CharField(max_length=20, verbose_name="종목 코드")
    stock_name = models.CharField(max_length=100, verbose_name="종목명", blank=True)
    market_name = models.CharField(max_length=20, verbose_name="시장 구분", blank=True)
    added_at = models.DateTimeField(auto_now_add=True, verbose_name="추가된 날짜")

    class Meta:
        verbose_name = "관심 종목"
        verbose_name_plural = "관심 종목 목록"
        unique_together = ('user', 'stock_code') 
        ordering = ['user', '-added_at']

    def __str__(self):
        return f"{self.user.username} - {self.stock_name} ({self.stock_code})"
