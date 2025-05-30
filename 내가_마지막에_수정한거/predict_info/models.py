# predict_info/models.py
from django.db import models
from django.utils import timezone
from django.conf import settings # AUTH_USER_MODEL 사용을 위해 추가

class MarketIndex(models.Model):
    """
    일별 시장 지수(코스피, 코스닥) 정보를 저장하는 모델
    """
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

    def __str__(self):
        return f"{self.date} - {self.get_market_name_display()}: {self.close_price}"

    def calculate_changes(self):
        if self.previous_day_close_price is not None and self.close_price is not None:
            self.change_value = self.close_price - self.previous_day_close_price
            if self.previous_day_close_price != 0:
                self.change_percent = ((self.close_price - self.previous_day_close_price) / self.previous_day_close_price) * 100
            else:
                self.change_percent = 0 if self.close_price == self.previous_day_close_price else float('inf')
        else:
            self.change_value = None
            self.change_percent = None

    def save(self, *args, **kwargs):
        self.calculate_changes()
        super().save(*args, **kwargs)


class StockPrice(models.Model):
    """
    개별 종목의 일별 가격, 투자자별 거래량 및 주요 펀더멘털 정보를 저장하는 모델
    """
    stock_code = models.CharField(max_length=20, verbose_name="종목 코드")
    stock_name = models.CharField(max_length=100, verbose_name="종목명")
    market_name = models.CharField(max_length=10, verbose_name="시장 구분", help_text="예: KOSPI, KOSDAQ")
    date = models.DateField(verbose_name="날짜")
    open_price = models.FloatField(verbose_name="시가", null=True, blank=True)
    high_price = models.FloatField(verbose_name="고가", null=True, blank=True)
    low_price = models.FloatField(verbose_name="저가", null=True, blank=True)
    close_price = models.FloatField(verbose_name="종가")
    previous_day_close_price = models.FloatField(verbose_name="전일 종가", null=True, blank=True)
    change_value = models.FloatField(verbose_name="대비", null=True, blank=True)
    change_percent = models.FloatField(verbose_name="등락률 (%)", null=True, blank=True)
    volume = models.BigIntegerField(verbose_name="거래량", null=True, blank=True)
    trade_value = models.BigIntegerField(verbose_name="거래대금", null=True, blank=True)
    indi_volume = models.BigIntegerField(verbose_name="개인 순매수 거래량", null=True, blank=True, help_text="개인 투자자 순매수량 (주)")
    foreign_volume = models.BigIntegerField(verbose_name="외국인 순매수 거래량", null=True, blank=True, help_text="외국인 투자자 순매수량 (주)")
    organ_volume = models.BigIntegerField(verbose_name="기관 순매수 거래량", null=True, blank=True, help_text="기관 투자자 순매수량 (주)")
    market_cap = models.BigIntegerField(verbose_name="시가총액 (원)", null=True, blank=True, help_text="해당일의 시가총액 또는 최근 스냅샷 값")
    per = models.FloatField(verbose_name="PER", null=True, blank=True, help_text="Price Earning Ratio")
    pbr = models.FloatField(verbose_name="PBR", null=True, blank=True, help_text="Price Book-value Ratio")
    eps = models.FloatField(verbose_name="EPS", null=True, blank=True, help_text="Earnings Per Share")
    bps = models.FloatField(verbose_name="BPS", null=True, blank=True, help_text="Book-value Per Share")
    dps = models.FloatField(verbose_name="DPS", null=True, blank=True, help_text="Dividend Per Share")
    roe = models.FloatField(verbose_name="ROE", null=True, blank=True, help_text="Return On Equity")
    
    class Meta:
        verbose_name = "개별 종목 상세 정보"
        verbose_name_plural = "개별 종목 상세 정보 목록"
        unique_together = ('stock_code', 'date')
        ordering = ['-date', 'stock_name']
        indexes = [
            models.Index(fields=['date', 'market_name', '-change_percent']),
            models.Index(fields=['date', 'stock_name']),
            models.Index(fields=['date', 'stock_code']),
            models.Index(fields=['stock_code', 'date'], name='stock_date_idx'), 
        ]

    def __str__(self):
        return f"{self.date} - {self.stock_name} ({self.stock_code}): {self.close_price}"

    def calculate_changes(self):
        if self.previous_day_close_price is not None and self.close_price is not None:
            self.change_value = self.close_price - self.previous_day_close_price
            if self.previous_day_close_price != 0:
                self.change_percent = ((self.close_price - self.previous_day_close_price) / self.previous_day_close_price) * 100
            else:
                self.change_percent = 0 if self.close_price == self.previous_day_close_price else float('inf')
        else:
            self.change_value = None
            self.change_percent = None

    def save(self, *args, **kwargs):
        self.calculate_changes() 
        super().save(*args, **kwargs)

class PredictedStockPrice(models.Model):
    ANALYSIS_CHOICES = [
        ('technical', '기술적 분석'),
        ('comprehensive', '종합 분석'), 
    ]
    stock_code = models.CharField(max_length=20, verbose_name="종목 코드", db_index=True)
    stock_name = models.CharField(max_length=100, verbose_name="종목명", blank=True) 
    market_name = models.CharField(max_length=10, verbose_name="시장 구분", blank=True) 
    prediction_base_date = models.DateField(verbose_name="예측 기준일", db_index=True, help_text="이 날짜의 종가까지를 기반으로 예측이 수행됨")
    predicted_date = models.DateField(verbose_name="예측 대상일", help_text="실제 주가가 예측된 미래의 날짜")
    predicted_price = models.FloatField(verbose_name="예측 종가")
    analysis_type = models.CharField(max_length=20, choices=ANALYSIS_CHOICES, default='technical', verbose_name="분석 유형")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="생성 시각")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="수정 시각")

    class Meta:
        verbose_name = "종목별 예측 가격"
        verbose_name_plural = "종목별 예측 가격 목록"
        unique_together = ('stock_code', 'prediction_base_date', 'predicted_date', 'analysis_type')
        ordering = ['-prediction_base_date', 'stock_code', 'predicted_date']
        indexes = [
            models.Index(fields=['stock_code', 'prediction_base_date', 'analysis_type', 'predicted_date'], name='pred_price_lookup_idx'),
            models.Index(fields=['prediction_base_date']), 
        ]

    def __str__(self):
        return f"{self.stock_name}({self.stock_code}) - 기준일:{self.prediction_base_date} -> 예측일:{self.predicted_date} = {self.predicted_price} ({self.get_analysis_type_display()})"

# --- FavoriteStock 모델 추가 ---
class FavoriteStock(models.Model):
    """
    사용자별 관심 종목 정보를 저장하는 모델
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, # Django의 User 모델 또는 account 앱의 User 모델
        on_delete=models.CASCADE, 
        related_name='favorite_stocks',
        verbose_name="사용자"
    )
    stock_code = models.CharField(max_length=20, verbose_name="종목 코드")
    stock_name = models.CharField(max_length=100, verbose_name="종목명", blank=True) # 편의를 위해 저장
    market_name = models.CharField(max_length=20, verbose_name="시장 구분", blank=True) # 편의를 위해 저장
    added_at = models.DateTimeField(auto_now_add=True, verbose_name="추가된 시각")

    class Meta:
        verbose_name = "관심 종목"
        verbose_name_plural = "관심 종목 목록"
        # 한 사용자는 동일한 종목 코드를 중복해서 추가할 수 없음
        unique_together = ('user', 'stock_code') 
        ordering = ['-added_at']

    def __str__(self):
        return f"{self.user.username} - {self.stock_name} ({self.stock_code})"
