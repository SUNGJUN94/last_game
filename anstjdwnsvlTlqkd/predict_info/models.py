from django.db import models
from django.utils import timezone

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
        """등락 및 등락률을 계산하여 필드에 설정합니다."""
        if self.previous_day_close_price is not None and self.close_price is not None:
            self.change_value = self.close_price - self.previous_day_close_price
            if self.previous_day_close_price != 0:
                self.change_percent = ((self.close_price - self.previous_day_close_price) / self.previous_day_close_price) * 100
            else:
                # 전일 종가가 0이고 현재 종가도 0이면 등락률 0, 아니면 무한대 또는 특정 값
                self.change_percent = 0 if self.close_price == self.previous_day_close_price else float('inf')
        else:
            self.change_value = None
            self.change_percent = None

    def save(self, *args, **kwargs):
        self.calculate_changes() # 저장 전에 항상 등락률 계산
        super().save(*args, **kwargs)


class StockPrice(models.Model):
    """
    개별 종목의 일별 가격 정보를 저장하는 모델
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

    class Meta:
        verbose_name = "개별 종목 가격"
        verbose_name_plural = "개별 종목 가격 목록"
        unique_together = ('stock_code', 'date')
        ordering = ['-date', 'stock_name']
        indexes = [
            models.Index(fields=['date', 'market_name', '-change_percent']),
            models.Index(fields=['date', 'stock_name']),
            models.Index(fields=['date', 'stock_code']),
        ]

    def __str__(self):
        return f"{self.date} - {self.stock_name} ({self.stock_code}): {self.close_price}"

    def calculate_changes(self):
        """등락 및 등락률을 계산하여 필드에 설정합니다."""
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
        self.calculate_changes() # 저장 전에 항상 등락률 계산
        super().save(*args, **kwargs)
