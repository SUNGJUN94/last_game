# predict_info/admin.py
from django.contrib import admin
from .models import MarketIndex, StockPrice, PredictedStockPrice, FavoriteStock # FavoriteStock 임포트

@admin.register(MarketIndex)
class MarketIndexAdmin(admin.ModelAdmin):
    list_display = ('date', 'market_name', 'close_price', 'change_percent', 'volume', 'trade_value')
    list_filter = ('market_name', 'date')
    search_fields = ('market_name',)

@admin.register(StockPrice)
class StockPriceAdmin(admin.ModelAdmin):
    list_display = ('date', 'stock_name', 'stock_code', 'market_name', 'close_price', 'change_percent', 'volume', 'trade_value', 'market_cap', 'per', 'pbr')
    list_filter = ('market_name', 'date')
    search_fields = ('stock_name', 'stock_code')
    ordering = ('-date', 'stock_name')

@admin.register(PredictedStockPrice)
class PredictedStockPriceAdmin(admin.ModelAdmin):
    list_display = ('prediction_base_date', 'stock_name', 'stock_code', 'predicted_date', 'predicted_price', 'analysis_type', 'created_at')
    list_filter = ('analysis_type', 'prediction_base_date', 'market_name')
    search_fields = ('stock_name', 'stock_code')
    ordering = ('-prediction_base_date', 'stock_code', 'predicted_date')

# --- FavoriteStock 모델 관리자 페이지 등록 ---
@admin.register(FavoriteStock)
class FavoriteStockAdmin(admin.ModelAdmin):
    list_display = ('user', 'stock_name', 'stock_code', 'market_name', 'added_at')
    list_filter = ('user', 'market_name')
    search_fields = ('user__username', 'stock_name', 'stock_code') # 사용자 이름으로도 검색 가능
    ordering = ('-added_at',)
