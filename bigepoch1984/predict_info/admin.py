from django.contrib import admin
from .models import MarketIndex, StockPrice

@admin.register(MarketIndex)
class MarketIndexAdmin(admin.ModelAdmin):
    list_display = ('date', 'market_name', 'close_price', 'previous_day_close_price', 'change_value', 'change_percent', 'volume', 'trade_value')
    list_filter = ('market_name', 'date')
    search_fields = ('market_name',)
    ordering = ('-date', 'market_name')

@admin.register(StockPrice)
class StockPriceAdmin(admin.ModelAdmin):
    list_display = ('date', 'stock_name', 'stock_code', 'market_name', 'close_price', 'previous_day_close_price', 'change_percent', 'volume', 'trade_value')
    list_filter = ('market_name', 'date')
    search_fields = ('stock_name', 'stock_code')
    ordering = ('-date', 'stock_name')
    # 많은 데이터가 예상되므로 list_per_page 조절 가능
    list_per_page = 25 
