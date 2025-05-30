# predict_info/admin.py
from django.contrib import admin
from .models import MarketIndex, StockPrice, PredictedStockPrice, FavoriteStock # FavoriteStock 임포트
from django.utils.html import format_html # 숫자를 보기 좋게 포맷팅하기 위해 추가 (선택 사항)

@admin.register(MarketIndex)
class MarketIndexAdmin(admin.ModelAdmin):
    list_display = ('date', 'market_name', 'close_price', 'change_percent', 'volume', 'trade_value')
    list_filter = ('market_name', 'date')
    search_fields = ('market_name',)

@admin.register(StockPrice)
class StockPriceAdmin(admin.ModelAdmin):
    # 기존 list_display에서 'change_percent'와 'trade_value'를 메소드 이름으로 변경
    list_display = ('date', 'stock_name', 'stock_code', 'market_name', 'close_price', 
                    'display_change_percent', 'volume', 'display_trade_value', 
                    'market_cap', 'per', 'pbr')
    list_filter = ('market_name', 'date')
    search_fields = ('stock_name', 'stock_code')
    ordering = ('-date', 'stock_name')

    def display_change_percent(self, obj):
        if obj.change is not None:
            # StockPrice 모델의 'change' 필드는 이미 소수점 형태의 등락률임 (예: 0.05)
            # 이를 퍼센트로 변환하여 표시
            return f"{obj.change * 100:.2f}%"
        return None
    display_change_percent.short_description = "등락률 (%)" # Admin 페이지에 표시될 컬럼명

    def display_trade_value(self, obj):
        if obj.close_price is not None and obj.volume is not None:
            trade_val = obj.close_price * obj.volume
            # 숫자를 보기 좋게 쉼표 등으로 포맷팅 (선택 사항)
            try:
                # 예를 들어 1,000,000,000 처럼 표시 (Python 3.6+ f-string)
                return format_html(f"{trade_val:,.0f}") 
            except (ValueError, TypeError):
                return trade_val # 포맷팅 실패 시 원래 값 반환
        return None
    display_trade_value.short_description = "거래대금" # Admin 페이지에 표시될 컬럼명
    display_trade_value.admin_order_field = None # 거래대금은 계산 필드이므로 정렬 미지원 명시 (선택)


@admin.register(PredictedStockPrice)
class PredictedStockPriceAdmin(admin.ModelAdmin):
    list_display = ('prediction_base_date', 'stock_name', 'stock_code', 'predicted_date', 'predicted_price', 'analysis_type', 'created_at')
    list_filter = ('analysis_type', 'prediction_base_date', 'market_name')
    search_fields = ('stock_name', 'stock_code')
    ordering = ('-prediction_base_date', 'stock_code', 'predicted_date')

@admin.register(FavoriteStock)
class FavoriteStockAdmin(admin.ModelAdmin):
    list_display = ('user', 'stock_name', 'stock_code', 'market_name', 'added_at')
    list_filter = ('user', 'market_name')
    search_fields = ('user__username', 'stock_name', 'stock_code')
    ordering = ('-added_at',)