from django import template

register = template.Library()

@register.filter
def to_eok(value):
    try:
        value = int(value)
        if value >= 1000000000000:  # 1조 이상
            jo = value // 1000000000000
            eok = (value % 1000000000000) / 100000000
            return f"{jo}조 {eok:.1f}억"
        else:
            eok = value / 100000000
            return f"{eok:.1f}억"
    except (ValueError, TypeError):
        return "-" 