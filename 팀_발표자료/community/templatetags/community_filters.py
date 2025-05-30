# community/templatetags/community_filters.py
from django import template
from django.utils.safestring import mark_safe
from community.utils import detect_and_replace_curse

register = template.Library()

@register.filter
def filter_curse(value):
    return mark_safe(detect_and_replace_curse(value))