import torch
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path
import re

#org1
#250522_community/models 안에 구글드라이브의 kobert_curse_model 을 복사했다면 아래 코드 주석해제해도 됨
# MODEL_PATH = Path(__file__).parent / "models" / "kobert_curse_model"
# tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
# model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# CURSE_WORDS_FILE = Path(__file__).parent / "text" / "edit_curse_words.txt"

# def load_curse_words(file_path):
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             curse_words = [line.strip() for line in file if line.strip()]
#         return curse_words
#     except FileNotFoundError:
#         print(f"욕설 단어 파일을 찾을 수 없습니다: {file_path}")
#         return []

# CURSE_WORDS = load_curse_words(CURSE_WORDS_FILE)
# PATTERNS = [re.compile(r'\b' + re.escape(word) + r'[\w]*\b', re.IGNORECASE) for word in CURSE_WORDS]

# def detect_and_replace_curse(text):
#     for pattern in PATTERNS:
#         text = pattern.sub("**", text)

#     inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     print(f"Logits: {logits}")
#     probabilities = torch.softmax(logits, dim=1)
#     prob_curse = probabilities[0][1].item()
#     print(f"욕설 확률: {prob_curse}")

#     if prob_curse > 0.35:
#         text = "**" * len(text.split())

#     return text

# from django.db import models

# class CurseWord(models.Model):
#     word = models.CharField(max_length=100, unique=True)
#     created_at = models.DateTimeField(auto_now_add=True)

#     def __str__(self):
#         return self.word

#     class Meta:
#         db_table = 'curse_words'
#org2
####################
####################
#edit1
#250522_community/models 안이 비어있더라도 욕설필터링 최소구현은 되게끔 edit부분에 구현함
# 욕설 단어 파일 경로
CURSE_WORDS_FILE_TEXT = Path(__file__).parent / "text" / "edit_curse_words.txt"

def load_curse_words(file_path):
    """
    edit_curse_words.txt에서 욕설 단어를 로드합니다.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            curse_words = [line.strip() for line in file if line.strip()]
        return curse_words
    except FileNotFoundError:
        print(f"욕설 단어 파일을 찾을 수 없습니다: {file_path}")
        return []

# 욕설 단어와 정규 표현식 패턴 로드
CURSE_WORDS = load_curse_words(CURSE_WORDS_FILE_TEXT)
PATTERNS = [re.compile(r'\b' + re.escape(word) + r'[\w]*\b', re.IGNORECASE) for word in CURSE_WORDS]

def detect_and_replace_curse(text):
    """
    텍스트에서 edit_curse_words.txt의 욕설 단어를 찾아 **로 대체합니다.
    Args:
        text (str): 입력 텍스트
    Returns:
        str: 욕설이 **로 대체된 텍스트
    """
    if not text:
        return text
    
    # 정규 표현식으로 욕설 단어 대체
    for pattern in PATTERNS:
        text = pattern.sub("**", text)
    
    return text
#edit2