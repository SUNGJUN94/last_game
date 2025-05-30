# community/jobs.py

from apscheduler.schedulers.background import BackgroundScheduler
from django_apscheduler.jobstores import DjangoJobStore, register_events
from django.core.management import call_command
from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone as django_timezone
from datetime import datetime, timedelta
import logging
import requests
# from dateutil import parser as date_parser # Replaced with datetime.strptime for specific format
import html
import re

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler(timezone=settings.TIME_ZONE)
scheduler.add_jobstore(DjangoJobStore(), "default")

User = get_user_model()

# --- DART 공시 관련 작업 ---
def fetch_dart_job():
    """
    (스케줄러에 의해 주기적으로 실행됨)
    1. DART API에서 최신 공시를 가져와 DartDisclosure 테이블을 업데이트합니다.
    2. DartDisclosure 테이블의 내용을 FreeBoard로 포스팅합니다.
    """
    try:
        logger.info("스케줄러 실행: DART 공시 업데이트 작업 시작 (fetch_disclosure 명령어 호출)...")
        call_command('fetch_disclosure', '--days', '1')
        logger.info("스케줄러 실행: DART 공시 업데이트 작업 완료 (fetch_disclosure 명령어 호출 완료).")

        logger.info("스케줄러 실행: DartDisclosure -> FreeBoard 포스팅 작업 시작...")
        auto_post_disclosures_to_freeboard() # Pass initial_run if needed, or determine context
        logger.info("스케줄러 실행: DartDisclosure -> FreeBoard 포스팅 작업 완료.")
    except Exception as e:
        logger.error(f"스케줄러 실행: DART 공시 처리 작업 중 오류: {e}")

def auto_post_disclosures_to_freeboard(initial_run=False):
    """
    DartDisclosure 테이블의 내용을 FreeBoard 게시글로 자동 등록/업데이트합니다.
    """
    from .models import DartDisclosure, FreeBoard # 지연 import

    log_prefix = "초기 실행 - 공시 자동 포스팅:" if initial_run else "주기 실행 - 공시 자동 포스팅:"
    logger.info(f"{log_prefix} DartDisclosure 내용을 FreeBoard로 자동 포스팅/업데이트 시작...")

    author = User.objects.filter(is_superuser=True).order_by('pk').first() or \
               User.objects.order_by('pk').first()

    if not author:
        logger.error(f"{log_prefix} FreeBoard에 글을 작성할 사용자를 찾을 수 없습니다. 자동 포스팅을 중단합니다.")
        return

    logger.info(f"{log_prefix} 작성자: '{getattr(author, author.USERNAME_FIELD, author.pk)}'")
    
    updated_count = 0
    newly_posted_count = 0

    # 처리할 공시 범위 결정
    if initial_run and settings.DEBUG:
        # 서버 시작 & 디버그 모드: 최근 10개 공시만 FreeBoard로 포스팅 시도
        disclosures_to_process_qs = DartDisclosure.objects.order_by('-rcept_dt', '-rcept_no')[:10]
        logger.info(f"{log_prefix} TEST_MODE_LIMIT: FreeBoard 포스팅 대상 공시를 최근 10개로 제한합니다.")
    else:
        # 일반 실행 또는 운영 모드 서버 시작: 최근 1일치 공시 처리
        one_day_ago = django_timezone.now().date() - timedelta(days=1)
        disclosures_to_process_qs = DartDisclosure.objects.filter(rcept_dt__gte=one_day_ago).order_by('-rcept_dt', '-rcept_no')
    
    disclosures_to_process = list(disclosures_to_process_qs)
    logger.info(f"{log_prefix} FreeBoard로 옮기거나 업데이트할 대상 DartDisclosure 수: {len(disclosures_to_process)}")

    for dart_item in disclosures_to_process:
        try:
            # reg_dt는 auto_now_add=True이므로 defaults에서 설정해도 반영 안될 수 있음.
            # 필요시 생성 후 별도 업데이트 또는 모델 필드 수정 고려.
            # 여기서는 DART의 rcept_dt를 기준으로 함 (make_aware 처리)
            publication_time = django_timezone.make_aware(
                datetime.combine(dart_item.rcept_dt, datetime.min.time()),
                timezone=django_timezone.get_default_timezone() # 또는 적절한 timezone
            )

            freeboard_post, created = FreeBoard.objects.get_or_create(
                dart_rcept_no=dart_item.rcept_no, # DART 공시의 고유 ID로 중복 방지
                defaults={
                    'user': author,
                    'title': dart_item.report_nm,
                    'content': f"회사명: {dart_item.corp_name}\n접수일자: {dart_item.rcept_dt.strftime('%Y-%m-%d')}\n\n{dart_item.document_content[:1000]}...\n\n공시 원문 보기: {dart_item.report_link}",
                    'category': 'API공시',
                    'reg_dt': publication_time, # 게시글의 등록 시간을 공시 접수일로 설정 시도
                    'is_deleted': False
                }
            )
            if created:
                # get_or_create 사용 시 reg_dt가 defaults에 있어도 auto_now_add=True면 현재시간으로 설정됨.
                # 명시적으로 공시 시간을 사용하려면, 생성 후 업데이트 필요 또는 auto_now_add=False로 변경.
                if freeboard_post.reg_dt.date() != publication_time.date(): # 생성시 시간이 다르면 업데이트
                    freeboard_post.reg_dt = publication_time
                    # freeboard_post.save(update_fields=['reg_dt']) # 주석처리: auto_now_add는 이렇게 업데이트 안됨
                newly_posted_count += 1
                logger.info(f"{log_prefix} 새 FreeBoard 게시글 등록 완료 - RceptNo: {dart_item.rcept_no}, Title: {dart_item.report_nm}")
            else:
                # 기존 게시물 업데이트 로직 (필요시 추가)
                made_change = False
                if freeboard_post.title != dart_item.report_nm: # 제목 변경 시 업데이트
                    freeboard_post.title = dart_item.report_nm
                    made_change = True
                # 내용이나 카테고리 등 다른 필드 변경 감지 및 업데이트 로직 추가 가능
                if made_change:
                    freeboard_post.save()
                    updated_count +=1
                    logger.info(f"{log_prefix} 기존 FreeBoard 게시글 업데이트 완료 - RceptNo: {dart_item.rcept_no}")


        except Exception as e:
            logger.error(f"{log_prefix} FreeBoard 처리 중 오류 발생 (rcept_no: {dart_item.rcept_no}): {e}")
    logger.info(f"{log_prefix} FreeBoard 자동 포스팅/업데이트 완료. 신규: {newly_posted_count}건, 업데이트: {updated_count}건")


# --- 네이버 뉴스 관련 작업 ---
def _call_naver_news_api_and_save(client_id, client_secret, query, display, start, log_prefix_detail):
    """
    네이버 뉴스 API를 호출하고 새로운 기사를 NewsArticle 모델에 저장하는 내부 함수.
    저장된 새 기사 수를 반환합니다.
    """
    from .models import NewsArticle # 지연 import

    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    params = {"query": query, "display": display, "sort": "date", "start": start}

    logger.info(f"{log_prefix_detail} API 호출 시작 (query: {query}, display: {display}, start: {start})")
    saved_count_this_call = 0
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        news_data = response.json()
    except requests.exceptions.Timeout:
        logger.error(f"{log_prefix_detail} 네이버 뉴스 API 요청 시간 초과 (start: {start})")
        return 0
    except requests.exceptions.RequestException as e:
        logger.error(f"{log_prefix_detail} 네이버 뉴스 API 요청 실패 (start: {start}): {e}")
        return 0
    except ValueError: 
        logger.error(f"{log_prefix_detail} 네이버 뉴스 API 응답 JSON 디코딩 실패 (start: {start})")
        return 0

    items = news_data.get("items", [])
    if not items:
        logger.info(f"{log_prefix_detail} API 응답에 뉴스 항목이 없습니다 (start: {start}).")
        return 0

    for item_idx, item in enumerate(items):
        original_link = item.get('originallink')
        if not original_link: 
            original_link = item.get('link')
        
        if not original_link:
            logger.warning(f"{log_prefix_detail} 항목에 원본 링크가 없어 건너<0xEB><0x8D>니다: {item.get('title')}")
            continue

        raw_title = item.get("title", "")
        title = html.unescape(re.sub(r'<[^>]+>', '', raw_title))
        
        raw_description = item.get("description", "")
        description = html.unescape(re.sub(r'<[^>]+>', '', raw_description))
        
        pub_date_str = item.get("pubDate")
        pub_date_to_save = None

        try:
            dt_aware = datetime.strptime(pub_date_str, '%a, %d %b %Y %H:%M:%S %z')
            if not settings.USE_TZ: # USE_TZ=False 이면 naive datetime으로 변환
                dt_aware = dt_aware.astimezone(django_timezone.get_current_timezone()).replace(tzinfo=None)
            # USE_TZ=True 이면 dt_aware (aware datetime) 그대로 사용
            pub_date_to_save = dt_aware
        except (ValueError, TypeError) as e:
            logger.warning(f"{log_prefix_detail} 뉴스 발행일 파싱 오류 (제목: {title}, pubDate: {pub_date_str}): {e}. 현재 시간으로 대체.")
            pub_date_to_save = django_timezone.now() # 오류 시 현재 시간으로 대체

        try:
            # NewsArticle 모델에 original_link가 unique=True로 설정되어 있다고 가정
            article, created = NewsArticle.objects.get_or_create(
                original_link=original_link,
                defaults={
                    'title': title,
                    'naver_link': item.get('link'),
                    'description': description,
                    'pub_date': pub_date_to_save,
                }
            )
            if created:
                saved_count_this_call += 1
                logger.info(f"{log_prefix_detail} 새 뉴스 NewsArticle 저장: {title} ({original_link})")
        except Exception as e:
            logger.error(f"{log_prefix_detail} NewsArticle 저장 중 오류 발생 '{title}': {e}")

    logger.info(f"{log_prefix_detail} API 호출 (start: {start}, display: {display}): 총 {len(items)}건 수신, {saved_count_this_call}건 NewsArticle 신규 저장.")
    return saved_count_this_call

def auto_post_news_to_freeboard(initial_run=False):
    """
    NewsArticle 테이블의 내용을 FreeBoard 게시글('실시간뉴스' 카테고리)로 자동 등록합니다.
    """
    from .models import NewsArticle, FreeBoard # 지연 import

    log_prefix = "초기 실행 - 뉴스 자동 포스팅:" if initial_run else "주기 실행 - 뉴스 자동 포스팅:"
    logger.info(f"{log_prefix} NewsArticle 내용을 FreeBoard로 자동 포스팅 시작...")

    author = User.objects.filter(is_superuser=True).order_by('pk').first() or \
               User.objects.order_by('pk').first()

    if not author:
        logger.error(f"{log_prefix} FreeBoard에 글을 작성할 사용자를 찾을 수 없습니다. 자동 포스팅을 중단합니다.")
        return
    
    logger.info(f"{log_prefix} 작성자: '{getattr(author, author.USERNAME_FIELD, author.pk)}'")

    newly_posted_to_freeboard_count = 0

    # 처리할 NewsArticle 범위 결정
    if initial_run and settings.DEBUG:
        # 서버 시작 & 디버그 모드: 최근 저장된 NewsArticle 10개 처리 시도
        news_articles_to_process = NewsArticle.objects.order_by('-created_at')[:10]
        logger.info(f"{log_prefix} TEST_MODE_LIMIT: FreeBoard 포스팅 대상 NewsArticle을 최근 저장된 10개로 제한합니다.")
    else:
        # 일반 실행 또는 운영 모드 서버 시작: 최근 24시간 내 저장된 NewsArticle 처리
        # 또는 마지막으로 FreeBoard에 포스팅한 뉴스 이후의 뉴스를 가져오는 로직도 고려 가능
        since_time = django_timezone.now() - timedelta(hours=24) 
        news_articles_to_process = NewsArticle.objects.filter(created_at__gte=since_time).order_by('-created_at')
        logger.info(f"{log_prefix} 지난 24시간 동안 저장된 NewsArticle을 FreeBoard 포스팅 대상으로 합니다.")

    logger.info(f"{log_prefix} FreeBoard로 옮길 대상 NewsArticle 수: {len(news_articles_to_process)}")

    for news_item in news_articles_to_process:
        try:
            # FreeBoard에 해당 뉴스가 이미 포스팅되었는지 확인 (original_link를 사용)
            # 더 강력한 중복 방지를 위해 FreeBoard 모델에 news_article_id 같은 외래키나,
            # news_original_link 필드를 추가하고 unique 제약조건을 거는 것이 좋음.
            # 여기서는 content 내부에 고유 식별자를 포함하여 확인.
            unique_marker_in_content = f"[원문링크:{news_item.original_link}]"
            
            if FreeBoard.objects.filter(category='실시간뉴스', content__contains=unique_marker_in_content).exists():
                logger.info(f"{log_prefix} 이미 FreeBoard에 포스팅된 뉴스입니다: {news_item.title}")
                continue

            # FreeBoard 게시글 내용 구성
            content = f"{news_item.description}\n\n{unique_marker_in_content}\n게시일: {news_item.pub_date.strftime('%Y-%m-%d %H:%M') if news_item.pub_date else '날짜 정보 없음'}"
            
            # reg_dt는 auto_now_add=True이므로, 뉴스 발행 시간을 사용하려면 생성 후 업데이트 또는 모델 변경 필요.
            # 여기서는 FreeBoard의 reg_dt는 자동 생성되도록 둠.
            FreeBoard.objects.create(
                user=author,
                title=news_item.title,
                content=content,
                category='실시간뉴스', # models.py의 FreeBoard.CATEGORY_CHOICES에 '실시간뉴스'가 있어야 함
                # image 등 다른 필드는 필요에 따라 추가
            )
            newly_posted_to_freeboard_count += 1
            logger.info(f"{log_prefix} 새 FreeBoard 뉴스 게시글 등록 완료: {news_item.title}")

        except Exception as e:
            logger.error(f"{log_prefix} FreeBoard 뉴스 포스팅 중 오류 발생 (NewsArticle ID: {news_item.id}): {e}")

    logger.info(f"{log_prefix} FreeBoard 뉴스 자동 포스팅 완료. 신규 FreeBoard 게시글: {newly_posted_to_freeboard_count}건")


def fetch_and_save_naver_news_job(initial_run=False):
    """
    네이버 뉴스를 NewsArticle에 저장하고, 그 결과를 FreeBoard에도 포스팅합니다.
    """
    log_prefix = "초기 실행 - 네이버 뉴스 작업:" if initial_run else "주기 실행 - 네이버 뉴스 작업:"
    logger.info(f"{log_prefix} 시작...")
    
    client_id = getattr(settings, "NAVER_NEWS_API_CLIENT_ID", None)
    client_secret = getattr(settings, "NAVER_NEWS_API_CLIENT_SECRET", None)

    if not client_id or not client_secret:
        logger.error(f"{log_prefix} NAVER_NEWS_API_CLIENT_ID 또는 NAVER_NEWS_API_CLIENT_SECRET이 settings.py에 설정되지 않았습니다.")
        return

    query = "증시 OR 주식 OR 코스피 OR 코스닥 OR 경제 OR 금융 OR 투자 OR 기업"
    
    display_count_first_attempt = 10 if initial_run and settings.DEBUG else (20 if initial_run else 10) # 초기 실행 시 조금 더, 평소엔 10개
    display_count_second_attempt = 10

    # 첫 번째 시도 (NewsArticle 저장)
    logger.info(f"{log_prefix} 첫 번째 NewsArticle 저장 시도...")
    saved_to_newsarticle_attempt1 = _call_naver_news_api_and_save(
        client_id, client_secret, query, display_count_first_attempt, 1, f"{log_prefix} 첫 번째 NewsArticle 저장 -"
    )

    if saved_to_newsarticle_attempt1 == 0:
        logger.warning(f"{log_prefix} 첫 번째 시도에서 새로운 뉴스를 NewsArticle에 저장하지 못했습니다. 이전 뉴스를 가져옵니다.")
        start_index_attempt2 = 1 + display_count_first_attempt 
        logger.info(f"{log_prefix} 두 번째 NewsArticle 저장 시도 (start: {start_index_attempt2})...")
        _call_naver_news_api_and_save( # 두 번째 시도에서 저장된 카운트는 현재 사용하지 않음
            client_id, client_secret, query, display_count_second_attempt, start_index_attempt2, f"{log_prefix} 두 번째 NewsArticle 저장 -"
        )
    else:
        logger.info(f"{log_prefix} 첫 번째 시도에서 {saved_to_newsarticle_attempt1}건의 새 뉴스를 NewsArticle에 저장했습니다.")

    # NewsArticle에 저장된 내용을 FreeBoard로 포스팅
    logger.info(f"{log_prefix} NewsArticle -> FreeBoard 포스팅 작업 호출...")
    auto_post_news_to_freeboard(initial_run=initial_run)
    
    logger.info(f"{log_prefix} 완료.")


def start_scheduler():
    if settings.DEBUG: # 디버그 모드일 때 기존 작업 제거 (재시작 시 중복 방지)
        try:
            if scheduler.get_job('fetch_dart_disclosure_job'): scheduler.remove_job('fetch_dart_disclosure_job')
            if scheduler.get_job('fetch_naver_news_job'): scheduler.remove_job('fetch_naver_news_job')
            logger.info("스케줄러: 기존 작업들 제거 시도 완료 (DEBUG 모드).")
        except Exception as e:
            logger.warning(f"스케줄러: 기존 작업 제거 중 오류: {e}")

    try:
        # DART 공시 처리 작업 스케줄링
        scheduler.add_job(
            fetch_dart_job, # DART 공시 가져오기 및 FreeBoard 포스팅 포함
            trigger='interval', hours=2, id='fetch_dart_disclosure_job',
            max_instances=1, replace_existing=True, misfire_grace_time=60*10 
        )
        logger.info("스케줄러: 'fetch_dart_disclosure_job' 작업 추가 (2시간 간격).")

        # 네이버 뉴스 처리 작업 스케줄링
        scheduler.add_job(
            fetch_and_save_naver_news_job, # 네이버 뉴스 가져오기 및 FreeBoard 포스팅 포함
            trigger='interval', minutes=30, id='fetch_naver_news_job', 
            max_instances=1, replace_existing=True, misfire_grace_time=60*5 
        )
        logger.info("스케줄러: 'fetch_naver_news_job' 작업 추가 (30분 간격).")

    except Exception as e:
        logger.error(f"스케줄러: 작업 추가 중 오류: {e}")

    register_events(scheduler) # Django-APScheduler 이벤트 리스너 등록

    if not scheduler.running:
        try:
            logger.info("스케줄러: 스케줄러를 시작합니다...")
            scheduler.start()
            # 서버 시작 시 초기 작업 실행 (선택 사항)
            # logger.info("스케줄러: 서버 시작 - 초기 DART 공시 처리 실행...")
            # fetch_dart_job() # auto_post_disclosures_to_freeboard(initial_run=True) 내부 호출 고려
            # logger.info("스케줄러: 서버 시작 - 초기 네이버 뉴스 처리 실행...")
            # fetch_and_save_naver_news_job(initial_run=True)
        except KeyboardInterrupt: 
            logger.info("스케줄러: 사용자에 의해 중지됨."); scheduler.shutdown()
        except Exception as e: 
            logger.error(f"스케줄러: 시작 중 오류: {e}")
    else:
        logger.info("스케줄러: 이미 실행 중입니다.")

# 만약 apps.py 등에서 서버 시작 시 초기 작업을 실행하고 싶다면,
# start_scheduler() 호출 후 또는 별도로 다음 함수들을 initial_run=True로 호출할 수 있습니다.
# 예:
# def run_initial_jobs():
#     logger.info("초기 작업 실행: DART 공시...")
#     auto_post_disclosures_to_freeboard(initial_run=True) # fetch_dart_job 대신 직접 호출 가능
#     logger.info("초기 작업 실행: 네이버 뉴스...")
#     fetch_and_save_naver_news_job(initial_run=True)
