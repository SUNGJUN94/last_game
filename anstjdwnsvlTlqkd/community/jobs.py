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
from dateutil import parser as date_parser

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
        auto_post_disclosures_to_freeboard()
        logger.info("스케줄러 실행: DartDisclosure -> FreeBoard 포스팅 작업 완료.")
    except Exception as e:
        logger.error(f"스케줄러 실행: DART 공시 처리 작업 중 오류: {e}")

def auto_post_disclosures_to_freeboard(initial_run=False):
    """
    DartDisclosure 테이블의 내용을 FreeBoard 게시글로 자동 등록/업데이트합니다.
    initial_run=True일 경우 (서버 시작 시), 테스트 모드에서는 최근 10개만 처리합니다.
    initial_run=False일 경우 (스케줄러에 의한 실행), 최근 1일치 중 새로운 것만 처리합니다.
    """
    from .models import DartDisclosure, FreeBoard

    log_prefix = "초기 실행 - 자동 포스팅:" if initial_run else "주기 실행 - 자동 포스팅:"
    logger.info(f"{log_prefix} DartDisclosure 내용을 FreeBoard로 자동 포스팅/업데이트 시작...")

    author = None
    try:
        logger.info(f"{log_prefix} 슈퍼유저를 작성자로 탐색 중...")
        author = User.objects.filter(is_superuser=True).order_by('pk').first() # pk 순으로 정렬하여 일관성 유지
        if author:
            logger.info(f"{log_prefix} 슈퍼유저 '{getattr(author, author.USERNAME_FIELD, author.pk)}'를 작성자로 사용합니다.")
        else:
            logger.info(f"{log_prefix} 슈퍼유저를 찾지 못했습니다. 첫 번째 일반 사용자를 탐색합니다...")
            author = User.objects.order_by('pk').first() # pk 순으로 정렬하여 일관성 유지
            if author:
                logger.info(f"{log_prefix} 일반 사용자 '{getattr(author, author.USERNAME_FIELD, author.pk)}'를 작성자로 사용합니다.")
            else:
                logger.error(f"{log_prefix} FreeBoard에 글을 작성할 사용자를 찾을 수 없습니다. (슈퍼유저 및 일반 사용자 모두 없음). 자동 포스팅을 중단합니다.")
                return
    except Exception as e:
        logger.error(f"{log_prefix} 작성자 검색 중 오류: {e}")
        return

    updated_count = 0
    newly_posted_count = 0

    one_day_ago = django_timezone.now().date() - timedelta(days=1)
    disclosures_to_process_qs = DartDisclosure.objects.filter(rcept_dt__gte=one_day_ago).order_by('-rcept_dt', '-rcept_no')

    if initial_run and settings.DEBUG:
        TEST_MODE_LIMIT_COUNT_DISCLOSURE_POST = 10
        logger.info(f"{log_prefix} TEST_MODE_LIMIT: FreeBoard 포스팅 대상 공시를 {TEST_MODE_LIMIT_COUNT_DISCLOSURE_POST}개로 제한합니다.")
        disclosures_to_process = list(disclosures_to_process_qs[:TEST_MODE_LIMIT_COUNT_DISCLOSURE_POST])
    else:
        disclosures_to_process = list(disclosures_to_process_qs)

    logger.info(f"{log_prefix} FreeBoard로 옮기거나 업데이트할 대상 DartDisclosure 수: {len(disclosures_to_process)}")

    for dart_item in disclosures_to_process:
        try:
            freeboard_post, created = FreeBoard.objects.get_or_create(
                dart_rcept_no=dart_item.rcept_no,
                defaults={
                    'user': author,
                    'title': dart_item.report_nm,
                    'content': f"회사명: {dart_item.corp_name}\n접수일자: {dart_item.rcept_dt.strftime('%Y-%m-%d')}\n\n{dart_item.document_content[:1000]}...\n\n공시 원문 보기: {dart_item.report_link}",
                    'category': 'API공시',
                    'reg_dt': django_timezone.make_aware(datetime.combine(dart_item.rcept_dt, datetime.min.time())),
                    'is_deleted': False
                }
            )

            if created:
                newly_posted_count += 1
                logger.info(f"{log_prefix} 새 FreeBoard 게시글 등록 완료 - RceptNo: {dart_item.rcept_no}, Title: {dart_item.report_nm}")
            else:
                made_change = False
                if freeboard_post.category != 'API공시':
                    freeboard_post.category = 'API공시'
                    made_change = True
                if freeboard_post.is_deleted:
                    freeboard_post.is_deleted = False
                    made_change = True
                
                if made_change:
                    freeboard_post.save()
                    updated_count += 1
                    logger.info(f"{log_prefix} 기존 FreeBoard 게시글 업데이트 완료 - RceptNo: {dart_item.rcept_no}")

        except Exception as e:
            logger.error(f"{log_prefix} FreeBoard 처리 중 오류 발생 (rcept_no: {dart_item.rcept_no}): {e}")

    logger.info(f"{log_prefix} FreeBoard 자동 포스팅/업데이트 완료. 신규: {newly_posted_count}건, 업데이트: {updated_count}건")

# --- 네이버 뉴스 관련 작업 ---
def fetch_and_save_naver_news_job(initial_run=False):
    from .models import NewsArticle

    log_prefix = "초기 실행 - 네이버 뉴스:" if initial_run else "주기 실행 - 네이버 뉴스:"
    client_id = getattr(settings, "NAVER_NEWS_API_CLIENT_ID", None)
    client_secret = getattr(settings, "NAVER_NEWS_API_CLIENT_SECRET", None)

    if not client_id or not client_secret:
        logger.error(f"{log_prefix} API ID 또는 Secret이 settings.py에 설정되지 않았습니다.")
        return

    query = "증시 OR 주식 OR 코스피 OR 코스닥 OR 경제 OR 금융 OR 투자 OR 기업"
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}

    TEST_MODE_LIMIT_COUNT_NEWS = 10
    display_api_param = TEST_MODE_LIMIT_COUNT_NEWS if initial_run and settings.DEBUG else (50 if initial_run else 30)

    params = {"query": query, "display": display_api_param, "sort": "date", "start": 1}
    if initial_run and settings.DEBUG:
        logger.info(f"{log_prefix} TEST_MODE_LIMIT: Naver News API 'display' 파라미터를 {display_api_param}(으)로 설정합니다.")

    logger.info(f"{log_prefix} API 호출 시작 (검색어: {query}, display: {params['display']})")
    saved_count = 0
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        news_data = response.json()

        if news_data.get('items'):
            for item_idx, item in enumerate(news_data['items']):
                if initial_run and settings.DEBUG and saved_count >= TEST_MODE_LIMIT_COUNT_NEWS:
                    logger.info(f"{log_prefix} TEST_MODE_LIMIT: 저장된 뉴스 기사 수가 {TEST_MODE_LIMIT_COUNT_NEWS}개에 도달하여 중단합니다.")
                    break

                original_link = item.get('originallink')
                naver_link = item.get('link')
                title = item.get('title', '').replace('<b>', '').replace('</b>', '').replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                description = item.get('description', '').replace('<b>', '').replace('</b>', '').replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')

                pub_date_to_save = None
                try:
                    parsed_dt = date_parser.parse(item.get('pubDate'))
                    if settings.USE_TZ:
                        if django_timezone.is_naive(parsed_dt):
                            pub_date_to_save = django_timezone.make_aware(parsed_dt, timezone=django_timezone.get_default_timezone())
                        else:
                            pub_date_to_save = parsed_dt.astimezone(django_timezone.get_default_timezone())
                    else:
                        if django_timezone.is_aware(parsed_dt):
                            local_dt = parsed_dt.astimezone(django_timezone.get_default_timezone())
                            pub_date_to_save = local_dt.replace(tzinfo=None)
                        else:
                            pub_date_to_save = parsed_dt
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning(f"{log_prefix} 뉴스 발행일 파싱/변환 오류 (제목: {title}, pubDate: {item.get('pubDate')}): {e}. 현재 시간(naive)으로 대체합니다.")
                    pub_date_to_save = datetime.now()

                if original_link:
                    # NewsArticle 모델에 created_at 필드가 정상적으로 존재하고 마이그레이션 되었다면,
                    # 이 부분에서 오류가 발생하지 않아야 합니다.
                    if not NewsArticle.objects.filter(original_link=original_link).exists():
                        NewsArticle.objects.create(
                            title=title, original_link=original_link, naver_link=naver_link,
                            description=description, pub_date=pub_date_to_save
                            # created_at은 auto_now_add=True이므로 명시적으로 전달할 필요 없음
                        )
                        saved_count += 1
            logger.info(f"{log_prefix} {len(news_data['items'])}건 중 {saved_count}건 신규 저장 완료.")
        else:
            logger.info(f"{log_prefix} API로부터 가져온 뉴스 항목이 없습니다. 응답: {news_data}")
    except requests.exceptions.RequestException as e:
        logger.error(f"{log_prefix} API 요청 실패: {e}")
    except Exception as e: # 여기서 MySQL 오류 (1054, "Unknown column 'created_at'...")가 발생했습니다.
        logger.error(f"{log_prefix} 처리 중 오류 발생: {e}")

def start_scheduler():
    if settings.DEBUG:
        try:
            if scheduler.get_job('fetch_dart_disclosure_job'): scheduler.remove_job('fetch_dart_disclosure_job')
            if scheduler.get_job('fetch_naver_news_job'): scheduler.remove_job('fetch_naver_news_job')
            logger.info("스케줄러: 기존 작업들 제거 시도 완료.")
        except Exception: pass

    try:
        scheduler.add_job(
            fetch_dart_job,
            trigger='interval', hours=2, id='fetch_dart_disclosure_job',
            max_instances=1, replace_existing=True, misfire_grace_time=60*10
        )
        logger.info("스케줄러: 'fetch_dart_disclosure_job' 작업 추가 (2시간 간격).")

        scheduler.add_job(
            fetch_and_save_naver_news_job,
            trigger='interval', minutes=30, id='fetch_naver_news_job',
            max_instances=1, replace_existing=True, misfire_grace_time=60*5
        )
        logger.info("스케줄러: 'fetch_naver_news_job' 작업 추가 (30분 간격).")

    except Exception as e:
        logger.error(f"스케줄러: 작업 추가 중 오류: {e}")

    register_events(scheduler)

    if not scheduler.running:
        try:
            logger.info("스케줄러: 스케줄러를 시작합니다...")
            scheduler.start()
        except KeyboardInterrupt: logger.info("스케줄러: 중지됨."); scheduler.shutdown()
        except Exception as e: logger.error(f"스케줄러: 시작 중 오류: {e}")
    else:
        logger.info("스케줄러: 이미 실행 중입니다.")
