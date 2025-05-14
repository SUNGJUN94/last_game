# community/apps.py

from django.apps import AppConfig
import os
from django.conf import settings
import logging
from django.core.management import call_command # call_command 임포트

logger = logging.getLogger(__name__)

class CommunityConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'community'

    def ready(self):
        run_main_direct = os.environ.get('RUN_MAIN', None) == 'true'
        
        if settings.DEBUG and run_main_direct:
            from . import jobs  # jobs.py 임포트

            # 1. 서버 시작 시 1회 실행: DART API -> DartDisclosure 테이블 업데이트
            logger.info("커뮤니티 앱 준비: 서버 시작 시 DART 공시 데이터 가져오기 (fetch_disclosure)...")
            try:
                call_command('fetch_disclosure', '--days', '1')
            except Exception as e:
                logger.error(f"커뮤니티 앱 준비: fetch_disclosure 명령어 실행 중 오류: {e}")

            # 2. 서버 시작 시 1회 실행: DartDisclosure -> FreeBoard 동기화
            logger.info("커뮤니티 앱 준비: 서버 시작 시 DartDisclosure -> FreeBoard 동기화 시도...")
            try:
                jobs.auto_post_disclosures_to_freeboard(initial_run=True) # 초기 실행 플래그 전달
            except Exception as e:
                logger.error(f"커뮤니티 앱 준비: FreeBoard 자동 포스팅 중 오류 발생: {e}")

            # 3. 서버 시작 시 1회 실행: 네이버 뉴스 수집
            logger.info("커뮤니티 앱 준비: 서버 시작 시 네이버 뉴스 수집 시도...")
            try:
                jobs.fetch_and_save_naver_news_job(initial_run=True) # 초기 실행 플래그 전달
            except Exception as e:
                logger.error(f"커뮤니티 앱 준비: 네이버 뉴스 수집 중 오류 발생: {e}")

            # 4. 스케줄러 시작 (주기적인 업데이트용)
            if not jobs.scheduler.running:
                jobs.start_scheduler() # jobs.py의 start_scheduler는 내부적으로 두 작업을 등록함
                logger.info("커뮤니티 앱 준비: DART 및 네이버 뉴스 스케줄러 시작됨.")
            else:
                logger.info("커뮤니티 앱 준비: 스케줄러가 이미 실행 중입니다.")
        # else:
            # current_scheduler_running = 'N/A'
            # try:
            #     from . import jobs
            #     current_scheduler_running = jobs.scheduler.running
            # except ImportError: 
            #     pass 
            # logger.debug(f"커뮤니티 앱 준비 - 스케줄러 시작 조건 미충족 (DEBUG: {settings.DEBUG}, RUN_MAIN: {run_main_direct}, SCHEDULER_RUNNING: {current_scheduler_running})")

