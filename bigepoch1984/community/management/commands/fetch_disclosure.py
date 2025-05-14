import requests
from datetime import datetime, timedelta
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone as django_timezone
import zipfile
import io
from bs4 import BeautifulSoup
import re
import logging
import time

from community.models import DartDisclosure
# from community.models import FreeBoard # FreeBoard에 직접 저장하지 않음

logger = logging.getLogger(__name__)

LIST_URL = "https://opendart.fss.or.kr/api/list.json"
DOCUMENT_URL = "https://opendart.fss.or.kr/api/document.xml"

class Command(BaseCommand):
    help = "최근 지정된 일수 DART 공시 데이터를 가져와 DartDisclosure에 저장하고, 해당 기간 이전 데이터는 삭제합니다."

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=1, # 기본값을 1일로 변경
            help='가져올 최근 공시 기간(일 수)'
        )
        # TEST_MODE_LIMIT: 테스트용으로 시작 시 가져올 공시 개수를 제한하는 옵션 추가 (선택 사항)
        # 이 옵션을 사용하지 않고 settings.DEBUG를 직접 확인할 수도 있습니다.
        # parser.add_argument(
        #     '--limit_on_startup',
        #     type=int,
        #     default=None, # 기본값은 제한 없음
        #     help='(테스트용) 시작 시 가져올 최대 새 공시 개수'
        # )


    def _fetch_document_content_util(self, rcept_no, api_key):
        if not api_key:
            logger.error("API 키가 없어 본문 내용을 가져올 수 없습니다.")
            return None
        params = {'crtfc_key': api_key, 'rcept_no': rcept_no}
        try:
            response = requests.get(DOCUMENT_URL, params=params, timeout=60)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                xml_filename = None
                for name in zf.namelist():
                    if name.lower().endswith(('.xml', '.xhtml', '.html')):
                        xml_filename = name
                        break
                if xml_filename:
                    with zf.open(xml_filename) as xml_file:
                        content_bytes = xml_file.read()
                        try:
                            try: content_str = content_bytes.decode('utf-8')
                            except UnicodeDecodeError: content_str = content_bytes.decode('euc-kr', errors='replace')
                            soup = BeautifulSoup(content_str, 'lxml')
                            for element in soup(["script", "style", "comment", "header", "footer", "nav", "aside"]):
                                element.decompose()
                            text_content = soup.get_text(separator='\n', strip=True)
                            text_content = re.sub(r'\n\s*\n+', '\n', text_content)
                            text_content = re.sub(r'[ \t]+', ' ', text_content)
                            return text_content.strip()
                        except Exception as e:
                            logger.error(f"본문 내용 디코딩/파싱 오류 (rcept_no: {rcept_no}): {e}")
                            return None
                else:
                    logger.warning(f"ZIP 파일 내 XML/XHTML 파일 없음 (rcept_no: {rcept_no})")
                    return None
        except requests.exceptions.Timeout:
             logger.error(f"DART 공시 문서 API 요청 시간 초과 (rcept_no: {rcept_no})")
        except requests.exceptions.RequestException as e:
            logger.error(f"DART 공시 문서 API 요청 실패 (rcept_no: {rcept_no}): {e}")
        except zipfile.BadZipFile:
            logger.error(f"잘못된 ZIP 파일 형식 (rcept_no: {rcept_no})")
        except Exception as e:
             logger.error(f"본문 처리 중 예상치 못한 오류 발생 (rcept_no: {rcept_no}): {e}")
        return None

    def _delete_old_disclosures(self, days_to_keep=1): # 기본값 1일로 변경
        cutoff_date = django_timezone.now().date() - timedelta(days=days_to_keep)
        old_disclosures = DartDisclosure.objects.filter(rcept_dt__lt=cutoff_date)
        count_deleted, _ = old_disclosures.delete()
        if count_deleted > 0:
            self.stdout.write(self.style.SUCCESS(f"총 {count_deleted}개의 오래된 공시(접수일 < {cutoff_date})를 삭제했습니다."))
        else:
            self.stdout.write(f"삭제할 오래된 공시(접수일 < {cutoff_date})가 없습니다.")
        return count_deleted

    def handle(self, *args, **options):
        days_to_fetch = options['days']
        # startup_limit = options.get('limit_on_startup') # 옵션으로 처리할 경우
        self.stdout.write(f"[{django_timezone.now().strftime('%Y-%m-%d %H:%M:%S')}] DART 공시 업데이트 시작 (최근 {days_to_fetch}일)...")

        self._delete_old_disclosures(days_to_keep=days_to_fetch)

        api_key = getattr(settings, 'DART_API_KEY', None)
        if not api_key:
            self.stderr.write(self.style.ERROR("settings.py에 DART_API_KEY가 설정되지 않았습니다. 작업을 중단합니다."))
            return

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_to_fetch)
        params = {
            'crtfc_key': api_key,
            'bgn_de': start_date.strftime("%Y%m%d"),
            'end_de': end_date.strftime("%Y%m%d"),
            'page_count': 100, 'page_no': 1,
        }
        self.stdout.write(f"{start_date.strftime('%Y-%m-%d')}부터 {end_date.strftime('%Y-%m-%d')}까지의 공시를 가져옵니다...")

        saved_count_this_run = 0 # 이번 실행에서 저장된 공시 수
        total_fetched_pages = 0
        processed_rcept_nos = set(DartDisclosure.objects.values_list('rcept_no', flat=True))
        self.stdout.write(f"DB에 이미 저장된 공시 수: {len(processed_rcept_nos)}건")

        # TEST_MODE_LIMIT: DEBUG 모드에서 시작 시 가져올 새 공시 개수 제한
        # 원래 로직으로 복원하려면 MAX_NEW_DISCLOSURES_ON_STARTUP_DEBUG 변수와 관련된 if 조건들을 제거하세요.
        MAX_NEW_DISCLOSURES_ON_STARTUP_DEBUG = 10
        is_startup_test_mode = settings.DEBUG # DEBUG 모드일 때만 제한 적용 (apps.py에서 호출 시)

        if is_startup_test_mode:
            self.stdout.write(self.style.WARNING(f"TEST_MODE_LIMIT: DEBUG 모드이므로 시작 시 최대 {MAX_NEW_DISCLOSURES_ON_STARTUP_DEBUG}개의 새 공시만 가져옵니다."))


        try:
            while True:
                # TEST_MODE_LIMIT: DEBUG 모드이고, 이미 충분한 새 공시를 저장했다면 페이지 루프 중단
                if is_startup_test_mode and saved_count_this_run >= MAX_NEW_DISCLOSURES_ON_STARTUP_DEBUG:
                    self.stdout.write(self.style.SUCCESS(f"TEST_MODE_LIMIT: 시작 시 목표했던 {MAX_NEW_DISCLOSURES_ON_STARTUP_DEBUG}개의 새 공시를 저장하여 추가 페이지 처리를 중단합니다."))
                    break

                if total_fetched_pages > 0:
                    self.stdout.write(f"{params['page_no']-1} 페이지 처리 완료, 다음 페이지 요청 전 0.5초 대기...")
                    time.sleep(0.5)
                try:
                    self.stdout.write(f"{params['page_no']} 페이지 공시 목록 요청 중...")
                    response = requests.get(LIST_URL, params=params, timeout=20)
                    response.raise_for_status()
                    data = response.json()
                except requests.exceptions.Timeout:
                     self.stderr.write(self.style.ERROR(f"DART 공시 목록 API 요청 시간 초과 (page {params['page_no']})"))
                     break
                except requests.exceptions.RequestException as e:
                    self.stderr.write(self.style.ERROR(f"DART 공시 목록 API 요청 실패 (page {params['page_no']}): {e}"))
                    break
                except ValueError:
                    self.stderr.write(self.style.ERROR(f"DART 공시 목록 API 응답 JSON 디코딩 실패 (page {params['page_no']})"))
                    break

                if data.get('status') != '000':
                    if data.get('status') == '013':
                         self.stdout.write(f"페이지 {params['page_no']}에 더 이상 공시가 없습니다 (status: 013).")
                         break
                    self.stderr.write(self.style.ERROR(f"DART API 오류 (page {params['page_no']}): {data.get('status')} - {data.get('message')}"))
                    break

                disclosures_list = data.get('list', [])
                if not disclosures_list:
                    self.stdout.write(f"페이지 {params['page_no']}에 더 이상 공시가 없습니다 (빈 리스트).")
                    break

                total_fetched_pages +=1
                newly_saved_in_page = 0

                for item_idx, item in enumerate(disclosures_list):
                    rcept_no = item.get('rcept_no')
                    if not rcept_no:
                        logger.warning("접수번호(rcept_no)가 없는 항목이 있어 건너<0xEB><0x8D>니다.")
                        continue
                    if rcept_no in processed_rcept_nos:
                        continue

                    # TEST_MODE_LIMIT: DEBUG 모드이고, 이미 충분한 새 공시를 저장했다면 현재 페이지 아이템 루프 중단
                    if is_startup_test_mode and saved_count_this_run >= MAX_NEW_DISCLOSURES_ON_STARTUP_DEBUG:
                        self.stdout.write(f"    -> TEST_MODE_LIMIT: 시작 시 목표했던 {MAX_NEW_DISCLOSURES_ON_STARTUP_DEBUG}개의 새 공시를 저장하여 현재 페이지의 나머지 항목 처리를 중단합니다.")
                        break 

                    self.stdout.write(f"  {params['page_no']}-{item_idx+1}: 새 공시 발견 ({rcept_no}), 본문 가져오기 시도...")
                    try:
                        rcept_dt_obj = datetime.strptime(item.get("rcept_dt"), "%Y%m%d").date()
                        time.sleep(0.2)
                        document_content = self._fetch_document_content_util(rcept_no, api_key)
                        if document_content is None: document_content = ""

                        DartDisclosure.objects.create(
                            corp_code=item.get("corp_code"), corp_name=item.get("corp_name"),
                            stock_code=item.get("stock_code") if item.get("stock_code") else None,
                            corp_cls=item.get("corp_cls"), report_nm=item.get("report_nm"),
                            rcept_no=rcept_no, flr_nm=item.get("flr_nm"), rcept_dt=rcept_dt_obj,
                            rm=item.get("rm") if item.get("rm") else None,
                            report_link=f"http://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}",
                            doc_url=f"http://dart.fss.or.kr/report/viewer.do?rcpNo={rcept_no}&dcmNo=TEMP&eleId=TEMP&offset=TEMP&length=TEMP&dtd=TEMP",
                            document_content=document_content
                        )
                        saved_count_this_run += 1
                        newly_saved_in_page += 1
                        processed_rcept_nos.add(rcept_no)
                        self.stdout.write(f"    -> 저장 완료: {rcept_no} - {item.get('report_nm')}")
                    except Exception as e:
                        logger.error(f"공시 정보 저장 중 오류 발생 (rcept_no: {rcept_no}, report_nm: {item.get('report_nm')}): {e}")
                        continue

                self.stdout.write(f"{len(disclosures_list)}건 공시 처리 완료 (페이지 {params['page_no']}). 이번 페이지 신규 저장: {newly_saved_in_page}건, 누적 신규 저장: {saved_count_this_run}건")

                # TEST_MODE_LIMIT: (중복 확인) DEBUG 모드이고, 이미 충분한 새 공시를 저장했다면 페이지 루프 중단
                if is_startup_test_mode and saved_count_this_run >= MAX_NEW_DISCLOSURES_ON_STARTUP_DEBUG:
                    # 위에서 이미 break 했겠지만, 안전장치로 한 번 더 확인
                    break


                if params['page_no'] >= data.get('total_page', 1):
                    self.stdout.write("마지막 페이지까지 처리했습니다.")
                    break
                params['page_no'] += 1
        except KeyboardInterrupt:
             self.stdout.write(self.style.WARNING("\n사용자에 의해 작업이 중단되었습니다."))
        finally:
            self.stdout.write(self.style.SUCCESS(f"총 {total_fetched_pages} 페이지 처리, {saved_count_this_run}건의 새로운 공시를 저장했습니다."))
            self.stdout.write(f"[{django_timezone.now().strftime('%Y-%m-%d %H:%M:%S')}] DART 공시 업데이트 완료.")

