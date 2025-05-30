# predict_Info/apps.py
from django.apps import AppConfig
import os # 환경 변수 사용을 위해 추가

class PredictInfoConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "predict_info"

    def ready(self):
        # runserver로 실행될 때만 이 로직을 실행 (manage.py makemigrations 등에서는 실행 방지)
        # 또는 특정 환경 변수를 설정하여 제어할 수 있음
        # 예: if os.environ.get('RUN_MAIN') == 'true' and not os.environ.get('SKIP_STARTUP_TASKS'):
        
        # DEBUG 모드이거나, 특정 환경 변수가 설정된 경우에만 실행하도록 조건 추가 가능
        # from django.conf import settings
        # if settings.DEBUG: # 또는 if os.environ.get('DJANGO_RUNSERVER_STARTUP_TASKS') == 'true':

        # manage.py runserver시에만 실행되도록 하는 일반적인 방법은
        # sys.argv를 확인하는 것.
        import sys
        is_runserver = any(cmd in sys.argv for cmd in ['runserver'])
        # 또는, Django 초기화 과정에서 여러 번 호출될 수 있으므로,
        # 한 번만 실행되도록 플래그 관리 필요. (더 복잡)

        # 가장 간단한 방법은 환경 변수로 제어하거나, DEBUG 모드에서만 실행.
        # 여기서는 서버 시작 시 항상 시도하도록 단순화.
        # 실제 프로덕션에서는 이런 방식은 부적절할 수 있음.
        
        # RUN_MAIN 환경 변수는 Django가 runserver로 메인 프로세스를 실행할 때 설정됨
        if os.environ.get('RUN_MAIN') == 'true':
            print("Django 서버 메인 프로세스 시작 감지 (RUN_MAIN)")
            # 별도의 스레드나 비동기 작업으로 실행하는 것이 좋음 (서버 시작 지연 방지)
            # 여기서는 간단히 직접 호출. 실제 운영 시에는 Background Task 고려.
            print("서버 시작 시 일일 데이터 업데이트 및 모델 학습 작업 실행 시도...")
            try:
                from . import startup_tasks
                # startup_tasks.run_daily_startup_tasks() # 서버 시작 속도를 위해 초기에는 주석 처리하고 필요시 활성화
                print("startup_tasks.run_daily_startup_tasks() 호출은 초기 서버 시작 속도 저하 가능성으로 주석 처리됨.")
                print("필요시 predict_Info/apps.py에서 해당 라인의 주석을 해제하거나, management command로 별도 실행 권장.")
                
                # 대신, 모델 로드만 수행 (views.py에서 이미 수행 중일 수 있음)
                # from .views import load_all_models_and_scalers
                # load_all_models_and_scalers() # views.py에서 이미 로드하므로 중복 호출 불필요
                
            except Exception as e:
                print(f"Startup 작업 중 오류 발생: {e}")
        else:
            # RUN_MAIN이 'true'가 아닌 경우 (e.g., manage.py makemigrations, manage.py shell 등)
            # 또는 Django가 자동 리로드하는 자식 프로세스인 경우
            pass