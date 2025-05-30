# predict_Info/management/commands/run_daily_updates.py
from django.core.management.base import BaseCommand
# startup_tasks.py의 위치에 따라 임포트 경로 수정 필요
# predict_Info 앱 내에 startup_tasks.py가 있다면:
from ...startup_tasks import run_daily_startup_tasks_main
# 또는 from predict_Info.startup_tasks import run_daily_startup_tasks_main (프로젝트 구조에 따라 다를 수 있음)

class Command(BaseCommand):
    help = 'Updates daily stock data CSVs and can optionally retrain LSTM models.'

    def add_arguments(self, parser):
        """
        Management command에 인자를 추가합니다.
        --retrain-models: 이 옵션이 주어지면 모델 재학습을 활성화합니다.
        """
        parser.add_argument(
            '--retrain-models',
            action='store_true',  # 이 옵션이 커맨드 라인에 있으면 True로 설정됨
            default=False,        # 기본값은 False (재학습 안 함)
            help='Enable model retraining after updating CSV data.',
        )

    def handle(self, *args, **options):
        # add_arguments에서 정의한 옵션 가져오기
        retrain_enabled = options['retrain_models']
        
        self.stdout.write(self.style.SUCCESS(f'Starting daily data updates (Model retraining: {"Enabled" if retrain_enabled else "Disabled"})...'))
        try:
            # run_daily_startup_tasks_main 함수에 재학습 여부 전달
            run_daily_startup_tasks_main(enable_model_retraining=retrain_enabled)
            self.stdout.write(self.style.SUCCESS('Successfully finished daily updates.'))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f'An error occurred: {e}'))
            import traceback
            traceback.print_exc()

