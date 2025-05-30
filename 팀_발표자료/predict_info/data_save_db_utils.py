# data_save_db_utils.py
import pymysql
from datetime import datetime, timedelta
import os
import pandas as pd

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'YOUR_EC2_PUBLIC_IP'),
    'user': os.getenv('DB_USER', 'your_user'),
    'password': os.getenv('DB_PASSWORD', 'your_password'),
    'database': os.getenv('DB_NAME', 'your_db'),
    'port': 3306,
    'charset': 'utf8mb4'
}

SAVE_DIR = "data_store/backup_exports"
os.makedirs(SAVE_DIR, exist_ok=True)
total_bytes = 0
MAX_BYTES_PER_DAY = 300 * 1024 * 1024  # 300MB
current_day = datetime.today().date()

def count_bytes(*args):
    return sum(len(str(arg).encode('utf-8')) for arg in args)

def save_record_to_db(record: dict, table: str, unique_key: str) -> bool:
    global total_bytes, current_day
    today = datetime.today().date()

    if today != current_day:
        current_day = today
        total_bytes = 0
        export_table_to_csv(table, 'date' if table == 'news_articles' else 'rcept_dt', current_day - timedelta(days=1))

    if table == "news_articles":
        size = count_bytes(record['종목명'], record['날짜'], record['제목'], record['요약'], record['본문'], record['URL'], record['언론사'])
    elif table == "dart_disclosures":
        size = count_bytes(record['corp_name'], record['corp_code'], record['rcept_no'], record['report_nm'], record['rcept_dt'], record['market'], record['url'], record['content_html'], record['content_text'], record['summary'])
    else:
        return False

    if total_bytes + size > MAX_BYTES_PER_DAY:
        print("\n 300MB 트래픽 한도 도달 - 저장 중단")
        return False

    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        sql_check = f"SELECT id FROM {table} WHERE {unique_key} = %s LIMIT 1"
        cursor.execute(sql_check, (record[unique_key],))
        if cursor.fetchone():
            return True

        if table == "news_articles":
            sql_insert = f"""
                INSERT INTO {table} (stock_name, date, title, summary, content, url, press)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                record['종목명'], record['날짜'], record['제목'],
                record['요약'], record['본문'], record['URL'], record['언론사']
            )
        elif table == "dart_disclosures":
            sql_insert = f"""
                INSERT INTO {table} (corp_name, corp_code, rcept_no, report_nm, rcept_dt, market, url, content_html, content_text, summary)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                record['corp_name'], record['corp_code'], record['rcept_no'], record['report_nm'],
                record['rcept_dt'], record['market'], record['url'], record['content_html'],
                record['content_text'], record['summary']
            )

        cursor.execute(sql_insert, values)
        conn.commit()
        total_bytes += size
        return True
    except Exception as e:
        print(f"[DB 오류] {record.get(unique_key, 'UNKNOWN')} ▶ {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def export_table_to_csv(table: str, date_field: str, target_date: datetime.date):
    try:
        conn = pymysql.connect(**DB_CONFIG)
        sql = f"""
            SELECT * FROM {table}
            WHERE {date_field} = %s
        """
        df = pd.read_sql(sql, conn, params=[target_date])
        export_path = os.path.join(SAVE_DIR, f"backup_db_{table}_{target_date}.csv")
        df.to_csv(export_path, index=False, encoding="utf-8-sig")
        print(f"\n📦 {table} 전일 데이터 CSV 저장 완료: {export_path}")
    except Exception as e:
        print(f"[CSV 저장 실패] ▶ {e}")
    finally:
        conn.close()
