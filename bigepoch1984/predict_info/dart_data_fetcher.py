import os
import time
import sys
import requests
import pandas as pd
import xml.etree.ElementTree as ET
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from tqdm import tqdm
from tqdm import tqdm
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

# 설정
API_KEY = "629d8774e6b825dba0c3ee060bd7e19646d286de"
START_DATE = "20200101"
END_DATE = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")
TODAY = datetime.today().strftime("%Y%m%d")
SAVE_DIR = f"./data/dart_data/{TODAY}"
os.makedirs(SAVE_DIR, exist_ok=True)
MAX_WORKERS = 4


def get_completed_corp_codes(path):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(df['corp_code'].astype(str).unique())


def generate_corp_df():
    tree = ET.parse("./data/CORPCODE.xml")
    root = tree.getroot()
    dart_list = [{
        "corp_name": child.find("corp_name").text,
        "corp_code": child.find("corp_code").text,
        "stock_code": child.find("stock_code").text
    } for child in root.iter("list") if child.find("stock_code").text]
    dart_df = pd.DataFrame(dart_list)

    fdr_df = fdr.StockListing("KRX")[["Name", "Code", "Market"]]
    fdr_df.columns = ["corp_name", "stock_code", "market"]
    merged = pd.merge(fdr_df, dart_df, on=["corp_name", "stock_code"], how="inner")
    merged["시장구분"] = merged["market"]
    return merged


def fetch_dart_reports(corp_code, bgn_date, end_date):
    url = "https://opendart.fss.or.kr/api/list.json"
    params = {
        "crtfc_key": API_KEY,
        "corp_code": corp_code,
        "bgn_de": bgn_date,
        "end_de": end_date,
        "page_count": 100
    }
    try:
        res = requests.get(url, params=params)
        data = res.json()
        if data.get("status") != "013" and "list" in data:
            return data["list"]
    except Exception as e:
        tqdm.write(f"[예외] {corp_code}: {e}")
    return []


def fetch_pdf_text(rcp_no, dcm_no):
    try:
        pdf_url = f"https://dart.fss.or.kr/pdf/download/pdf.do?rcp_no={rcp_no}&dcm_no={dcm_no}"
        pdf_path = f"/tmp/{rcp_no}_{dcm_no}.pdf"
        res = requests.get(pdf_url, timeout=15)
        with open(pdf_path, "wb") as f:
            f.write(res.content)
        text = extract_text(pdf_path)
        os.remove(pdf_path)
        return text[:3000] if text else "[본문 없음] PDF 비어 있음"
    except PDFSyntaxError:
        return "[본문 오류] PDF 문법 오류"
    except Exception as e:
        return f"[본문 오류] {e}"


def fetch_dart_text(rcp_no):
    try:
        base_url = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcp_no}"
        res = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        iframe = soup.find("iframe")
        if not iframe:
            return "[본문 없음: iframe 없음]", "[본문 없음: iframe 없음]"

        iframe_url = "https://dart.fss.or.kr" + iframe["src"]
        dcm_no = iframe_url.split("dcmNo=")[-1]
        detail = requests.get(iframe_url, timeout=10)
        detail_soup = BeautifulSoup(detail.text, "html.parser")
        html_text = detail_soup.prettify()
        text_only = detail_soup.get_text(separator="\n", strip=True)

        if len(text_only.strip()) < 50:
            pdf_text = fetch_pdf_text(rcp_no, dcm_no)
            return html_text, pdf_text

        time.sleep(0.3)  # 딜레이 추가
        return html_text, text_only[:3000]
    except Exception as e:
        return "", f"[본문 오류] {e}"


def process_company(row):
    name, code, market = row['corp_name'], str(row['corp_code']), row['시장구분']
    reports = fetch_dart_reports(code, START_DATE, END_DATE)
    results = []
    for r in reports:
        rcp = r.get("rcept_no")
        html, text = fetch_dart_text(rcp)
        tqdm.write(f"[처리중] {name} - {rcp}")
        enriched = {
            "corp_name": name,
            "corp_code": code,
            "rcept_no": rcp,
            "report_nm": r.get("report_nm"),
            "rcept_dt": r.get("rcept_dt"),
            "market": market,
            "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcp}",
            "content_html": html,
            "content_text": text,
            "summary": ""
        }
        results.append(enriched)
    return results, market


def main():
    corp_df = generate_corp_df()
    kospi_done = get_completed_corp_codes(os.path.join(SAVE_DIR, "dart_kospi_with_text.csv"))
    kosdaq_done = get_completed_corp_codes(os.path.join(SAVE_DIR, "dart_kosdaq_with_text.csv"))
    kospi_reports, kosdaq_reports = [], []
    cols = ["rcept_dt", "market", "corp_name", "corp_code", "rcept_no", "report_nm", "url", "content_html", "content_text", "summary"]

    tqdm.write(f"📅 수집 범위: {START_DATE} ~ {END_DATE} | 총 기업 수: {len(corp_df)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for _, row in corp_df.iterrows():
            if (row['시장구분'] == "KOSPI" and str(row['corp_code']) in kospi_done) or \
               (row['시장구분'] == "KOSDAQ" and str(row['corp_code']) in kosdaq_done):
                continue
            futures.append(executor.submit(process_company, row))

        for future in tqdm(as_completed(futures), total=len(futures), ncols=100):
            try:
                results, market = future.result()
                if market == "KOSPI":
                    kospi_reports.extend(results)
                else:
                    kosdaq_reports.extend(results)
            except Exception as e:
                tqdm.write(f"[오류] 스레드 처리 중 문제 발생: {e}")

    if kospi_reports:
        pd.DataFrame(kospi_reports)[cols].to_csv(
            os.path.join(SAVE_DIR, "dart_kospi_data.csv"),
            index=False, encoding="utf-8-sig")
        tqdm.write(f"\n✅ KOSPI 저장 완료: {len(kospi_reports)}건")

    if kosdaq_reports:
        pd.DataFrame(kosdaq_reports)[cols].to_csv(
            os.path.join(SAVE_DIR, "dart_kosdaq_data.csv"),
            index=False, encoding="utf-8-sig")
        tqdm.write(f"\n✅ KOSDAQ 저장 완료: {len(kosdaq_reports)}건")

    tqdm.write("\n🏁 전체 공시 본문 수집 완료")

if __name__ == "__main__":
    main()
