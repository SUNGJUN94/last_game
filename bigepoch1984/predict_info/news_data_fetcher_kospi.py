from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import time, os, re, sys
from datetime import datetime, timedelta
from tqdm import tqdm
import FinanceDataReader as fdr
from time import perf_counter

# 설정
CHROMEDRIVER_PATH = "/usr/bin/chromedriver"
SAVE_DIR = "news_data/chosun"
os.makedirs(SAVE_DIR, exist_ok=True)
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime.today() - timedelta(days=1)

# 완료된 종목 체크
def get_completed_stocks(file_path):
    if not os.path.exists(file_path):
        return set()
    df = pd.read_csv(file_path)
    return set(df['종목명'].unique())

# 날짜 추출
def extract_date(text):
    match = re.search(r"\d{4}\.\d{2}\.\d{2}", text)
    if match:
        try:
            return datetime.strptime(match.group(0), "%Y.%m.%d")
        except:
            return None
    return None

#크롬 드라이버 설정
def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("user-agent=Mozilla/5.0")
    options.add_argument("--disable-gpu")
    options.add_argument("--blink-settings=imagesEnabled=false")
    
    return webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=options)
    
# 서브도메인 구분 및 파싱
def parse_chosun(driver):
    soup = BeautifulSoup(driver.page_source, "html.parser")
    paragraphs = soup.select("p.article-body__content-text")
    return " ".join(p.get_text(strip=True) for p in paragraphs)

def parse_biz(driver):
    soup = BeautifulSoup(driver.page_source, "html.parser")
    paragraphs = soup.select("p.article-body__content.article-body__content-text")
    return " ".join(p.get_text(strip=True) for p in paragraphs)

def parse_it(driver):
    soup = BeautifulSoup(driver.page_source, "html.parser")
    body = soup.select_one("div.article-body")
    return " ".join(p.get_text(strip=True) for p in body.find_all("p")) if body else ""

def parse_tv(driver):
    soup = BeautifulSoup(driver.page_source, "html.parser")
    box = soup.select_one("div.text-box")
    return " ".join(p.get_text(strip=True) for p in box.find_all("p")) if box else ""

def get_parser_by_url(url):
    if "biz.chosun.com" in url:
        return parse_biz, "비즈조선"
    elif "it.chosun.com" in url:
        return parse_it, "IT조선"
    elif "tvchosun.com" in url:
        return parse_tv, "TV조선"
    else:
        return parse_chosun, "조선일보"

# 기사 크롤링
def crawl_chosun_news(keyword):
    driver = get_driver()
    page, print_count, empty_count = 1, 0, 0
    results, seen_links = [], set()

    while True:
        query_url = f"https://www.chosun.com/nsearch/?query={keyword}&pageno={page}"
        driver.get(query_url)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        cards = soup.select("div.story-card")
        if not cards:
            break

        total_cards = len(cards)
        added, excluded = 0, 0
        valid = False

        for card in cards:
            title_tag = card.select_one("a.text__link.story-card__headline")
            summary_tag = card.select_one("span.story-card__deck") or card.select_one("span[style]")
            breadcrumb_tag = card.select_one("div.story-card__breadcrumb")

            if not title_tag:
                excluded += 1
                continue

            title = title_tag.get_text(strip=True)
            link = title_tag["href"]
            if link.startswith("/"):
                link = "https://www.chosun.com" + link
            if link in seen_links or not any(d in link for d in ["chosun.com"]):
                excluded += 1
                continue
            seen_links.add(link)

            summary = summary_tag.get_text(strip=True) if summary_tag else ""
            breadcrumb_text = breadcrumb_tag.get_text(" ", strip=True) if breadcrumb_tag else ""
            date = extract_date(breadcrumb_text)
            if not date or date < START_DATE or date > END_DATE:
                continue
            valid = True

            try:
                driver.get(link)
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                parser, press = get_parser_by_url(link)
                content = parser(driver)
            except:
                continue

            results.append({
                "종목명": keyword,
                "날짜": date.strftime("%Y-%m-%d"),
                "제목": title,
                "요약": summary,
                "본문": content,
                "URL": link,
                "언론사": press
            })
            added += 1

        tqdm.write(
            f"[🔍 {keyword}] Page {page} | 기사 수: {total_cards}, 제외: {excluded}, 수집: {added}, 누적: {print_count}"
        )
        print_count += added

        if not valid:
            empty_count_in_page += 1
        else:
            empty_count_in_page = 0

        if empty_count_in_page >= 200:
            tqdm.write(f"[중단] {keyword} - 최근 기사 없음 (200페이지 연속)")
            break

        page += 1

    driver.quit()
    return results, print_count

# 실행부
if __name__ == "__main__":
    start_time = perf_counter()

    print("FDR에서 코스피/코스닥 종목 불러오는 중...")
    kospi_list = fdr.StockListing('KOSPI')['Name'].dropna().unique().tolist()

    completed_kospi = get_completed_stocks(os.path.join(SAVE_DIR, "chosun_kospi_articles.csv"))

    kospi_list = [s for s in kospi_list if s not in completed_kospi]
    print(f"→ 이어받기 적용: KOSPI {len(kospi_list)}개")

    combined_list = [("KOSPI", stock) for stock in kospi_list]
    
    all_kospi = []
    total_kospi = 0

    def threaded_crawl(args):
        market, stock = args
        try:
            articles, count = crawl_chosun_news(stock)
            return (market, stock, articles, count)
        except Exception as e:
            tqdm.write(f"[⚠] 실패: {stock} ({e})")
            return (market, stock, [], 0)

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(threaded_crawl, arg) for arg in combined_list]
            for future in tqdm(as_completed(futures), total=len(futures), desc="전체 진행률", ncols=100, dynamic_ncols=True):
                market, stock, articles, count = future.result()
                if market == "KOSPI":
                    all_kospi.extend(articles)
                    total_kospi += count
                else:
                    continue
                tqdm.write(f"[✔] {stock} 수집 완료 ▶ 종목 누적: {count}, KOSPI 총: {total_kospi}")

    except KeyboardInterrupt:
        print("\n⛔ [중단] 사용자 수동 종료")

    finally:
        cols = ["종목명", "날짜", "제목", "요약", "본문", "URL", "언론사"]
        if all_kospi:
            pd.DataFrame(all_kospi)[cols].to_csv(os.path.join(SAVE_DIR, "chosun_kospi_articles.csv"), index=False, encoding="utf-8-sig")
            print(f"KOSPI 저장 완료: {len(all_kospi)}건")
        elapsed = perf_counter() - start_time
        print(f"\n✅ 전체 기사 크롤링 완료 | 소요 시간: {elapsed:.2f}초 ≈ {elapsed/60:.1f}분")