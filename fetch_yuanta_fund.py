#!/usr/bin/env python3
"""
元大全球龍頭台幣A 歷史淨值抓取工具
輸出：yuanta_nav.csv（可直接餵進 backtest_html.py）

執行方式：
  python3 -m pip install requests beautifulsoup4 pandas
  python3 fetch_yuanta_fund.py
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time, os

FUND_CODE  = "ACYT168-YT81"
URL        = "https://yuantabank.moneydj.com/w/wr/wr02.djhtm"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "yuanta_nav.csv")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Referer": f"https://yuantabank.moneydj.com/w/wr/wr02.djhtm?a={FUND_CODE}",
}

def fetch_range(start: str, end: str) -> list[dict]:
    """
    抓取指定日期區間的淨值（start/end 格式：YYYYMMDD）
    回傳 [{"date": "2024-01-01", "nav": 20.5}, ...]
    """
    # 先嘗試 GET（帶參數）
    params = {"a": FUND_CODE, "b": "1", "c": start, "d": end}
    try:
        resp = requests.get(URL, params=params, headers=HEADERS, timeout=15)
        resp.encoding = "big5"
        html = resp.text
    except Exception as e:
        print(f"  GET 失敗 ({e})，嘗試 POST...")
        data = {"a": FUND_CODE, "b": "1", "c": start, "d": end}
        resp = requests.post(URL, data=data, headers=HEADERS, timeout=15)
        resp.encoding = "big5"
        html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) >= 2:
                date_str = tds[0].get_text(strip=True)
                nav_str  = tds[1].get_text(strip=True)
                # 日期格式：2024/01/02
                if "/" in date_str and len(date_str) == 10:
                    try:
                        date = datetime.strptime(date_str, "%Y/%m/%d").strftime("%Y-%m-%d")
                        nav  = float(nav_str.replace(",", ""))
                        rows.append({"date": date, "nav": nav})
                    except ValueError:
                        continue
    return rows


def fetch_all(years: int = 5) -> pd.DataFrame:
    """分季抓取，避免單次請求過大"""
    end   = datetime.today()
    start = end - timedelta(days=years * 365)

    all_rows = []
    cur = start
    while cur < end:
        seg_end = min(cur + timedelta(days=90), end)  # 每次抓 3 個月
        s = cur.strftime("%Y%m%d")
        e = seg_end.strftime("%Y%m%d")
        print(f"  抓取 {cur.strftime('%Y-%m')} ~ {seg_end.strftime('%Y-%m')}...")
        rows = fetch_range(s, e)
        print(f"    → {len(rows)} 筆")
        all_rows.extend(rows)
        cur = seg_end + timedelta(days=1)
        time.sleep(0.5)  # 避免請求過快

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("開始抓取元大全球龍頭台幣A 歷史淨值...")
    df = fetch_all(years=5)

    if df.empty:
        print("\n❌ 未抓到任何資料，請確認網路連線或 URL 是否有效。")
    else:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ 完成！共 {len(df)} 筆，已存至：{OUTPUT_CSV}")
        print(f"   期間：{df['date'].min().date()} ~ {df['date'].max().date()}")
        print(f"   起始淨值：{df['nav'].iloc[0]:.4f}")
        print(f"   最新淨值：{df['nav'].iloc[-1]:.4f}")
        print(f"   最大淨值：{df['nav'].max():.4f}（{df.loc[df['nav'].idxmax(),'date'].date()}）")
        print(f"   最小淨值：{df['nav'].min():.4f}（{df.loc[df['nav'].idxmin(),'date'].date()}）")
