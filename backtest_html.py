#!/usr/bin/env python3
"""
ETF 回測工具 — 單一 HTML，兩個分頁
  Tab 1 標準回測：固定起始點圖表 + 績效表，可切換「最新 / 截至 2024/12/31」
  Tab 2 滾動回測：多起始點 10 年窗口統計表（無圖）

執行方式：
  python3 -m pip install yfinance pandas numpy
  python3 backtest_html.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json, os, warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  ★ 參數設定
# ══════════════════════════════════════════════════════════════
YEARS         = 20
ROLLING_YEARS = 10
TWD_LUMP      = 1_000_000
TWD_MONTHLY   = 4_200
TWD_PER_USD   = 30.0
OUTPUT_DIR    = os.path.dirname(os.path.abspath(__file__))

TICKERS = {
    "0050.TW":   {"name": "元大台灣50",          "currency": "TWD"},
    "00631L.TW": {"name": "元大台灣50正向2倍",    "currency": "TWD"},
    "QQQ":       {"name": "Invesco QQQ",         "currency": "USD"},
    "QLD":       {"name": "ProShares Ultra QQQ", "currency": "USD"},
    "SPY":       {"name": "SPDR S&P 500",        "currency": "USD"},
}
TICKER_COLORS = {
    "0050.TW":   ("#3fb950", "#79c0ff"),
    "00631L.TW": ("#ff7b72", "#ffa657"),
    "QQQ":       ("#d2a8ff", "#a5d6ff"),
    "QLD":       ("#f0883e", "#bc8cff"),
    "SPY":       ("#e3b341", "#f0c97f"),
}
MANUAL_SPLITS = {
    "0050.TW":   [(4,  None)],
    "00631L.TW": [(22, "2015-01-01")],
}
LOCAL_TICKERS = {
    "元大龍頭台幣A": {"name": "元大龍頭台幣A",    "currency": "TWD", "file": "元大龍頭台幣"},
    "安聯台灣智慧":  {"name": "安聯台灣智慧基金",  "currency": "TWD", "file": "安聯台灣智慧基金"},
    "安聯台灣科技":  {"name": "安聯台灣科技基金",  "currency": "TWD", "file": "安聯台灣科技"},
}
TICKER_COLORS.update({
    "元大龍頭台幣A": ("#56d364", "#1f6feb"),
    "安聯台灣智慧":  ("#ffa657", "#ff7b72"),
    "安聯台灣科技":  ("#bc8cff", "#d2a8ff"),
})
BADGE_STYLES = {
    "0050.TW":   ("#1f3a5f", "#79c0ff"),
    "00631L.TW": ("#3d1f1f", "#ff7b72"),
    "QQQ":       ("#2d2438", "#d2a8ff"),
    "QLD":       ("#2a1e0f", "#f0883e"),
    "SPY":       ("#2e2a1a", "#e3b341"),
    "元大龍頭台幣A": ("#1e3a2e", "#56d364"),
    "安聯台灣智慧":  ("#3d2a0f", "#ffa657"),
    "安聯台灣科技":  ("#2a1e3d", "#bc8cff"),
}
DEFAULT_TICKER = "0050.TW"
CUTOFF_2024    = "2024-12-31"
# ══════════════════════════════════════════════════════════════


# ── 工具函式 ────────────────────────────────────────────────

def load_local_fund(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df = pd.read_csv(path, parse_dates=["日期"])
    df = df[["日期", "淨值"]].rename(columns={"日期": "Date", "淨值": "Close"})
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna().set_index("Date").sort_index()
    df.index = pd.DatetimeIndex(df.index)
    return df["Close"]


def download(ticker):
    df = yf.download(ticker, start="2000-01-01",
                     end=datetime.today().strftime("%Y-%m-%d"),
                     auto_adjust=True, progress=False)
    df = df[["Close"]].copy()
    df.columns = ["Close"]
    df.dropna(inplace=True)
    return df


def apply_split(prices, ticker):
    if ticker not in MANUAL_SPLITS:
        return prices
    adj = prices.copy()
    for ratio, sdate in MANUAL_SPLITS[ticker]:
        if sdate:
            sd = pd.Timestamp(sdate)
        else:
            mp = adj.resample("MS").first().dropna()
            pc = mp.pct_change().dropna()
            sd = (pc - (-(1 - 1 / ratio))).abs().idxmin()
            print(f"  [分割偵測] {ticker}：{sd.date()}  當月 {pc[sd]:.1%} → {ratio}:1")
        mask = adj.index < sd
        adj[mask] /= ratio
        print(f"  [分割調整] {ticker}：{sd.date()} 前 {mask.sum()} 筆 ÷ {ratio}")
    return adj


def backtest_window(monthly_px, lump, monthly):
    n = len(monthly_px)
    bah = (lump / monthly_px.iloc[0]) * monthly_px
    shares, dvals = 0.0, []
    for px in monthly_px:
        shares += monthly / px
        dvals.append(shares * px)
    dca  = pd.Series(dvals, index=monthly_px.index)
    cost = pd.Series([monthly * (i + 1) for i in range(n)], index=monthly_px.index)
    return bah, dca, cost


def cagr_f(final, invested, years):
    return (final / invested) ** (1 / years) - 1


def annual_table(bah, dca, cost, lump):
    """逐年績效：B&H YoY、累計 CAGR；DCA 總報酬率、累計 CAGR"""
    try:
        bah_y  = bah.resample("YE").last().dropna()
        dca_y  = dca.resample("YE").last().dropna()
        cost_y = cost.resample("YE").last().dropna()
    except ValueError:
        bah_y  = bah.resample("A").last().dropna()
        dca_y  = dca.resample("A").last().dropna()
        cost_y = cost.resample("A").last().dropna()

    rows = []
    prev_bah = lump
    for i, ts in enumerate(bah_y.index):
        bah_val  = float(bah_y[ts])
        dca_val  = float(dca_y.reindex([ts]).iloc[0]) if ts in dca_y.index else float("nan")
        invested = float(cost_y.reindex([ts]).iloc[0]) if ts in cost_y.index else float("nan")
        k = i + 1
        bah_yoy  = round((bah_val / prev_bah - 1) * 100, 2)
        bah_cagr = round((cagr_f(bah_val, lump, k)) * 100, 2)
        dca_tr   = round((dca_val / invested - 1) * 100, 2) if invested else None
        dca_cagr = round(cagr_f(dca_val, invested, k) * 100, 2)   if invested else None
        prev_bah = bah_val
        rows.append({
            "year":     ts.year,
            "bah_val":  round(bah_val),
            "bah_yoy":  bah_yoy,
            "bah_cagr": bah_cagr,
            "dca_val":  round(dca_val),
            "invested": round(invested),
            "dca_tr":   dca_tr,
            "dca_cagr": dca_cagr,
        })
    return rows


def max_dd(series):
    return ((series - series.cummax()) / series.cummax()).min()


def sharpe(series):
    r = series.pct_change().dropna()
    return (r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else 0


def calc_stats(series, total_invested, label):
    y  = (series.index[-1] - series.index[0]).days / 365.25
    f  = series.iloc[-1]
    tr = (f - total_invested) / total_invested
    return {
        "label":           label,
        "total_invested":  round(total_invested),
        "final_value":     round(f),
        "total_ret":       f"{tr:.1%}",
        "cagr":            f"{cagr_f(f, total_invested, y):.2%}",
        "max_dd":          f"{max_dd(series):.1%}",
        "sharpe":          f"{sharpe(series):.2f}",
        "years":           round(y, 1),
        "note":            "",
    }


def to_pts(s):
    return [{"x": d.strftime("%Y-%m"), "y": round(float(v), 0)} for d, v in s.items()]


def dd_pts(s):
    dd = (s - s.cummax()) / s.cummax() * 100
    return [{"x": d.strftime("%Y-%m"), "y": round(float(v), 2)} for d, v in dd.items()]


def run_standard(px_full, lump, monthly, cutoff=None):
    """標準回測：從最早有資料的月份開始（或截至 cutoff）"""
    px = px_full.copy()
    if cutoff:
        px = px[px.index <= cutoff]
    std_px = px.resample("MS").first().dropna()
    bah, dca, cost = backtest_window(std_px, lump, monthly)
    return bah, dca, cost, std_px


def run_rolling(mp, lump, monthly):
    W = ROLLING_YEARS * 12
    if len(mp) < W + 1:
        return None
    bah_cagrs, dca_cagrs, starts, ends = [], [], [], []
    for i in range(len(mp) - W):
        w  = mp.iloc[i:i + W]
        p0 = float(w.iloc[0])
        bah_final = (lump / p0) * float(w.iloc[-1])
        bc = cagr_f(bah_final, lump, ROLLING_YEARS)
        shares = sum(monthly / float(px) for px in w)
        dca_final = shares * float(w.iloc[-1])
        dc = cagr_f(dca_final, monthly * W, ROLLING_YEARS)
        bah_cagrs.append(round(bc * 100, 2))
        dca_cagrs.append(round(dc * 100, 2))
        starts.append(w.index[0].strftime("%Y-%m"))
        ends.append(w.index[-1].strftime("%Y-%m"))

    def rs(arr):
        a = np.array(arr)
        bi = int(np.argmax(a))
        wi = int(np.argmin(a))
        return {
            "n":           len(a),
            "mean":        round(float(np.mean(a)), 2),
            "median":      round(float(np.median(a)), 2),
            "best":        round(float(a[bi]), 2),
            "best_range":  f"{starts[bi]} ~ {ends[bi]}",
            "worst":       round(float(a[wi]), 2),
            "worst_range": f"{starts[wi]} ~ {ends[wi]}",
            "std":         round(float(np.std(a)), 2),
            "win_pct":     round(float(np.mean(a > 0) * 100), 1),
        }

    return {"bah": rs(bah_cagrs), "dca": rs(dca_cagrs)}


# ── 下載與計算 ───────────────────────────────────────────────

print("下載資料中...")
raw = {}
for ticker, cfg in TICKERS.items():
    print(f"  {cfg['name']} ({ticker})...")
    df = download(ticker)
    df["Close"] = apply_split(df["Close"], ticker)
    print(f"    ✓ {df.index[0].date()} ~ {df.index[-1].date()}")
    raw[ticker] = df["Close"]

print("讀取本地基金資料...")
for ticker, cfg in LOCAL_TICKERS.items():
    print(f"  {cfg['name']}...")
    px = load_local_fund(cfg["file"])
    print(f"    ✓ {px.index[0].date()} ~ {px.index[-1].date()}  ({len(px)} 筆)")
    raw[ticker] = px

ALL_TICKERS = {**TICKERS, **LOCAL_TICKERS}

# 標準回測：兩個版本
std_versions = {
    "latest": {"label": "最新資料",     "cutoff": None},
    "y2024":  {"label": "截至 2024/12/31", "cutoff": CUTOFF_2024},
}

chart_data = {}   # ticker -> version -> chart series
all_stats  = {}   # version -> list of stat dicts

for ver_key, ver_cfg in std_versions.items():
    print(f"\n標準回測（{ver_cfg['label']}）...")
    all_stats[ver_key] = []
    chart_data[ver_key] = {}

    for ticker, cfg in ALL_TICKERS.items():
        lump    = TWD_LUMP    if cfg["currency"] == "TWD" else TWD_LUMP    / TWD_PER_USD
        monthly = TWD_MONTHLY if cfg["currency"] == "TWD" else TWD_MONTHLY / TWD_PER_USD
        bah, dca, cost, std_px = run_standard(raw[ticker], lump, monthly, ver_cfg["cutoff"])

        chart_data[ver_key][ticker] = {
            "name":     cfg["name"],
            "currency": cfg["currency"],
            "lump":     round(lump),
            "monthly":  round(monthly),
            "colors":   list(TICKER_COLORS.get(ticker, ("#58a6ff", "#d2a8ff"))),
            "bah":      to_pts(bah),
            "dca":      to_pts(dca),
            "cost":     to_pts(cost),
            "bah_dd":   dd_pts(bah),
            "dca_dd":   dd_pts(dca),
            "annual":   annual_table(bah, dca, cost, lump),
            "price":    [{"x": d.strftime("%Y-%m"), "y": round(float(v), 2)}
                         for d, v in std_px.items()],
        }
        s_bah = calc_stats(bah, lump,             f"{cfg['name']} 買入持有")
        s_dca = calc_stats(dca, monthly * len(std_px), f"{cfg['name']} 定期定額")
        all_stats[ver_key].extend([s_bah, s_dca])
        print(f"  {cfg['name']}  B&H {s_bah['cagr']}  DCA {s_dca['cagr']}")

    pass  # (MANUAL_FUNDS removed — local funds now handled in ALL_TICKERS)

# 滾動回測（用完整資料）
print(f"\n滾動回測（{ROLLING_YEARS} 年窗口）...")
rolling_rows = []
for ticker, cfg in ALL_TICKERS.items():
    lump    = TWD_LUMP    if cfg["currency"] == "TWD" else TWD_LUMP    / TWD_PER_USD
    monthly = TWD_MONTHLY if cfg["currency"] == "TWD" else TWD_MONTHLY / TWD_PER_USD
    mp  = raw[ticker].resample("MS").first().dropna()
    res = run_rolling(mp, lump, monthly)
    if res is None:
        print(f"  {ticker}：資料不足，跳過")
        continue
    bg, fg = BADGE_STYLES.get(ticker, ("#21262d", "#cdd9e5"))
    rolling_rows.append({
        "name": cfg["name"], "badge_bg": bg, "badge_fg": fg, **res,
    })
    print(f"  {cfg['name']}  B&H 中位數 {res['bah']['median']:.2f}%  "
          f"DCA 中位數 {res['dca']['median']:.2f}%  窗口數 {res['bah']['n']}")


# ── 組 JSON ─────────────────────────────────────────────────

js_chart   = json.dumps(chart_data,   ensure_ascii=False)
js_stats   = json.dumps(all_stats,    ensure_ascii=False)
js_rolling = json.dumps(rolling_rows, ensure_ascii=False)

select_options = "\n".join(
    f'        <option value="{tk}"{" selected" if tk == DEFAULT_TICKER else ""}>'
    f'{cfg["name"]}</option>'
    for tk, cfg in ALL_TICKERS.items()
)

# ── HTML ────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ETF 回測｜B&H vs DCA</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:#0d1117; color:#e6edf3; font-family:-apple-system,"Segoe UI","Microsoft JhengHei",sans-serif; padding-bottom:60px; }}
h1 {{ text-align:center; padding:26px 16px 4px; font-size:1.4rem; color:#f0f6fc; }}
.sub {{ text-align:center; color:#8b949e; font-size:0.83rem; margin-bottom:16px; line-height:1.8; }}
.wrap {{ max-width:1400px; margin:0 auto; padding:0 18px; }}

/* 分頁 tabs */
.tabs {{ display:flex; gap:0; border-bottom:2px solid #30363d; margin-bottom:20px; }}
.tab-btn {{
  background:none; border:none; color:#8b949e; padding:11px 28px;
  font-size:0.93rem; cursor:pointer; border-bottom:2px solid transparent;
  margin-bottom:-2px; transition:color .15s,border-color .15s;
}}
.tab-btn:hover {{ color:#e6edf3; }}
.tab-btn.active {{ color:#58a6ff; border-bottom-color:#58a6ff; font-weight:600; }}
.tab-panel {{ display:none; }}
.tab-panel.active {{ display:block; }}

/* toolbar */
.toolbar {{ display:flex; align-items:center; gap:12px; margin-bottom:18px;
            padding:12px 18px; background:#161b22; border:1px solid #30363d; border-radius:10px; flex-wrap:wrap; }}
.toolbar label {{ color:#8b949e; font-size:0.88rem; white-space:nowrap; }}
select.sel {{
  background:#21262d; color:#e6edf3; border:1px solid #30363d;
  border-radius:8px; padding:6px 12px; font-size:0.88rem; cursor:pointer; outline:none;
}}
select.sel:hover {{ border-color:#58a6ff; }}

.card {{ background:#161b22; border:1px solid #30363d; border-radius:12px; padding:18px; margin-bottom:18px; }}
.card-title {{ font-size:0.93rem; color:#58a6ff; font-weight:600; margin-bottom:14px; }}
.card-note {{ color:#8b949e; font-size:0.78rem; margin-bottom:14px; line-height:1.7; }}

table {{ width:100%; border-collapse:collapse; font-size:0.83rem; }}
th {{ background:#21262d; color:#8b949e; padding:8px 11px; text-align:left;
     border-bottom:1px solid #30363d; white-space:nowrap; }}
td {{ padding:8px 11px; border-bottom:1px solid #21262d; white-space:nowrap; vertical-align:middle; }}
tr.sep > td {{ border-top:2px solid #30363d; }}
tr:hover td {{ background:#1c2128; }}
tr.row-highlight td {{ background:#1a2535 !important; }}
.pos {{ color:#3fb950; font-weight:600; }}
.neg {{ color:#f85149; }}
.muted {{ color:#8b949e; }}
.strong {{ font-weight:700; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }}
.strat {{ display:inline-block; padding:2px 7px; border-radius:4px; font-size:0.75rem; font-weight:600; }}
.strat.bah {{ background:#1e3a1e; color:#3fb950; }}
.strat.dca {{ background:#2d1f3d; color:#d2a8ff; }}

.row2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media(max-width:860px){{ .row2{{ grid-template-columns:1fr; }} }}
.chart-card {{ background:#161b22; border:1px solid #30363d; border-radius:12px; padding:16px; }}
.chart-card h3 {{ font-size:0.85rem; color:#8b949e; margin-bottom:10px; }}
canvas {{ max-height:260px; }}
.chart-panel {{ display:none; }}
.chart-panel.active {{ display:block; }}
</style>
</head>
<body>
<h1>ETF 回測｜買入持有 vs 定期定額</h1>
<p class="sub">
  B&amp;H：NT${TWD_LUMP:,} 一次投入 ／ DCA：每月 NT${TWD_MONTHLY:,}<br>
  USD 標的以 1 USD ≈ {int(TWD_PER_USD)} TWD 換算
</p>

<div class="wrap">

<!-- 分頁按鈕 -->
<div class="tabs">
  <button class="tab-btn active" data-tab="std">📊 標準回測</button>
  <button class="tab-btn"        data-tab="roll">🔄 滾動回測</button>
  <button class="tab-btn"        data-tab="calc">🧮 報酬計算機</button>
  <button class="tab-btn"        data-tab="lev">⚖️ 槓桿計算機</button>
</div>

<!-- ══ Tab 1：標準回測 ══ -->
<div id="tab-std" class="tab-panel active">

  <div class="toolbar">
    <label for="verSelect">資料版本：</label>
    <select class="sel" id="verSelect">
      <option value="latest">最新資料</option>
      <option value="y2024">截至 2024/12/31</option>
    </select>
    <label for="tickerSelect" style="margin-left:12px">選擇標的：</label>
    <select class="sel" id="tickerSelect">
{select_options}
    </select>
    <div style="display:flex;gap:16px;margin-left:auto;align-items:center;">
      <label style="display:flex;align-items:center;gap:6px;cursor:pointer;color:#e6edf3;">
        <input type="checkbox" id="chkBaH" checked
               style="width:15px;height:15px;accent-color:#3fb950;cursor:pointer;">
        <span style="font-size:0.88rem;">買入持有</span>
      </label>
      <label style="display:flex;align-items:center;gap:6px;cursor:pointer;color:#e6edf3;">
        <input type="checkbox" id="chkDCA"
               style="width:15px;height:15px;accent-color:#d2a8ff;cursor:pointer;">
        <span style="font-size:0.88rem;">定期定額</span>
      </label>
    </div>
  </div>

  <!-- 績效表格 -->
  <div class="card">
    <div class="card-title">績效摘要</div>
    <table>
      <thead><tr>
        <th>標的</th><th>策略</th><th>期間</th>
        <th>總投入</th><th>最終資產</th>
        <th>總報酬率</th><th>年化 CAGR</th>
        <th>最大回撤</th><th>Sharpe</th>
      </tr></thead>
      <tbody id="stdBody"></tbody>
    </table>
  </div>

  <!-- 圖表 -->
  <div class="card">
    <div class="card-title">圖表分析</div>
    <div id="std-panels"></div>
  </div>

  <!-- 逐年報酬 -->
  <div class="card">
    <div class="card-title">📅 逐年報酬明細</div>
    <div id="annual-panels"></div>
  </div>

  <!-- 歷年價格 -->
  <div class="card">
    <div class="card-title">📈 歷年月收盤價</div>
    <div id="price-panels"></div>
  </div>

</div><!-- /tab-std -->

<!-- ══ Tab 2：滾動回測 ══ -->
<div id="tab-roll" class="tab-panel">

  <div class="card">
    <div class="card-title">滾動回測統計（{ROLLING_YEARS} 年窗口，完整歷史資料，所有可能起始月份）</div>
    <p class="card-note">
      每個窗口 = 從某月開始投資、持滿 {ROLLING_YEARS} 年後的年化報酬率（CAGR）。<br>
      中位數 CAGR 代表：隨機一個時間點進場、持滿 {ROLLING_YEARS} 年，有一半機率優於此數字。
    </p>
    <div style="display:flex;gap:16px;margin-bottom:14px;">
      <label style="display:flex;align-items:center;gap:6px;cursor:pointer;color:#e6edf3;">
        <input type="checkbox" id="rollChkBaH" checked
               style="width:15px;height:15px;accent-color:#3fb950;cursor:pointer;">
        <span style="font-size:0.88rem;">買入持有</span>
      </label>
      <label style="display:flex;align-items:center;gap:6px;cursor:pointer;color:#e6edf3;">
        <input type="checkbox" id="rollChkDCA"
               style="width:15px;height:15px;accent-color:#d2a8ff;cursor:pointer;">
        <span style="font-size:0.88rem;">定期定額</span>
      </label>
    </div>
    <table>
      <thead><tr>
        <th>標的</th><th>策略</th><th>窗口數</th>
        <th>平均 CAGR</th><th>中位數 CAGR</th>
        <th>最佳 CAGR</th><th>最差 CAGR</th>
        <th>標準差</th><th>正報酬勝率</th>
      </tr></thead>
      <tbody id="rollBody"></tbody>
    </table>
  </div>

</div><!-- /tab-roll -->

<!-- ══ Tab 3：報酬計算機 ══ -->
<div id="tab-calc" class="tab-panel">

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start;">

    <!-- 方案 A -->
    <div>
      <div class="card" style="margin-bottom:14px;">
        <div class="card-title" style="display:flex;justify-content:space-between;align-items:center;">
          <span>💰 方案 A</span>
          <input id="calcTitleA" type="text" placeholder="方案名稱" maxlength="30"
                 style="background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:6px;
                        padding:4px 10px;font-size:0.82rem;outline:none;width:160px;text-align:right;">
        </div>
        <p class="card-note" style="margin-bottom:16px;">輸入多組「標的 + 本金 + 年化報酬率」，共用持有年數計算最終資產。</p>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
          <label style="color:#8b949e;font-size:0.88rem;white-space:nowrap;">持有年數</label>
          <input id="calcYearsA" type="number" value="20" min="1" max="50" step="1"
                 style="width:80px;background:#21262d;color:#e6edf3;border:1px solid #30363d;
                        border-radius:8px;padding:6px 10px;font-size:1rem;outline:none;text-align:center;">
          <span style="color:#8b949e;font-size:0.88rem;">年</span>
        </div>
        <div id="calcRowsA" style="display:flex;flex-direction:column;gap:8px;margin-bottom:14px;"></div>
        <button id="addRowBtnA"
                style="background:#21262d;color:#58a6ff;border:1px dashed #30363d;border-radius:8px;
                       padding:7px 0;font-size:0.83rem;cursor:pointer;width:100%;margin-bottom:14px;">
          ＋ 新增一組
        </button>
        <button id="calcBtnA"
                style="background:#1f6feb;color:#fff;border:none;border-radius:8px;
                       padding:9px 0;font-size:0.92rem;font-weight:600;cursor:pointer;width:100%;">
          計算
        </button>
      </div>
      <div id="calcResultA" style="display:none;">
        <div class="card" style="margin-bottom:14px;">
          <div class="card-title">方案 A 結果</div>
          <table>
            <thead><tr>
              <th>#</th><th>標的</th><th>本金（萬）</th><th>年化報酬</th>
              <th>最終資產（萬）</th><th>獲利（萬）</th><th>總報酬率</th><th>年化報酬率</th>
            </tr></thead>
            <tbody id="calcBodyA"></tbody>
            <tfoot id="calcFootA"></tfoot>
          </table>
        </div>
        <div class="card" id="calcPessimisticA">
          <div class="card-title" style="color:#f0883e;">⚠️ 悲觀情境：初期腰斬 −50%，持續 1 年</div>
          <p class="card-note" style="margin-bottom:12px;">
            假設投入後第 1 年腰斬 −50%、第 2 年持平，後段每年報酬率調升至 g，使全期算術平均年報酬仍等於輸入的年化報酬。<br>
            g = (N × r + 50) / (N − 1)　　悲觀終值 = 本金 × 0.5 × (1 + g%)^(N−1)
          </p>
          <table>
            <thead><tr>
              <th>#</th><th>標的</th><th>本金（萬）</th>
              <th>輸入年化</th><th>後段補償率 g</th>
              <th>悲觀終值（萬）</th><th>vs 正常少賺（萬）</th><th>有效 CAGR</th>
            </tr></thead>
            <tbody id="calcPessBodyA"></tbody>
            <tfoot id="calcPessFootA"></tfoot>
          </table>
        </div>
      </div>
    </div>

    <!-- 方案 B -->
    <div>
      <div class="card" style="margin-bottom:14px;">
        <div class="card-title" style="display:flex;justify-content:space-between;align-items:center;">
          <span>💰 方案 B</span>
          <input id="calcTitleB" type="text" placeholder="方案名稱" maxlength="30"
                 style="background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:6px;
                        padding:4px 10px;font-size:0.82rem;outline:none;width:160px;text-align:right;">
        </div>
        <p class="card-note" style="margin-bottom:16px;">輸入多組「標的 + 本金 + 年化報酬率」，共用持有年數計算最終資產。</p>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
          <label style="color:#8b949e;font-size:0.88rem;white-space:nowrap;">持有年數</label>
          <input id="calcYearsB" type="number" value="20" min="1" max="50" step="1"
                 style="width:80px;background:#21262d;color:#e6edf3;border:1px solid #30363d;
                        border-radius:8px;padding:6px 10px;font-size:1rem;outline:none;text-align:center;">
          <span style="color:#8b949e;font-size:0.88rem;">年</span>
        </div>
        <div id="calcRowsB" style="display:flex;flex-direction:column;gap:8px;margin-bottom:14px;"></div>
        <button id="addRowBtnB"
                style="background:#21262d;color:#58a6ff;border:1px dashed #30363d;border-radius:8px;
                       padding:7px 0;font-size:0.83rem;cursor:pointer;width:100%;margin-bottom:14px;">
          ＋ 新增一組
        </button>
        <button id="calcBtnB"
                style="background:#1f6feb;color:#fff;border:none;border-radius:8px;
                       padding:9px 0;font-size:0.92rem;font-weight:600;cursor:pointer;width:100%;">
          計算
        </button>
      </div>
      <div id="calcResultB" style="display:none;">
        <div class="card" style="margin-bottom:14px;">
          <div class="card-title">方案 B 結果</div>
          <table>
            <thead><tr>
              <th>#</th><th>標的</th><th>本金（萬）</th><th>年化報酬</th>
              <th>最終資產（萬）</th><th>獲利（萬）</th><th>總報酬率</th><th>年化報酬率</th>
            </tr></thead>
            <tbody id="calcBodyB"></tbody>
            <tfoot id="calcFootB"></tfoot>
          </table>
        </div>
        <div class="card" id="calcPessimisticB">
          <div class="card-title" style="color:#f0883e;">⚠️ 悲觀情境：初期腰斬 −50%，持續 1 年</div>
          <p class="card-note" style="margin-bottom:12px;">
            假設投入後第 1 年腰斬 −50%、第 2 年持平，後段每年報酬率調升至 g，使全期算術平均年報酬仍等於輸入的年化報酬。<br>
            g = (N × r + 50) / (N − 1)　　悲觀終值 = 本金 × 0.5 × (1 + g%)^(N−1)
          </p>
          <table>
            <thead><tr>
              <th>#</th><th>標的</th><th>本金（萬）</th>
              <th>輸入年化</th><th>後段補償率 g</th>
              <th>悲觀終值（萬）</th><th>vs 正常少賺（萬）</th><th>有效 CAGR</th>
            </tr></thead>
            <tbody id="calcPessBodyB"></tbody>
            <tfoot id="calcPessFootB"></tfoot>
          </table>
        </div>
      </div>
    </div>

  </div><!-- /grid -->

</div><!-- /tab-calc -->

<!-- ══ Tab 4：槓桿計算機 ══ -->
<div id="tab-lev" class="tab-panel">
  <div style="max-width:560px;margin:0 auto;">

    <div class="card" style="margin-bottom:18px;">
      <div class="card-title">⚖️ 槓桿率計算機</div>
      <p class="card-note" style="margin-bottom:20px;">
        槓桿率 = 總曝險 ÷ 淨資產<br>
        曝險2x 投資（如正2 ETF）的實際市場曝險為帳面金額 × 2。
      </p>

      <div style="display:grid;gap:14px;">

        <!-- 曝險 1x -->
        <div style="display:flex;align-items:center;gap:12px;">
          <label style="color:#8b949e;font-size:0.88rem;width:140px;flex-shrink:0;">曝險 1x 投資（萬）</label>
          <input id="lev1x" type="number" value="" min="0" step="1" placeholder="0"
                 style="flex:1;background:#21262d;color:#e6edf3;border:1px solid #30363d;
                        border-radius:8px;padding:8px 12px;font-size:0.95rem;outline:none;">
          <span style="color:#8b949e;font-size:0.82rem;width:24px;">萬</span>
        </div>

        <!-- 曝險 2x -->
        <div style="display:flex;align-items:center;gap:12px;">
          <label style="color:#8b949e;font-size:0.88rem;width:140px;flex-shrink:0;">曝險 2x 投資（萬）</label>
          <input id="lev2x" type="number" value="" min="0" step="1" placeholder="0"
                 style="flex:1;background:#21262d;color:#e6edf3;border:1px solid #30363d;
                        border-radius:8px;padding:8px 12px;font-size:0.95rem;outline:none;">
          <span style="color:#8b949e;font-size:0.82rem;width:24px;">萬</span>
        </div>

        <div style="border-top:1px solid #30363d;margin:2px 0;"></div>

        <!-- 負債 -->
        <div style="display:flex;align-items:center;gap:12px;">
          <label style="color:#8b949e;font-size:0.88rem;width:140px;flex-shrink:0;">負債（萬）</label>
          <input id="levDebt" type="number" value="" min="0" step="1" placeholder="0"
                 style="flex:1;background:#21262d;color:#e6edf3;border:1px solid #30363d;
                        border-radius:8px;padding:8px 12px;font-size:0.95rem;outline:none;">
          <span style="color:#8b949e;font-size:0.82rem;width:24px;">萬</span>
        </div>

        <!-- 淨資產 -->
        <div style="display:flex;align-items:center;gap:12px;">
          <label style="color:#8b949e;font-size:0.88rem;width:140px;flex-shrink:0;">淨資產（萬）</label>
          <input id="levNet" type="number" value="" min="1" step="1" placeholder="100"
                 style="flex:1;background:#21262d;color:#e6edf3;border:1px solid #30363d;
                        border-radius:8px;padding:8px 12px;font-size:0.95rem;outline:none;">
          <span style="color:#8b949e;font-size:0.82rem;width:24px;">萬</span>
        </div>
      </div>

      <button id="levCalcBtn"
              style="margin-top:20px;background:#1f6feb;color:#fff;border:none;border-radius:8px;
                     padding:10px 0;font-size:0.95rem;font-weight:600;cursor:pointer;width:100%;">
        計算槓桿率
      </button>
    </div>

    <!-- 結果 -->
    <div id="levResult" style="display:none;">
      <div class="card">
        <div class="card-title">計算結果</div>
        <div style="display:grid;gap:12px;margin-bottom:16px;">

          <div style="display:flex;justify-content:space-between;align-items:center;
                      padding:10px 14px;background:#21262d;border-radius:8px;">
            <span style="color:#8b949e;font-size:0.88rem;">曝險 1x 部位</span>
            <span id="levR1x" style="font-weight:600;"></span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;
                      padding:10px 14px;background:#21262d;border-radius:8px;">
            <span style="color:#8b949e;font-size:0.88rem;">曝險 2x 部位（帳面 × 2）</span>
            <span id="levR2x" style="font-weight:600;"></span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;
                      padding:10px 14px;background:#21262d;border-radius:8px;">
            <span style="color:#8b949e;font-size:0.88rem;">總曝險帳面（不含槓桿倍率）</span>
            <span id="levRFace" style="font-weight:600;"></span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;
                      padding:10px 14px;background:#21262d;border-radius:8px;border:1px solid #30363d;">
            <span style="color:#e6edf3;font-size:0.9rem;font-weight:600;">總曝險（含槓桿倍率）</span>
            <span id="levRTotal" style="font-weight:700;font-size:1rem;"></span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;
                      padding:10px 14px;background:#21262d;border-radius:8px;">
            <span style="color:#8b949e;font-size:0.88rem;">負債</span>
            <span id="levRDebt" style="font-weight:600;color:#f85149;"></span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;
                      padding:10px 14px;background:#21262d;border-radius:8px;">
            <span style="color:#8b949e;font-size:0.88rem;">淨資產</span>
            <span id="levRNet" style="font-weight:600;color:#3fb950;"></span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;
                      padding:10px 14px;background:#21262d;border-radius:8px;">
            <span style="color:#8b949e;font-size:0.88rem;">現金（總資產 − 曝險帳面）</span>
            <span id="levRCash" style="font-weight:600;"></span>
          </div>
        </div>

        <!-- 槓桿率主要顯示 -->
        <div id="levRateBox" style="padding:20px;border-radius:10px;text-align:center;">
          <div style="font-size:0.85rem;margin-bottom:6px;opacity:0.8;">槓桿率（總曝險 ÷ 淨資產）</div>
          <div id="levRate" style="font-size:2.4rem;font-weight:700;letter-spacing:1px;"></div>
          <div id="levRateNote" style="font-size:0.82rem;margin-top:8px;opacity:0.75;"></div>
        </div>
      </div>
    </div>

  </div>
</div><!-- /tab-lev -->

</div><!-- /wrap -->

<script>
const CHART_DATA   = {js_chart};
const ALL_STATS    = {js_stats};
const ROLLING_ROWS = {js_rolling};
const COST_C = '#444c56';

const BADGE_STYLE = {{
  '元大台灣50':           ['#1f3a5f','#79c0ff'],
  '元大台灣50正向2倍':    ['#3d1f1f','#ff7b72'],
  'Invesco QQQ':         ['#2d2438','#d2a8ff'],
  'ProShares Ultra QQQ': ['#2a1e0f','#f0883e'],
  'SPDR S&P 500':        ['#2e2a1a','#e3b341'],
  '元大龍頭台幣A':        ['#1e3a2e','#56d364'],
  '安聯台灣智慧基金':     ['#3d2a0f','#ffa657'],
  '安聯台灣科技基金':     ['#2a1e3d','#bc8cff'],
}};

function fmt(n, ccy) {{
  if (ccy==='TWD') return n>=1e6 ? (n/1e4).toFixed(0)+'萬' : n.toLocaleString();
  return n>=1e6 ? '$'+(n/1e6).toFixed(2)+'M' : '$'+Math.round(n).toLocaleString();
}}
function pctCell(v) {{
  const cls = v >= 0 ? 'pos' : 'neg';
  return `<td class="${{cls}}">${{v >= 0 ? '+' : ''}}${{v.toFixed(2)}}%</td>`;
}}

/* ══ 分頁切換 ══ */
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  }});
}});

/* ══ 標準回測 ══ */

let curVer    = 'latest';
let curTicker = '{DEFAULT_TICKER}';
const stdRendered = {{}};   // ver+ticker -> true

/* 建立所有 ticker 的 panel 骨架（只建一次） */
const stdPanels = document.getElementById('std-panels');
{list(ALL_TICKERS.keys())!r}.forEach(tk => {{
  const id  = tk.replace(/\\./g,'_');
  const div = document.createElement('div');
  div.id = 'sp_' + id;
  div.className = 'chart-panel' + (tk === '{DEFAULT_TICKER}' ? ' active' : '');
  div.innerHTML = `<div class="row2">
    <div class="chart-card"><h3>累積資產</h3><canvas id="sc_latest_${{id}}"></canvas><canvas id="sc_y2024_${{id}}" style="display:none"></canvas></div>
    <div class="chart-card"><h3>回撤幅度（%）</h3><canvas id="sd_latest_${{id}}"></canvas><canvas id="sd_y2024_${{id}}" style="display:none"></canvas></div>
  </div>`;
  stdPanels.appendChild(div);
}});

function dsLine(label, data, color, dashed, fill) {{
  return {{
    label, data,
    parsing: {{xAxisKey:'x', yAxisKey:'y'}},
    borderColor: color,
    backgroundColor: color + (fill ? '25' : '00'),
    fill: !!fill, borderWidth: dashed ? 1.5 : 2,
    borderDash: dashed ? [5,4] : [], pointRadius: 0, tension: 0.2,
  }};
}}

function mkChart(id, datasets, yFmt) {{
  const c = document.getElementById(id);
  if (!c) return;
  new Chart(c.getContext('2d'), {{
    type: 'line',
    data: {{ datasets }},
    options: {{
      responsive: true,
      interaction: {{ mode:'index', intersect:false }},
      plugins: {{
        legend: {{ labels: {{ color:'#8b949e', boxWidth:12, font:{{size:11}} }} }},
        tooltip: {{
          backgroundColor:'#21262d', titleColor:'#f0f6fc', bodyColor:'#cdd9e5',
          borderColor:'#30363d', borderWidth:1,
          callbacks: {{ label: c => ' ' + c.dataset.label + '：' + yFmt(c.parsed.y) }},
        }},
      }},
      scales: {{
        x: {{ type:'category', ticks:{{color:'#8b949e',maxTicksLimit:7}}, grid:{{color:'#21262d'}} }},
        y: {{ ticks:{{color:'#8b949e',callback:yFmt}}, grid:{{color:'#21262d'}} }},
      }},
    }},
  }});
}}

function renderStdChart(ver, tk) {{
  const key = ver + '_' + tk;
  if (stdRendered[key]) return;
  stdRendered[key] = true;

  const t  = CHART_DATA[ver][tk];
  const id = tk.replace(/\\./g,'_');
  const [c1, c2] = t.colors;
  const yFmt = v => fmt(v, t.currency);

  mkChart('sc_' + ver + '_' + id, [
    dsLine('買入持有', t.bah,  c1, false, false),
    dsLine('定期定額', t.dca,  c2, false, false),
    dsLine('累計成本', t.cost, COST_C, true, false),
  ], yFmt);

  mkChart('sd_' + ver + '_' + id, [
    dsLine('買入持有', t.bah_dd, c1, false, true),
    dsLine('定期定額', t.dca_dd, c2, false, true),
  ], v => v.toFixed(1) + '%');
}}

function showStdCanvas(ver, tk) {{
  const id = tk.replace(/\\./g,'_');
  ['sc','sd'].forEach(prefix => {{
    ['latest','y2024'].forEach(v => {{
      const el = document.getElementById(prefix + '_' + v + '_' + id);
      if (el) el.style.display = (v === ver) ? '' : 'none';
    }});
  }});
  renderStdChart(ver, tk);
}}

function renderStdTable(ver) {{
  const tbody = document.getElementById('stdBody');
  tbody.innerHTML = '';
  const curOf = {{}};
  Object.entries(CHART_DATA[ver]).forEach(([tk, t]) => {{
    curOf[t.name + ' 買入持有'] = t.currency;
    curOf[t.name + ' 定期定額'] = t.currency;
  }});
  ALL_STATS[ver].forEach(s => {{
    const ccy = curOf[s.label] || 'TWD';
    const pos  = parseFloat(s.total_ret) >= 0;
    const isB  = s.label.endsWith('買入持有');
    const name = s.label.replace(/ (買入持有|定期定額)$/, '');
    const bc   = BADGE_STYLE[name] || ['#21262d','#cdd9e5'];
    const sc   = isB ? '#3fb950' : '#d2a8ff';
    const note = s.note ? ` <span class="muted">(${{s.note}})</span>` : '';
    tbody.innerHTML += `<tr data-strat="${{isB?'bah':'dca'}}" data-ticker="${{name}}">
      <td><span class="badge" style="background:${{bc[0]}};color:${{bc[1]}}">${{name}}</span>${{note}}</td>
      <td><span class="strat ${{isB?'bah':'dca'}}">${{isB?'買入持有':'定期定額'}}</span></td>
      <td>${{s.years}}年</td>
      <td>${{fmt(s.total_invested, ccy)}}</td>
      <td class="${{pos?'pos':'neg'}}">${{fmt(s.final_value, ccy)}}</td>
      <td class="${{pos?'pos':'neg'}}">${{s.total_ret}}</td>
      <td class="${{pos?'pos':'neg'}}">${{s.cagr}}</td>
      <td class="neg">${{s.max_dd}}</td>
      <td>${{s.sharpe}}</td>
    </tr>`;
  }});
  applyStratFilter();
  highlightTicker(curTicker);
}}

function highlightTicker(tk) {{
  const name = CHART_DATA[curVer][tk]?.name || '';
  document.querySelectorAll('#stdBody tr').forEach(tr => {{
    tr.classList.toggle('row-highlight', tr.dataset.ticker === name);
  }});
}}

function applyStratFilter() {{
  const showBaH = document.getElementById('chkBaH').checked;
  const showDCA = document.getElementById('chkDCA').checked;
  document.querySelectorAll('#stdBody tr').forEach(tr => {{
    const s = tr.dataset.strat;
    tr.style.display = (s === 'bah' && showBaH) || (s === 'dca' && showDCA) ? '' : 'none';
  }});
}}

function showStdPanel(tk) {{
  document.querySelectorAll('#std-panels .chart-panel').forEach(p => p.classList.remove('active'));
  const panel = document.getElementById('sp_' + tk.replace(/\\./g,'_'));
  if (panel) panel.classList.add('active');
  showStdCanvas(curVer, tk);
}}

document.getElementById('tickerSelect').addEventListener('change', e => {{
  curTicker = e.target.value;
  showStdPanel(curTicker);
  showAnnual(curVer, curTicker);
  showPrice(curVer, curTicker);
  highlightTicker(curTicker);
}});

document.getElementById('verSelect').addEventListener('change', e => {{
  curVer = e.target.value;
  renderStdTable(curVer);
  showStdCanvas(curVer, curTicker);
  showAnnual(curVer, curTicker);
  showPrice(curVer, curTicker);
}});

/* ── 逐年表格 ── */

const annualPanels = document.getElementById('annual-panels');
{list(ALL_TICKERS.keys())!r}.forEach(tk => {{
  const id  = tk.replace(/\\./g,'_');
  const div = document.createElement('div');
  div.id = 'ap_' + id;
  div.className = 'chart-panel' + (tk === '{DEFAULT_TICKER}' ? ' active' : '');
  annualPanels.appendChild(div);
}});

const annualRendered = {{}};

function renderAnnual(ver, tk) {{
  const key = ver + '_' + tk;
  if (annualRendered[key]) return;
  annualRendered[key] = true;

  const t    = CHART_DATA[ver][tk];
  const rows = t.annual;
  const ccy  = t.currency;
  const [c1, c2] = t.colors;

  const fmtV = v => fmt(v, ccy);
  const pctSpan = (v, bold) => {{
    if (v == null) return '<span class="muted">—</span>';
    const cls = (v >= 0 ? 'pos' : 'neg') + (bold ? ' strong' : '');
    return `<span class="${{cls}}">${{v >= 0 ? '+' : ''}}${{v.toFixed(2)}}%</span>`;
  }};

  const thead = `<thead><tr style="position:sticky;top:0;z-index:1;">
    <th>年份</th>
    <th style="color:${{c1}}">B&H 年末資產</th>
    <th style="color:${{c1}}">B&H 當年漲跌</th>
    <th style="color:${{c1}}">B&H 累計 CAGR</th>
    <th style="color:${{c2}}">DCA 年末資產</th>
    <th style="color:${{c2}}">DCA 累計投入</th>
    <th style="color:${{c2}}">DCA 累計總報酬</th>
    <th style="color:${{c2}}">DCA 累計 CAGR</th>
  </tr></thead>`;

  const tbody = rows.map((r, i) => `<tr>
    <td class="muted">${{r.year}}</td>
    <td>${{fmtV(r.bah_val)}}</td>
    <td>${{pctSpan(r.bah_yoy, false)}}</td>
    <td>${{pctSpan(r.bah_cagr, true)}}</td>
    <td>${{fmtV(r.dca_val)}}</td>
    <td class="muted">${{fmtV(r.invested)}}</td>
    <td>${{pctSpan(r.dca_tr, false)}}</td>
    <td>${{pctSpan(r.dca_cagr, true)}}</td>
  </tr>`).join('');

  const panel = document.getElementById('ap_' + tk.replace(/\\./g,'_'));
  panel.innerHTML = `
    <div style="overflow-x:auto;max-height:420px;overflow-y:auto;">
      <table style="font-size:0.8rem;">${{thead}}<tbody>${{tbody}}</tbody></table>
    </div>`;
}}

function showAnnual(ver, tk) {{
  document.querySelectorAll('#annual-panels .chart-panel').forEach(p => p.classList.remove('active'));
  const panel = document.getElementById('ap_' + tk.replace(/\\./g,'_'));
  if (panel) panel.classList.add('active');
  renderAnnual(ver, tk);
}}

/* ── 歷年價格圖 ── */

const pricePanels = document.getElementById('price-panels');
{list(ALL_TICKERS.keys())!r}.forEach(tk => {{
  const id  = tk.replace(/\\./g,'_');
  const div = document.createElement('div');
  div.id = 'pp_' + id;
  div.className = 'chart-panel' + (tk === '{DEFAULT_TICKER}' ? ' active' : '');
  div.innerHTML = `<canvas id="pc_${{id}}" style="max-height:320px;"></canvas>`;
  pricePanels.appendChild(div);
}});

const priceRendered = {{}};

function renderPrice(ver, tk) {{
  const key = ver + '_' + tk;
  if (priceRendered[key]) return;
  priceRendered[key] = true;

  const t    = CHART_DATA[ver][tk];
  const [c1] = t.colors;
  const ccy  = t.currency;
  const id   = tk.replace(/\\./g,'_');
  const c    = document.getElementById('pc_' + id);
  if (!c) return;

  new Chart(c.getContext('2d'), {{
    type: 'line',
    data: {{
      datasets: [{{
        label: t.name,
        data:  t.price,
        parsing: {{ xAxisKey: 'x', yAxisKey: 'y' }},
        borderColor: c1,
        backgroundColor: c1 + '18',
        fill: true,
        borderWidth: 1.8,
        pointRadius: 0,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: c1,
        tension: 0.2,
      }}],
    }},
    options: {{
      responsive: true,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: '#21262d',
          titleColor: '#f0f6fc',
          bodyColor: '#cdd9e5',
          borderColor: '#30363d',
          borderWidth: 1,
          callbacks: {{
            title: items => items[0].label,
            label: item => ` ${{t.name}}：${{item.parsed.y.toLocaleString(undefined, {{minimumFractionDigits:2, maximumFractionDigits:2}})}} ${{ccy}}`,
          }},
        }},
      }},
      scales: {{
        x: {{
          type: 'category',
          ticks: {{ color: '#8b949e', maxTicksLimit: 10 }},
          grid: {{ color: '#21262d' }},
        }},
        y: {{
          ticks: {{
            color: '#8b949e',
            callback: v => ccy === 'TWD' ? v.toLocaleString() : '$' + v.toLocaleString(),
          }},
          grid: {{ color: '#21262d' }},
        }},
      }},
    }},
  }});
}}

function showPrice(ver, tk) {{
  document.querySelectorAll('#price-panels .chart-panel').forEach(p => p.classList.remove('active'));
  const panel = document.getElementById('pp_' + tk.replace(/\\./g,'_'));
  if (panel) panel.classList.add('active');
  renderPrice(ver, tk);
}}

document.getElementById('chkBaH').addEventListener('change', applyStratFilter);
document.getElementById('chkDCA').addEventListener('change', applyStratFilter);

/* 初始化標準回測 */
renderStdTable('latest');
renderStdChart('latest', '{DEFAULT_TICKER}');
showAnnual('latest', '{DEFAULT_TICKER}');
showPrice('latest', '{DEFAULT_TICKER}');


/* ══ 滾動回測 ══ */

const rollBody = document.getElementById('rollBody');
ROLLING_ROWS.forEach((r, i) => {{
  const bc = BADGE_STYLE[r.name] || ['#21262d','#cdd9e5'];
  const badge = `<span class="badge" style="background:${{bc[0]}};color:${{bc[1]}}">${{r.name}}</span>`;
  const dateTag = (range) =>
    `<div style="color:#6e7681;font-size:0.70rem;font-weight:normal;margin-top:3px;line-height:1.3;">${{range}}</div>`;
  rollBody.innerHTML += `
    <tr class="sep" data-roll-strat="bah">
      <td>${{badge}}</td>
      <td><span class="strat bah">買入持有</span></td>
      <td class="muted">${{r.bah.n}}</td>
      ${{pctCell(r.bah.mean)}}
      <td class="strong ${{r.bah.median>=0?'pos':'neg'}}">${{r.bah.median>=0?'+':''}}${{r.bah.median.toFixed(2)}}%</td>
      <td class="pos">${{r.bah.best >= 0 ? '+' : ''}}${{r.bah.best.toFixed(2)}}%${{dateTag(r.bah.best_range)}}</td>
      <td class="neg">${{r.bah.worst.toFixed(2)}}%${{dateTag(r.bah.worst_range)}}</td>
      <td class="muted">${{r.bah.std.toFixed(2)}}%</td>
      <td>${{r.bah.win_pct.toFixed(0)}}%</td>
    </tr>
    <tr data-roll-strat="dca">
      <td>${{badge}}</td>
      <td><span class="strat dca">定期定額</span></td>
      <td class="muted">${{r.dca.n}}</td>
      ${{pctCell(r.dca.mean)}}
      <td class="strong ${{r.dca.median>=0?'pos':'neg'}}">${{r.dca.median>=0?'+':''}}${{r.dca.median.toFixed(2)}}%</td>
      <td class="pos">${{r.dca.best >= 0 ? '+' : ''}}${{r.dca.best.toFixed(2)}}%${{dateTag(r.dca.best_range)}}</td>
      <td class="neg">${{r.dca.worst.toFixed(2)}}%${{dateTag(r.dca.worst_range)}}</td>
      <td class="muted">${{r.dca.std.toFixed(2)}}%</td>
      <td>${{r.dca.win_pct.toFixed(0)}}%</td>
    </tr>`;
}});

function applyRollFilter() {{
  const showBaH = document.getElementById('rollChkBaH').checked;
  const showDCA = document.getElementById('rollChkDCA').checked;
  document.querySelectorAll('#rollBody tr').forEach(tr => {{
    const s = tr.dataset.rollStrat;
    tr.style.display = (s === 'bah' && showBaH) || (s === 'dca' && showDCA) ? '' : 'none';
  }});
}}

document.getElementById('rollChkBaH').addEventListener('change', applyRollFilter);
document.getElementById('rollChkDCA').addEventListener('change', applyRollFilter);
applyRollFilter();


/* ══ 報酬計算機 ══ */

const calcRowCounts = {{ A: 0, B: 0 }};

function calcRowHTML(side, idx, lbl, principal, rate) {{
  const rowId = `calcrow_${{side}}_${{idx}}`;
  return `<div id="${{rowId}}" style="display:flex;align-items:center;gap:8px;">
    <span style="color:#8b949e;font-size:0.78rem;width:16px;text-align:right;">${{idx}}</span>
    <input type="text" placeholder="標的" value="${{lbl ?? ''}}"
           style="width:100px;background:#21262d;color:#e6edf3;border:1px solid #30363d;
                  border-radius:8px;padding:6px 10px;font-size:0.85rem;outline:none;">
    <input type="number" placeholder="本金（萬）" value="${{principal ?? ''}}" min="0" step="0.1"
           style="flex:1;background:#21262d;color:#e6edf3;border:1px solid #30363d;
                  border-radius:8px;padding:6px 10px;font-size:0.85rem;outline:none;">
    <span style="color:#8b949e;font-size:0.78rem;">萬 ×</span>
    <input type="number" placeholder="年化 %" value="${{rate ?? ''}}" min="-50" max="200" step="0.1"
           style="width:90px;background:#21262d;color:#e6edf3;border:1px solid #30363d;
                  border-radius:8px;padding:6px 10px;font-size:0.85rem;outline:none;">
    <span style="color:#8b949e;font-size:0.78rem;">%</span>
    <button onclick="document.getElementById('${{rowId}}').remove(); saveCalc('${{side}}')"
            style="background:none;border:none;color:#f85149;font-size:1rem;cursor:pointer;
                   padding:3px 6px;line-height:1;" title="刪除">✕</button>
  </div>`;
}}

function addCalcRow(side, lbl, principal, rate) {{
  calcRowCounts[side]++;
  const idx = calcRowCounts[side];
  const container = document.getElementById('calcRows' + side);
  container.insertAdjacentHTML('beforeend', calcRowHTML(side, idx, lbl, principal, rate));
  container.querySelectorAll('input').forEach(inp => inp.addEventListener('input', () => saveCalc(side)));
}}

/* localStorage */
function saveCalc(side) {{
  const rows = [...document.getElementById('calcRows' + side).children];
  const entries = rows.map(row => {{
    const inp = row.querySelectorAll('input');
    return {{ lbl: inp[0].value, p: inp[1].value, r: inp[2].value }};
  }});
  const title = document.getElementById('calcTitle' + side).value;
  const years = document.getElementById('calcYears' + side).value;
  localStorage.setItem('calc_' + side, JSON.stringify({{ title, years, entries }}));
}}

function loadCalc(side) {{
  const raw = localStorage.getItem('calc_' + side);
  if (!raw) return false;
  try {{
    const data = JSON.parse(raw);
    document.getElementById('calcTitle' + side).value = data.title || '';
    document.getElementById('calcYears' + side).value = data.years || 20;
    (data.entries || []).forEach(e => addCalcRow(side, e.lbl, e.p, e.r));
    return true;
  }} catch(e) {{ return false; }}
}}

function runCalc(side) {{
  const years = parseFloat(document.getElementById('calcYears' + side).value);
  if (!years || years <= 0) {{ alert('請輸入正確的持有年數'); return; }}

  const rows = [...document.getElementById('calcRows' + side).children];
  const entries = [];
  for (const row of rows) {{
    const inp = row.querySelectorAll('input');
    const lbl = inp[0].value.trim();
    const p   = parseFloat(inp[1].value);
    const r   = parseFloat(inp[2].value);
    if (isNaN(p) || isNaN(r)) {{ alert('請填寫每一列的本金與年化報酬率'); return; }}
    entries.push({{ lbl, p, r }});
  }}
  if (entries.length === 0) {{ alert('請至少新增一組'); return; }}

  let totalPrincipal = 0, totalFinal = 0;
  const tbody = document.getElementById('calcBody' + side);
  tbody.innerHTML = '';

  entries.forEach((e, i) => {{
    const final   = e.p * Math.pow(1 + e.r / 100, years);
    const profit  = final - e.p;
    const tr      = profit / e.p;
    const rowCagr = (Math.pow(final / e.p, 1 / years) - 1) * 100;
    totalPrincipal += e.p;
    totalFinal     += final;
    tbody.innerHTML += `<tr>
      <td class="muted">${{i + 1}}</td>
      <td>${{e.lbl || '<span class="muted">—</span>'}}</td>
      <td>${{e.p.toLocaleString(undefined, {{maximumFractionDigits:2}})}} 萬</td>
      <td class="${{e.r >= 0 ? 'pos' : 'neg'}}">${{e.r >= 0 ? '+' : ''}}${{e.r}}%</td>
      <td class="pos strong">${{final.toFixed(2)}} 萬</td>
      <td class="${{profit >= 0 ? 'pos' : 'neg'}}">${{profit >= 0 ? '+' : ''}}${{profit.toFixed(2)}} 萬</td>
      <td class="${{tr >= 0 ? 'pos' : 'neg'}}">${{tr >= 0 ? '+' : ''}}${{(tr * 100).toFixed(1)}}%</td>
      <td class="${{rowCagr >= 0 ? 'pos' : 'neg'}}">${{rowCagr >= 0 ? '+' : ''}}${{rowCagr.toFixed(2)}}%</td>
    </tr>`;
  }});

  const totalProfit = totalFinal - totalPrincipal;
  const totalTr     = totalProfit / totalPrincipal;
  const totalCagr   = (Math.pow(totalFinal / totalPrincipal, 1 / years) - 1) * 100;
  document.getElementById('calcFoot' + side).innerHTML = `
    <tr style="border-top:2px solid #30363d;background:#21262d;">
      <td colspan="3" style="color:#8b949e;font-weight:600;">合計（持有 ${{years}} 年）</td>
      <td>—</td>
      <td class="pos strong" style="font-size:1rem;">${{totalFinal.toFixed(2)}} 萬</td>
      <td class="${{totalProfit >= 0 ? 'pos' : 'neg'}} strong">${{totalProfit >= 0 ? '+' : ''}}${{totalProfit.toFixed(2)}} 萬</td>
      <td class="${{totalTr >= 0 ? 'pos' : 'neg'}} strong">${{totalTr >= 0 ? '+' : ''}}${{(totalTr * 100).toFixed(1)}}%</td>
      <td class="${{totalCagr >= 0 ? 'pos' : 'neg'}} strong">${{totalCagr >= 0 ? '+' : ''}}${{totalCagr.toFixed(2)}}%</td>
    </tr>`;
  document.getElementById('calcResult' + side).style.display = 'block';

  /* ── 悲觀情境 ── */
  const pessTbody  = document.getElementById('calcPessBody' + side);
  const pessFootEl = document.getElementById('calcPessFoot' + side);
  pessTbody.innerHTML = '';
  if (pessFootEl) pessFootEl.innerHTML = '';

  if (years <= 1) {{
    pessTbody.innerHTML = `<tr><td colspan="8" style="color:#f85149;text-align:center;">持有年數須大於 1 年才能計算悲觀情境</td></tr>`;
  }} else {{
    let totalNormal = 0, totalPess = 0, totalPrinc = 0;

    entries.forEach((e, i) => {{
      const g         = (years * e.r + 50) / (years - 1);
      const pessFinal = e.p * 0.5 * Math.pow(1 + g / 100, years - 1);
      const normFinal = e.p * Math.pow(1 + e.r / 100, years);
      const diff      = pessFinal - normFinal;
      const effCagr   = (Math.pow(pessFinal / e.p, 1 / years) - 1) * 100;
      totalPrinc  += e.p;
      totalNormal += normFinal;
      totalPess   += pessFinal;
      pessTbody.innerHTML += `<tr>
        <td class="muted">${{i + 1}}</td>
        <td>${{e.lbl || '<span class="muted">—</span>'}}</td>
        <td>${{e.p.toLocaleString(undefined,{{maximumFractionDigits:2}})}} 萬</td>
        <td class="${{e.r>=0?'pos':'neg'}}">${{e.r>=0?'+':''}}${{e.r}}%</td>
        <td style="color:#f0883e;">${{g>=0?'+':''}}${{g.toFixed(2)}}%</td>
        <td class="strong">${{pessFinal.toFixed(2)}} 萬</td>
        <td class="neg">${{diff.toFixed(2)}} 萬</td>
        <td class="${{effCagr>=0?'pos':'neg'}}">${{effCagr>=0?'+':''}}${{effCagr.toFixed(2)}}%</td>
      </tr>`;
    }});

    const totalDiff    = totalPess - totalNormal;
    const totalEffCagr = (Math.pow(totalPess / totalPrinc, 1 / years) - 1) * 100;
    if (pessFootEl) pessFootEl.innerHTML = `
      <tr style="border-top:2px solid #30363d;background:#21262d;">
        <td colspan="3" style="color:#f0883e;font-weight:600;">悲觀合計（持有 ${{years}} 年）</td>
        <td>—</td><td>—</td>
        <td class="strong">${{totalPess.toFixed(2)}} 萬</td>
        <td class="neg">${{totalDiff.toFixed(2)}} 萬</td>
        <td class="${{totalEffCagr>=0?'pos':'neg'}} strong">${{totalEffCagr>=0?'+':''}}${{totalEffCagr.toFixed(2)}}%</td>
      </tr>`;
  }}

  saveCalc(side);
}}

['A','B'].forEach(side => {{
  document.getElementById('addRowBtn' + side).addEventListener('click', () => {{
    addCalcRow(side); saveCalc(side);
  }});
  document.getElementById('calcBtn'   + side).addEventListener('click', () => runCalc(side));
  document.getElementById('calcTitle' + side).addEventListener('input', () => saveCalc(side));
  document.getElementById('calcYears' + side).addEventListener('input', () => saveCalc(side));

  /* 從 localStorage 恢復，或套入預設值 */
  const restored = loadCalc(side);
  if (!restored) {{
    if (side === 'A') {{
      addCalcRow('A', '0050', 100, 15);
      addCalcRow('A', 'QLD',  100, 30);
    }} else {{
      addCalcRow('B', 'QQQ',  100, 15);
      addCalcRow('B', 'QLD',  100, 30);
    }}
  }}
}});


/* ══ 槓桿計算機 ══ */

/* 槓桿計算機 localStorage */
const LEV_IDS = ['lev1x','lev2x','levDebt','levNet'];
LEV_IDS.forEach(id => {{
  const el = document.getElementById(id);
  const saved = localStorage.getItem('lev_' + id);
  if (saved !== null) el.value = saved;
  el.addEventListener('input', () => localStorage.setItem('lev_' + id, el.value));
}});

document.getElementById('levCalcBtn').addEventListener('click', () => {{
  const v1x  = parseFloat(document.getElementById('lev1x').value)   || 0;
  const v2x  = parseFloat(document.getElementById('lev2x').value)   || 0;
  const debt = parseFloat(document.getElementById('levDebt').value) || 0;
  const net  = parseFloat(document.getElementById('levNet').value);

  if (!net || net <= 0) {{ alert('請輸入淨資產'); return; }}

  const exp1x = v1x;
  const exp2x = v2x * 2;
  const total = exp1x + exp2x;
  const ratio = total / net;

  const fw = v => v.toLocaleString(undefined, {{maximumFractionDigits:1}}) + ' 萬';

  const cash = (net + debt) - (v1x + v2x);
  const cashEl = document.getElementById('levRCash');

  document.getElementById('levR1x').textContent    = fw(exp1x);
  document.getElementById('levR2x').textContent    = `${{fw(exp2x)}}（帳面 ${{fw(v2x)}}）`;
  document.getElementById('levRFace').textContent  = fw(v1x + v2x);
  document.getElementById('levRTotal').textContent = fw(total);
  document.getElementById('levRDebt').textContent  = fw(debt);
  document.getElementById('levRNet').textContent   = fw(net);
  cashEl.textContent   = (cash >= 0 ? '' : '') + fw(cash);
  cashEl.style.color   = cash >= 0 ? '#3fb950' : '#f85149';
  document.getElementById('levRate').textContent   = ratio.toFixed(2) + 'x';

  const box  = document.getElementById('levRateBox');
  const rate = document.getElementById('levRate');
  const note = document.getElementById('levRateNote');

  if (ratio < 1.5) {{
    box.style.cssText  += ';background:#1a2e1a;border:1px solid #2ea043;';
    rate.style.color    = '#3fb950';
    note.textContent    = '槓桿偏低，風險可控';
  }} else if (ratio < 2.5) {{
    box.style.cssText  += ';background:#2e2a10;border:1px solid #9e6a03;';
    rate.style.color    = '#e3b341';
    note.textContent    = '中等槓桿，注意市場回撤風險';
  }} else if (ratio < 4) {{
    box.style.cssText  += ';background:#2e1a10;border:1px solid #bd561d;';
    rate.style.color    = '#f0883e';
    note.textContent    = '槓桿偏高，大幅回撤可能造成強制賣出';
  }} else {{
    box.style.cssText  += ';background:#2e1010;border:1px solid #b91c1c;';
    rate.style.color    = '#f85149';
    note.textContent    = '極高槓桿，單日大幅下跌即可能歸零';
  }}

  document.getElementById('levResult').style.display = 'block';
}});

</script>
</body>
</html>"""

out = os.path.join(OUTPUT_DIR, "backtest_result.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\n✅ 完成！請用瀏覽器開啟：\n   {out}")
