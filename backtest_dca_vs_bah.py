"""
0050 vs QQQ — 定期定額(DCA) vs 買入持有(Buy & Hold) 回測
============================================================
使用方式：
  1. 安裝套件：pip install yfinance pandas matplotlib openpyxl
  2. 執行：python backtest_dca_vs_bah.py
  3. 結果會自動輸出 backtest_chart.png 與 backtest_results.xlsx

注意：
  - 0050.TW 於 2003/06 上市，最多回溯至 2003 年（約 22 年）
  - QQQ 於 1999/03 上市，最多回溯至 1999 年（約 26 年）
  - 資料來源：Yahoo Finance（需要網路連線）
  - 所有金額皆為標的本身貨幣（0050 = TWD，QQQ = USD）
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 無顯示環境也可執行；如要互動視窗改成 "TkAgg"
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import warnings, os

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
#  ★ 在這裡調整參數
# ═══════════════════════════════════════════════════════════
MONTHLY_DCA = 10_000        # 每月定期定額金額（單位：標的本身幣別）
OUTPUT_DIR  = "./"          # 輸出資料夾（預設為同目錄）

TICKERS = {
    "0050.TW": "元大台灣50 (0050)",
    "QQQ":     "Invesco QQQ (QQQ)",
}
# ═══════════════════════════════════════════════════════════


def download_data(tickers: dict) -> dict:
    """下載各標的歷史日收盤價"""
    data = {}
    for ticker, name in tickers.items():
        print(f"  下載 {name} 資料中...")
        df = yf.download(ticker, start="1993-01-01", auto_adjust=True, progress=False)
        df = df[["Close"]].copy()
        df.columns = ["Close"]
        df.dropna(inplace=True)
        data[ticker] = df
        print(f"    ✓ {df.index[0].date()} ~ {df.index[-1].date()}  ({len(df)} 筆日資料)")
    return data


def run_backtest(prices: pd.Series, monthly_amount: float) -> dict:
    """
    對單一標的執行 DCA 與 B&H 回測。
    回傳包含每月淨值序列與績效統計的 dict。
    """
    # 取每月第一個交易日價格
    monthly = prices.resample("MS").first().dropna()
    n_months = len(monthly)

    # ── 定期定額 DCA ──────────────────────────
    shares_dca = 0.0
    portfolio_dca = []
    for price in monthly:
        shares_dca += monthly_amount / price
        portfolio_dca.append(shares_dca * price)

    series_dca = pd.Series(portfolio_dca, index=monthly.index, name="DCA")
    cost_dca   = pd.Series(
        [monthly_amount * (i + 1) for i in range(n_months)],
        index=monthly.index, name="DCA_cost"
    )

    # ── 買入持有 B&H ──────────────────────────
    lump_sum   = monthly_amount * n_months   # 與 DCA 總投入相同，以利比較
    shares_bah = lump_sum / monthly.iloc[0]
    series_bah = (shares_bah * monthly).rename("BaH")
    cost_bah   = pd.Series([lump_sum] * n_months, index=monthly.index, name="BaH_cost")

    # ── 績效統計 ──────────────────────────────
    def calc_stats(series: pd.Series, total_cost: float, label: str) -> dict:
        final_val  = series.iloc[-1]
        total_ret  = (final_val - total_cost) / total_cost
        years      = (series.index[-1] - series.index[0]).days / 365.25
        cagr       = (final_val / total_cost) ** (1 / years) - 1
        roll_max   = series.cummax()
        max_dd     = ((series - roll_max) / roll_max).min()
        return {
            "策略":       label,
            "期間":       f"{series.index[0].year}–{series.index[-1].year}",
            "總投入":     f"{total_cost:,.0f}",
            "最終價值":   f"{final_val:,.0f}",
            "總報酬率":   f"{total_ret:.1%}",
            "年化報酬率": f"{cagr:.2%}",
            "最大回撤":   f"{max_dd:.1%}",
        }

    return {
        "dca":           series_dca,
        "bah":           series_bah,
        "cost_dca":      cost_dca,
        "cost_bah":      cost_bah,
        "monthly_prices": monthly,
        "stats": [
            calc_stats(series_dca, cost_dca.iloc[-1],  "定期定額 DCA"),
            calc_stats(series_bah, cost_bah.iloc[-1],  "買入持有 B&H"),
        ],
    }


def plot_results(results: dict, tickers: dict, output_path: str):
    """生成深色主題比較圖表"""
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#0f1117")
    gs  = GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

    C = {"dca": "#00d4ff", "bah": "#ff7f50", "cost": "#555566"}

    def fmt_k(x, _):
        if abs(x) >= 1e9: return f"{x/1e9:.1f}B"
        if abs(x) >= 1e6: return f"{x/1e6:.1f}M"
        if abs(x) >= 1e3: return f"{x/1e3:.0f}K"
        return f"{x:.0f}"

    for col, (ticker, name) in enumerate(tickers.items()):
        r = results[ticker]

        # 上圖：累積資產
        ax1 = fig.add_subplot(gs[0, col])
        ax1.fill_between(r["cost_dca"].index, r["cost_dca"],
                         alpha=0.12, color=C["cost"], label="_")
        ax1.plot(r["dca"].index, r["dca"],      color=C["dca"], lw=1.8, label="DCA 資產")
        ax1.plot(r["bah"].index, r["bah"],      color=C["bah"], lw=1.8, label="B&H 資產")
        ax1.plot(r["cost_dca"].index, r["cost_dca"],
                 color=C["cost"], lw=1.0, ls="--", label="累計成本")
        ax1.set_title(f"{name}\n累積資產對比", color="white", fontsize=11, pad=8)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
        ax1.legend(fontsize=8, facecolor="#1e2030", labelcolor="white", framealpha=0.8)
        ax1.set_facecolor("#1e2030")
        ax1.tick_params(colors="gray")
        for sp in ax1.spines.values(): sp.set_color("#333344")

        # 中圖：回撤
        ax2 = fig.add_subplot(gs[1, col])
        for key, color, lbl in [("dca", C["dca"], "DCA"), ("bah", C["bah"], "B&H")]:
            s  = r[key]
            dd = (s - s.cummax()) / s.cummax() * 100
            ax2.fill_between(dd.index, dd, alpha=0.30, color=color)
            ax2.plot(dd.index, dd, color=color, lw=1.2, label=lbl)
        ax2.set_title("水位回撤 (%)", color="white", fontsize=11, pad=8)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax2.legend(fontsize=8, facecolor="#1e2030", labelcolor="white", framealpha=0.8)
        ax2.set_facecolor("#1e2030")
        ax2.tick_params(colors="gray")
        for sp in ax2.spines.values(): sp.set_color("#333344")

    # 下方：績效表格
    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_facecolor("#1e2030")
    ax3.axis("off")

    all_stats = []
    for ticker, name in tickers.items():
        for s in results[ticker]["stats"]:
            all_stats.append({"標的": name, **s})

    cols = ["標的", "策略", "期間", "總投入", "最終價值", "總報酬率", "年化報酬率", "最大回撤"]
    rows = [[s[c] for c in cols] for s in all_stats]
    tbl  = ax3.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.2)

    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_facecolor("#2a2d3e" if ri % 2 == 0 else "#1e2030")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#333344")
        if ri == 0:
            cell.set_facecolor("#3a3d5e")
            cell.set_text_props(color="#aabbff", fontweight="bold")

    ax3.set_title("績效摘要", color="white", fontsize=12, pad=12)
    fig.suptitle(
        f"0050 vs QQQ｜定期定額 DCA vs 買入持有 B&H 回測\n每月投入 {MONTHLY_DCA:,}",
        color="white", fontsize=14, fontweight="bold", y=0.99
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  📊 圖表：{output_path}")


def export_excel(results: dict, tickers: dict, output_path: str):
    """輸出 Excel：摘要頁 + 每標的月資料"""
    all_stats = []
    for ticker, name in tickers.items():
        for s in results[ticker]["stats"]:
            all_stats.append({"標的": name, **s})

    cols = ["標的", "策略", "期間", "總投入", "最終價值", "總報酬率", "年化報酬率", "最大回撤"]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(all_stats)[cols].to_excel(writer, sheet_name="績效摘要", index=False)

        for ticker, name in tickers.items():
            r     = results[ticker]
            short = ticker.replace(".TW", "")
            df_m  = pd.DataFrame({
                "日期":          r["monthly_prices"].index.strftime("%Y-%m"),
                "月收盤價":      r["monthly_prices"].values.round(2),
                "DCA累積資產":   r["dca"].values.round(0),
                "DCA累計成本":   r["cost_dca"].values.round(0),
                "DCA未實現損益": (r["dca"] - r["cost_dca"]).values.round(0),
                "BaH累積資產":   r["bah"].values.round(0),
                "BaH未實現損益": (r["bah"] - r["cost_bah"]).values.round(0),
            })
            df_m.to_excel(writer, sheet_name=f"{short}_月資料", index=False)

    print(f"  📋 Excel：{output_path}")


# ── 主程式 ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  0050 vs QQQ  DCA vs Buy & Hold 回測")
    print(f"  每月定期定額：{MONTHLY_DCA:,}")
    print("=" * 55)

    print("\n[1/3] 下載資料...")
    data = download_data(TICKERS)

    print("\n[2/3] 執行回測...")
    results = {}
    for ticker in TICKERS:
        results[ticker] = run_backtest(data[ticker]["Close"], MONTHLY_DCA)
        for s in results[ticker]["stats"]:
            print(f"  {TICKERS[ticker]} {s['策略']}: "
                  f"年化 {s['年化報酬率']}  總報酬 {s['總報酬率']}  最大回撤 {s['最大回撤']}")

    print("\n[3/3] 輸出結果...")
    plot_results(results, TICKERS,
                 os.path.join(OUTPUT_DIR, "backtest_chart.png"))
    export_excel(results, TICKERS,
                 os.path.join(OUTPUT_DIR, "backtest_results.xlsx"))

    print("\n✅ 完成！請查看：")
    print("   → backtest_chart.png  （比較圖表）")
    print("   → backtest_results.xlsx  （每月詳細資料）")
