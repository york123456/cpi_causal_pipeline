# -*- coding: utf-8 -*-
"""
================================================================================
 tools.py  ─  可重用工具箱（管線的「動詞」）
================================================================================

這裡收納所有「做一件事」的函式：抓資料、解析、繪圖、轉檔、訓練模型……
pipeline.py 只負責「按順序呼叫」這些工具，邏輯則全部集中在這裡，
方便單獨測試、替換或在其他專案重用。

重點修正（對照原 main.ipynb）：
  ▸ build_correlated_plots()  ── 原 Cell 32 使用 np.random 模擬資料，
    現已切回 cpi_data.csv 的「真實」時間序列。
  ▸ run_ml_regression()       ── 原 Cell 34 使用 np.linspace+sin 模擬資料，
    現已切回以真實 CPI 欄位作為特徵/目標。

所有函式都接受明確參數、回傳明確結果，並使用模組層級 logger 記錄進度，
對人類與自動化都友善。
================================================================================
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# matplotlib 採非互動後端，確保在無視窗環境（伺服器 / CI）也能存圖。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


logger = logging.getLogger("cpi_pipeline")


# =========================================================================== #
# 0. 基礎建設：日誌 與 中文字體
# =========================================================================== #
def setup_logging(logs_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """建立同時輸出到檔案與螢幕的 logger（UTF-8，含時間與函式名）。"""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | [%(funcName)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("日誌系統就緒，記錄檔：%s", log_file)
    return logger


def setup_chinese_font(font_path: Path, font_family: str,
                       download_url: Optional[str] = None) -> bool:
    """
    設定 matplotlib 中文字體，避免圖表出現「□□□」亂碼。
    策略：先用本機 assets 字體；找不到時嘗試掃描系統 CJK 字體；
    （在可連網環境）也可由 download_url 取得，但離線時不強制。
    回傳 True 表示已成功掛上某種可顯示中文的字體。
    """
    
        
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    import os
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    
    # 假設您把字體檔 'NotoSansTC-Regular.ttf' 放專案根目錄下
    font_path = os.path.join(os.path.dirname(__file__), "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    
    if os.path.exists(font_path):
        # 1. 動態註冊字體
        fe = fm.FontEntry(fname=font_path, name="MyActionFont")
        fm.fontManager.ttflist.insert(0, fe)
        # 2. 全域指定
        plt.rcParams["font.family"] = "MyActionFont"
        print("成功載入專案字體！")
        logger.info("已載入內附中文字體")
        return True
    else:
        print("找不到字體檔案，將使用預設字體")
        logger.warning("找不到任何中文字體，圖表中文可能顯示為方塊。"
                   "可將 .ttf 放入 assets/ 或安裝系統 CJK 字體。")
        return False
    
    
    '''
    # 1) 優先使用內附字體
    if Path(font_path).exists():
        try:
            fm.fontManager.addfont(str(font_path))
            matplotlib.rc("font", family=font_family)
            matplotlib.rcParams["axes.unicode_minus"] = False
            logger.info("已載入內附中文字體：%s", font_path)
            return True
        except Exception as e:  # pragma: no cover
            logger.warning("載入內附字體失敗（%s），改掃描系統字體。", e)

    # 2) 退而求其次：掃描系統中已安裝的 CJK 字體
    cjk_keys = ["Hei", "Ming", "Song", "Kai", "JhengHei",
                "YaHei", "Noto Sans CJK", "WenQuanYi", "Source Han"]
    found = [f.name for f in fm.fontManager.ttflist
             if any(k in f.name for k in cjk_keys)]
    if found:
        plt.rcParams["font.sans-serif"] = found + ["sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        logger.info("使用系統 CJK 字體：%s", found[0])
        return True

    logger.warning("找不到任何中文字體，圖表中文可能顯示為方塊。"
                   "可將 .ttf 放入 assets/ 或安裝系統 CJK 字體。")
    return False
    '''

# =========================================================================== #
# 1. 取得 CPI 原始資料（主計總處 API）
# =========================================================================== #
def fetch_cpi_json(api_url: str, out_json: Path,
                   allow_offline: bool = True) -> Path:
    """
    從 DGBAS API 抓取 CPI 資料並存成 data.json。
    若連線失敗且 allow_offline=True 且既有 data.json 存在，則沿用既有檔。
    """
    try:
        import requests
        logger.info("向主計總處 API 請求資料 …")
        resp = requests.get(api_url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        out_json.write_text(json.dumps(data, ensure_ascii=False, indent=4),
                            encoding="utf-8")
        logger.info("API 取得成功，已寫入 %s", out_json)
        return out_json
    except Exception as e:
        if allow_offline and out_json.exists():
            logger.warning("API 取得失敗（%s）→ 沿用既有 %s", e, out_json)
            return out_json
        raise RuntimeError(f"無法取得 CPI 資料且無離線備援：{e}") from e


# =========================================================================== #
# 2. 解析 data.json → cpi_data.csv，並繪製總覽折線圖
# =========================================================================== #
def parse_cpi_json_to_csv(data_json: Path, out_csv: Path) -> pd.DataFrame:
    """把主計總處 SDMX-JSON 結構攤平成寬表（index=年月，columns=分類）。"""
    data = json.loads(Path(data_json).read_text(encoding="utf-8"))
    structure = data["data"]["structure"]

    times = [tp["id"] for tp in
             structure["dimensions"]["observation"][0]["values"]]
    series_names = [s["name"] for s in
                    structure["dimensions"]["series"][0]["values"]]
    series_data = data["data"]["dataSets"][0]["series"]

    df = pd.DataFrame(index=times)
    for s_idx_str, obs in series_data.items():
        s_idx = int(s_idx_str)
        name = series_names[s_idx]
        values = [obs["observations"].get(str(i), [None])[0]
                  for i in range(len(times))]
        df[name] = values

    df.to_csv(out_csv)
    logger.info("已解析 %d 期 × %d 類，寫入 %s", df.shape[0], df.shape[1], out_csv)
    return df


def plot_cpi_overview(cpi_csv: Path, out_png: Path,
                      title: str = "消費者物價基本分類指數") -> Path:
    """繪製所有分類的時間序列總覽折線圖。"""
    df = pd.read_csv(cpi_csv, index_col=0)
    times = list(df.index)

    plt.figure(figsize=(14, 7))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col, marker=".", markersize=4)
    plt.xticks(range(0, len(times), 6), times[::6], rotation=45)
    plt.title(title)
    plt.xlabel("年月 (Year-Month)")
    plt.ylabel("指數 (Index)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    logger.info("CPI 總覽圖已存：%s", out_png)
    return out_png


def load_cpi_timeseries(cpi_csv: Path) -> pd.DataFrame:
    """
    讀回 cpi_data.csv 為「真實」時間序列 DataFrame，index 轉成 datetime。
    這是修正模擬資料的關鍵：所有下游分析都從這裡取得真實資料。
    """
    df = pd.read_csv(cpi_csv, index_col=0)
    # index 形如 "1999-M1" → datetime
    try:
        df.index = pd.to_datetime(
            df.index.str.replace("-M", "-", regex=False), format="%Y-%m")
    except Exception:
        logger.warning("時間索引轉換失敗，保留原始字串索引。")
    return df


# =========================================================================== #
# 3. ARIMA 長期預測（需 statsmodels；缺套件則由 pipeline 跳過）
# =========================================================================== #
def run_arima_forecast(cpi_csv: Path, target: str, order: tuple,
                       steps: int, alpha: float,
                       out_png: Path, out_csv: Path) -> pd.DataFrame:
    """對指定目標欄位建立 ARIMA 模型並預測未來 steps 個月。"""
    from statsmodels.tsa.arima.model import ARIMA  # 延遲匯入

    df = load_cpi_timeseries(cpi_csv)
    df.index.freq = "MS"
    y = df[target].dropna()

    logger.info("訓練 ARIMA%s 於『%s』(%d 筆) …", order, target, len(y))
    result = ARIMA(y, order=order).fit()
    fc = result.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=alpha)

    plt.figure(figsize=(12, 6))
    plt.plot(y.index, y, label=f"歷史 {target}", color="blue")
    plt.plot(mean.index, mean, label="ARIMA 預測值", color="red")
    plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                     color="red", alpha=0.2,
                     label=f"{int((1-alpha)*100)}% 信賴區間")
    plt.title(f"{target} - ARIMA 長期預測（{steps//12} 年）")
    plt.xlabel("日期"); plt.ylabel("指數"); plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

    out = pd.DataFrame({"Forecast": mean,
                        "Lower_CI": ci.iloc[:, 0],
                        "Upper_CI": ci.iloc[:, 1]})
    out.to_csv(out_csv)
    logger.info("ARIMA 完成：%s / %s", out_png, out_csv)
    return out


# =========================================================================== #
# 4. CSV → XLSX
# =========================================================================== #
def csv_to_xlsx(csv_path: Path, xlsx_path: Path) -> Path:
    """將 cpi_data.csv 轉成 Excel（供因果引擎以 Excel 讀入）。"""
    from openpyxl import Workbook
    import csv as _csv

    wb = Workbook()
    ws = wb.active
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in _csv.reader(f):
            ws.append(row)
    wb.save(xlsx_path)
    logger.info("已轉存 Excel：%s", xlsx_path)
    return xlsx_path


# =========================================================================== #
# 5. 由 DoWhy 估計結果篩出強相關清單 → Filteredlist.csv
# =========================================================================== #
def filter_estimation_to_list(source_csv: Path, out_csv: Path,
                              metric_col: str = "LR_Effect",
                              threshold: float = 1.0) -> List[list]:
    """
    讀 04_*_DoWhy_Estimation.csv，保留 metric_col >= threshold 的邊，
    輸出 [Source, Target, IV_Effect, LR_Effect] 清單到 Filteredlist.csv。
    （對應原 Cell 28 / 29，並修掉 BOM 欄名的脆弱寫法。）
    """
    df = pd.read_csv(source_csv)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]  # 去除 BOM

    keep = df[pd.to_numeric(df[metric_col], errors="coerce") >= threshold]
    rows = keep[["Source", "Target", "IV_Effect", "LR_Effect"]].values.tolist()

    pd.DataFrame(rows).to_csv(out_csv, header=False, index=False)
    logger.info("篩選 %s >= %s：保留 %d / %d 條邊 → %s",
                metric_col, threshold, len(rows), len(df), out_csv)
    return rows


# =========================================================================== #
# 6. 【已修正】強相關配對視覺化 ── 改用真實 CPI 資料
#     （原 Cell 32 以 np.random 生成模擬 df_ts，此處切回真實時間序列）
# =========================================================================== #
def build_correlated_plots(filtered_csv: Path, cpi_csv: Path,
                           out_dir: Path, pct_change: bool = True,
                           dpi: int = 150) -> List[Path]:
    """
    讀 Filteredlist.csv 的強相關配對，對每一對畫出三聯圖：
      (1) 原始趨勢並列(雙Y軸)  (2) 期增率並列  (3) 趨勢+變化雙 Y 軸綜合比較
    資料來源為 cpi_data.csv 的「真實」序列（已修正模擬資料）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(filtered_csv, header=None,
                        names=["Cat1", "Cat2", "Correlation", "Distance"])

    # ★ 真實資料：直接讀 CPI 時間序列（不再用 np.random 模擬）★
    df_ts = load_cpi_timeseries(cpi_csv)

    # 確認配對中的類別都存在於真實資料欄位中
    needed = set(pairs["Cat1"]).union(pairs["Cat2"])
    missing = needed - set(df_ts.columns)
    if missing:
        logger.warning("以下類別不在 CPI 真實資料中，相關配對將略過：%s", missing)

    df_change = (df_ts.pct_change().dropna() * 100) if pct_change else None
    saved: List[Path] = []

    for _, row in pairs.iterrows():
        c1, c2, corr = row["Cat1"], row["Cat2"], row["Correlation"]
        if c1 not in df_ts.columns or c2 not in df_ts.columns:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(12, 16))
        fig.suptitle(f"強相關分析（真實資料）：{c1} vs {c2}\n(相關係數: {corr:.4f})",
                     fontsize=18, fontweight="bold")

        # =========================================================
        # (1) 原始趨勢（雙 Y 軸視覺正規化）
        # =========================================================
        ax0_l = axes[0]
        ax0_r = axes[0].twinx()
        
        l1_trend = ax0_l.plot(df_ts.index, df_ts[c1], label=f"{c1} (左軸)",
                              color="#1f77b4", linewidth=2.5)
        l2_trend = ax0_r.plot(df_ts.index, df_ts[c2], label=f"{c2} (右軸)",
                              color="#ff7f0e", linewidth=2.5)
        
        ax0_l.set_title("趨勢並列（雙 Y 軸視覺對齊）", fontsize=14)
        ax0_l.set_ylabel(f"{c1} 指數", color="#1f77b4", fontweight="bold")
        ax0_r.set_ylabel(f"{c2} 指數", color="#ff7f0e", fontweight="bold")
        ax0_l.tick_params(axis='y', labelcolor="#1f77b4")
        ax0_r.tick_params(axis='y', labelcolor="#ff7f0e")

        lines_trend = l1_trend + l2_trend
        ax0_l.legend(lines_trend, [l.get_label() for l in lines_trend], loc="upper left")
        ax0_l.grid(True, linestyle="--", alpha=0.6)

        # =========================================================
        # (2) 期增率
        # =========================================================
        if df_change is not None:
            axes[1].plot(df_change.index, df_change[c1], label=f"{c1} (變化率)",
                         color="#1f77b4", linestyle="--", linewidth=2)
            axes[1].plot(df_change.index, df_change[c2], label=f"{c2} (變化率)",
                         color="#ff7f0e", linestyle="--", linewidth=2)
        axes[1].set_title("變化並列（期增率 %）", fontsize=14)
        axes[1].set_ylabel("變化率 (%)")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, linestyle="--", alpha=0.6)

        # =========================================================
        # (3) 雙 Y 軸綜合 (原始邏輯)
        # =========================================================
        ax_l = axes[2]; ax_r = axes[2].twinx()
        l1 = ax_l.plot(df_ts.index, df_ts[c1], label=f"{c1} (趨勢)",
                       color="#1f77b4", linewidth=2)
        l2 = ax_l.plot(df_ts.index, df_ts[c2], label=f"{c2} (趨勢)",
                       color="#ff7f0e", linewidth=2)
        lines = l1 + l2
        if df_change is not None:
            l3 = ax_r.plot(df_change.index, df_change[c1], label=f"{c1} (變化率)",
                           color="#2ca02c", linestyle=":", linewidth=2)
            l4 = ax_r.plot(df_change.index, df_change[c2], label=f"{c2} (變化率)",
                           color="#d62728", linestyle=":", linewidth=2)
            lines += l3 + l4
        axes[2].set_title("趨勢與變化綜合比較", fontsize=14)
        ax_l.set_ylabel("指數（左軸）"); ax_r.set_ylabel("變化率 %（右軸）")
        ax_l.legend(lines, [l.get_label() for l in lines],
                    loc="upper left", bbox_to_anchor=(1.05, 1))
        ax_l.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        safe1 = c1.replace(".", "_").replace("/", "_")
        safe2 = c2.replace(".", "_").replace("/", "_")
        fp = out_dir / f"plot_{safe1}_vs_{safe2}.png"
        plt.savefig(fp, dpi=dpi, bbox_inches="tight")
        plt.close()
        saved.append(fp)

    logger.info("強相關配對圖（真實資料）完成 %d 張 → %s", len(saved), out_dir)
    return saved

'''
def build_correlated_plots(filtered_csv: Path, cpi_csv: Path,
                           out_dir: Path, pct_change: bool = True,
                           dpi: int = 150) -> List[Path]:
    """
    讀 Filteredlist.csv 的強相關配對，對每一對畫出三聯圖：
      (1) 原始趨勢並列  (2) 期增率並列  (3) 趨勢+變化雙 Y 軸綜合比較
    資料來源為 cpi_data.csv 的「真實」序列（已修正模擬資料）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(filtered_csv, header=None,
                        names=["Cat1", "Cat2", "Correlation", "Distance"])

    # ★ 真實資料：直接讀 CPI 時間序列（不再用 np.random 模擬）★
    df_ts = load_cpi_timeseries(cpi_csv)

    # 確認配對中的類別都存在於真實資料欄位中
    needed = set(pairs["Cat1"]).union(pairs["Cat2"])
    missing = needed - set(df_ts.columns)
    if missing:
        logger.warning("以下類別不在 CPI 真實資料中，相關配對將略過：%s", missing)

    df_change = (df_ts.pct_change().dropna() * 100) if pct_change else None
    saved: List[Path] = []

    for _, row in pairs.iterrows():
        c1, c2, corr = row["Cat1"], row["Cat2"], row["Correlation"]
        if c1 not in df_ts.columns or c2 not in df_ts.columns:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(12, 16))
        fig.suptitle(f"強相關分析（真實資料）：{c1} vs {c2}\n(相關係數: {corr:.4f})",
                     fontsize=18, fontweight="bold")

        # (1) 原始趨勢
        axes[0].plot(df_ts.index, df_ts[c1], label=f"{c1} (趨勢)",
                     color="#1f77b4", linewidth=2.5)
        axes[0].plot(df_ts.index, df_ts[c2], label=f"{c2} (趨勢)",
                     color="#ff7f0e", linewidth=2.5)
        axes[0].set_title("趨勢並列（原始指數）", fontsize=14)
        axes[0].set_ylabel("指數"); axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle="--", alpha=0.6)

        # (2) 期增率
        if df_change is not None:
            axes[1].plot(df_change.index, df_change[c1], label=f"{c1} (變化率)",
                         color="#1f77b4", linestyle="--", linewidth=2)
            axes[1].plot(df_change.index, df_change[c2], label=f"{c2} (變化率)",
                         color="#ff7f0e", linestyle="--", linewidth=2)
        axes[1].set_title("變化並列（期增率 %）", fontsize=14)
        axes[1].set_ylabel("變化率 (%)"); axes[1].legend(loc="upper left")
        axes[1].grid(True, linestyle="--", alpha=0.6)

        # (3) 雙 Y 軸綜合
        ax_l = axes[2]; ax_r = axes[2].twinx()
        l1 = ax_l.plot(df_ts.index, df_ts[c1], label=f"{c1} (趨勢)",
                       color="#1f77b4", linewidth=2)
        l2 = ax_l.plot(df_ts.index, df_ts[c2], label=f"{c2} (趨勢)",
                       color="#ff7f0e", linewidth=2)
        lines = l1 + l2
        if df_change is not None:
            l3 = ax_r.plot(df_change.index, df_change[c1], label=f"{c1} (變化率)",
                           color="#2ca02c", linestyle=":", linewidth=2)
            l4 = ax_r.plot(df_change.index, df_change[c2], label=f"{c2} (變化率)",
                           color="#d62728", linestyle=":", linewidth=2)
            lines += l3 + l4
        axes[2].set_title("趨勢與變化綜合比較", fontsize=14)
        ax_l.set_ylabel("指數（左軸）"); ax_r.set_ylabel("變化率 %（右軸）")
        ax_l.legend(lines, [l.get_label() for l in lines],
                    loc="upper left", bbox_to_anchor=(1.05, 1))
        ax_l.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        safe1 = c1.replace(".", "_").replace("/", "_")
        safe2 = c2.replace(".", "_").replace("/", "_")
        fp = out_dir / f"plot_{safe1}_vs_{safe2}.png"
        plt.savefig(fp, dpi=dpi, bbox_inches="tight")
        plt.close()
        saved.append(fp)

    logger.info("強相關配對圖（真實資料）完成 %d 張 → %s", len(saved), out_dir)
    return saved
'''

# =========================================================================== #
# 7. 【已修正】多模型回歸比較 ── 改用真實 CPI 資料
#     （原 Cell 34 以 linspace+sin+noise 生成模擬 X/y，此處切回真實欄位）
# =========================================================================== #
def _build_models(enabled: Sequence[str]) -> Dict[str, object]:
    """依設定建立模型字典（只建立 enabled 清單內的模型）。"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.kernel_ridge import KernelRidge

    catalog = {
        "GPR (高斯過程回歸)": GaussianProcessRegressor(
            kernel=C(1.0) * RBF(10), n_restarts_optimizer=10, normalize_y=True),
        "NN (神經網絡模型)": MLPRegressor(
            hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42),
        "SVM (向量機模型)": SVR(kernel="rbf", C=100, gamma=0.1),
        "Tree (樹模型)": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Ensemble (隨機森林)": RandomForestRegressor(
            n_estimators=100, random_state=42),
        "Ensemble (梯度提升)": GradientBoostingRegressor(
            n_estimators=100, random_state=42),
        "Kernel (核嶺回歸)": KernelRidge(alpha=0.1, kernel="rbf", gamma=0.1),
    }
    return {k: v for k, v in catalog.items() if k in enabled}


def run_ml_regression(cpi_csv: Path, target: str, features: List[str],
                      out_dir: Path, models_enabled: Sequence[str],
                      test_size: float = 0.2, random_state: int = 42,
                      dpi: int = 150) -> pd.DataFrame:
    """
    用真實 CPI 欄位訓練並比較多種回歸模型。
      X = features（一個或多個 CPI 分類欄位）
      y = target （預測目標，預設「總指數」）
    產出：模型比較長條圖、預測vs實際、Bland-Altman；
          單一特徵時額外畫 Response Plot。
    回傳：依 R² 排序的評估結果 DataFrame（同時寫成 ml_results.csv）。
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (mean_squared_error, r2_score,
                                  mean_absolute_error)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ★ 真實資料：以 CPI 欄位作為特徵與目標（不再用模擬 sin 曲線）★
    df = load_cpi_timeseries(cpi_csv).dropna()
    for col in features + [target]:
        if col not in df.columns:
            raise ValueError(f"欄位『{col}』不存在於 CPI 資料；可選欄位：{list(df.columns)}")

    X = df[features].values
    y = df[target].values
    single_feature = (len(features) == 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    models = _build_models(models_enabled)
    if not models:
        raise ValueError("未啟用任何模型，請檢查 config.ml_models_enabled。")

    results, predictions = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "R2 Score": r2_score(y_test, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "MAE": mean_absolute_error(y_test, y_pred),
        })
        predictions[name] = y_pred

    df_res = pd.DataFrame(results).sort_values("R2 Score")
    best = df_res.iloc[-1]["Model"]
    logger.info("最佳模型：%s（目標=%s，特徵=%s）", best, target, features)
    df_res.sort_values("R2 Score", ascending=False).to_csv(
        out_dir / "ml_results.csv", index=False, encoding="utf-8-sig")

    # (1) R² 比較長條圖
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_res["Model"], df_res["R2 Score"], color="skyblue")
    plt.xlabel("R² Score（越接近 1 越好）")
    plt.title(f"模型擬合測試結果比較（目標：{target}）")
    for b in bars:
        plt.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                 f"{b.get_width():.4f}", va="center")
    plt.tight_layout(); plt.savefig(out_dir / "01_Compare_Bar.png", dpi=dpi)
    plt.close()

    y_best = predictions[best]

    # (2) 預測 vs 實際
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_best, alpha=0.7, color="purple", edgecolors="k")
    lo, hi = min(y_test.min(), y_best.min()), max(y_test.max(), y_best.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=2, label="完美預測 (y=x)")
    plt.xlabel("實際值"); plt.ylabel("預測值")
    plt.title(f"Predicted vs Actual（{best}）"); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "02_Pred_vs_Actual.png", dpi=dpi)
    plt.close()

    # (3) Bland-Altman
    mean_vals = (y_test + y_best) / 2
    diff_vals = y_best - y_test
    md, sd = float(np.mean(diff_vals)), float(np.std(diff_vals))
    plt.figure(figsize=(10, 6))
    plt.scatter(mean_vals, diff_vals, alpha=0.7, color="teal", edgecolors="k")
    plt.axhline(md, color="red", label=f"Mean Diff ({md:.2f})")
    plt.axhline(md + 1.96 * sd, color="blue", linestyle="--", label="+1.96 SD")
    plt.axhline(md - 1.96 * sd, color="blue", linestyle="--", label="-1.96 SD")
    plt.xlabel("平均值"); plt.ylabel("差異（預測 - 實際）")
    plt.title(f"Bland-Altman（{best}）"); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "03_Bland_Altman.png", dpi=dpi)
    plt.close()

    # (4) Response Plot —— 僅單一特徵時可畫（高維無法以單軸呈現）
    if single_feature:
        plt.figure(figsize=(12, 7))
        plt.scatter(X, y, color="lightgray", label="全部實際資料")
        plt.scatter(X_test, y_test, color="black", label="測試資料")
        xr = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
        plt.plot(xr, models[best].predict(xr), color="red", linewidth=3,
                 label=f"{best} 擬合曲線")
        plt.xlabel(f"輸入特徵：{features[0]}"); plt.ylabel(f"目標：{target}")
        plt.title("Response Plot（特徵 → 預測曲線）"); plt.legend()
        plt.tight_layout(); plt.savefig(out_dir / "04_Response_Plot.png", dpi=dpi)
        plt.close()
    else:
        logger.info("多重特徵 (%d) → 略過 Response Plot。", len(features))

    logger.info("機器學習回歸（真實資料）完成 → %s", out_dir)
    return df_res.sort_values("R2 Score", ascending=False)
