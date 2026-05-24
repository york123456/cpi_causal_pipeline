# -*- coding: utf-8 -*-
"""
================================================================================
 config.py  ─  全域設定檔（唯一需要修改的地方）
================================================================================

這個檔案把整條管線的所有「可調參數」集中在一處。
你不需要動到任何邏輯程式碼，只要在這裡改值，整條流程就會跟著改變。

設計原則：
  1. 人類友善 ── 每個參數都有中文說明，標示單位與合理範圍。
  2. 最大化自定義 ── 每個階段都能單獨開關 (STAGES)，所有門檻、欄位、
     模型、輸出路徑都可在此覆寫。
  3. 不需改程式 ── 想換目標欄位、換模型、換 API 區間？改這裡就好。

使用方式：
  from config import CONFIG
  CONFIG.CPI_CSV          # 取用設定
================================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# --------------------------------------------------------------------------- #
# 根目錄：所有產出檔案都會落在 WORKDIR 底下，方便集中管理 / 打包 / 清除。
# --------------------------------------------------------------------------- #
WORKDIR = Path(__file__).resolve().parent


@dataclass
class Config:
    # ====================================================================== #
    # 0. 全域路徑設定
    # ====================================================================== #
    workdir: Path = WORKDIR
    assets_dir: Path = WORKDIR / "assets"          # 內附資源（字體等）
    results_dir: Path = WORKDIR / "results"        # 因果分析階段性產出
    logs_dir: Path = WORKDIR / "logs"              # 執行日誌
    plots_dir: Path = WORKDIR / "correlated_plots" # 強相關配對圖
    ml_dir: Path = WORKDIR / "ml_plots"            # 機器學習比較圖
    report_dir: Path = WORKDIR / "report"          # 最終 index.html 報告

    # 主要中介檔（沿用原 notebook 命名，方便對照）
    data_json: Path = WORKDIR / "data.json"
    cpi_csv: Path = WORKDIR / "cpi_data.csv"
    cpi_xlsx: Path = WORKDIR / "cpi_data.xlsx"
    cpi_plot: Path = WORKDIR / "cpi_plot.png"
    arima_plot: Path = WORKDIR / "arima_forecast_plot.png"
    arima_csv: Path = WORKDIR / "arima_forecast_data.csv"
    filtered_csv: Path = WORKDIR / "Filteredlist.csv"

    # 中文字體（繪圖防亂碼）。先找本機 assets，再考慮線上下載。
    font_path: Path = WORKDIR / "assets" / "TaipeiSansTCBeta-Regular.ttf"
    #font_family: str = "Taipei Sans TC Beta"
    font_family: str = "MyActionFont"
    font_download_url: str = (
        "https://drive.google.com/uc?id=1eGAsTN1HBpJAkeVM57_C7ccp7hbgSz3_&export=download"
    )

    # ====================================================================== #
    # 1. 階段開關 ── 想只跑某幾段？把不要的設成 False 即可。
    #    管線會「依序」執行 True 的階段，並在缺少相依套件時優雅跳過。
    # ====================================================================== #
    stages: Dict[str, bool] = field(default_factory=lambda: {
        "fetch_cpi":         True,   # 1. 從主計總處 API 抓 CPI → data.json
        "parse_and_plot":    True,   # 2. data.json → cpi_data.csv + 折線圖
        "arima_forecast":    True,   # 3. ARIMA 長期預測（需 statsmodels）
        "causal_impact":     True,  # 4. CausalImpact 介入分析（需 pycausalimpact，較重，預設關）
        "csv_to_xlsx":       True,   # 5. cpi_data.csv → cpi_data.xlsx
        "causal_engine":     True,   # 6. DoWhy + PC + EMD 因果探索（缺套件則沿用既有 results）
        "filter_estimation": True,   # 7. 由估計結果篩出強相關清單 → Filteredlist.csv
        "correlated_plots":  True,   # 8. 強相關配對視覺化（已改用真實資料）
        "ml_regression":     True,   # 9. 多模型回歸比較（已改用真實資料）
        "build_report":      True,   # 10. 彙整所有有效資料 → index.html
    })

    # ====================================================================== #
    # 2. CPI 資料來源（主計總處 DGBAS 開放 API）
    # ====================================================================== #

    from datetime import datetime
    
    now = datetime.now()
    
    # 依序放入年、月、日、時、分、秒
    time_list = [now.year, now.month, now.day, now.hour, now.minute, now.second]
    #print(time_list)
    
    #print(time_list[0]) #year
    #print(time_list[1]) #month
    month_now = time_list[1]
    # 輸出範例: [2026, 5, 24, 18, 47, 30]

    api_start_year: int = 1999
    api_end_year: int = time_list[0]
    api_end_month: int = month_now - 1 if month_now - 1 > 0 else 12          # 尚未發布的月份，API 端設為 0
    # API 模板：{start} / {end} / {month} 會被自動填入
    api_url_template: str = (
        "https://nstatdb.dgbas.gov.tw/dgbasall/webMain.aspx?"
        "sdmx/A030101015/1+2+29+36+52+62+66+75...M."
        "&startTime={start}&endTime={end}-M{month}"
    )
    # 若 API 無法連線（離線環境），是否允許沿用既有的 data.json / cpi_data.csv
    allow_offline_fallback: bool = True

    # ====================================================================== #
    # 3. ARIMA 預測設定
    # ====================================================================== #
    arima_target: str = "總指數"     # 預測目標欄位
    arima_order: tuple = (1, 1, 1)   # (p, d, q)
    arima_forecast_steps: int = 60   # 預測未來月數（60 = 5 年）
    arima_ci_alpha: float = 0.05     # 信賴區間 → 95%

    # ====================================================================== #
    # 4. CausalImpact 介入分析設定（若開啟 stage 4）
    # ====================================================================== #
    ci_covariates: List[str] = field(default_factory=lambda: [
        "一.食物類", "二.衣著類", "三.居住類", "四.交通及通訊類",
        "五.醫藥保健類", "六.教養娛樂類", "七.雜項類",
    ])
    ci_intervention_split_years_before_end: int = 5  # 介入點 = 結束年 - 5

    # ====================================================================== #
    # 5. 因果引擎（DoWhy / PC / EMD）設定 ── 透傳給 causal_engine.py
    #    所有參數皆可調，對應原 DoWhy_Command_2026.py 的 argparse。
    # ====================================================================== #
    ce_max_imf: int = 3                       # EMD 分解層數
    ce_pc_max_cond_vars: int = 3              # PC 演算法最大條件變數
    ce_ci_tests: str = "pearsonr"
    ce_ci_thresholds: str = "1.0"
    ce_enable_ci_filter: bool = False
    ce_refute_thresholds: str = "0.05,1"   # (IV, LR) 反駁顯著性門檻
    ce_enable_refute_filter: bool = True
    ce_iv_refute_mode: str = "abs_bandpass"   # 'abs_bandpass' | 'normal'
    ce_iv_refute_band: str = "0.05,100"      # 帶通濾波 (lower, upper)
    # 缺少 dowhy/pgmpy/PyEMD/pygraphviz 等重型套件時，是否沿用 results/ 既有產出
    ce_reuse_existing_results: bool = True

    # ====================================================================== #
    # 6. 強相關清單篩選（stage 7）
    # ====================================================================== #
    # 來源：results/04_Original_Data_DoWhy_Estimation.csv
    filter_source_csv: str = "results/04_Original_Data_DoWhy_Estimation.csv"
    filter_metric_col: str = "LR_Effect"   # 用哪個欄位當門檻
    filter_threshold: float = 1.0          # >= 此值才保留
    # 若估計檔不存在（沒跑因果引擎），是否沿用既有的 Filteredlist.csv
    filter_reuse_existing: bool = True

    # ====================================================================== #
    # 7. 強相關配對視覺化（stage 8）── 已修正為真實資料
    # ====================================================================== #
    # 直接讀 cpi_data.csv 的真實時間序列，欄位名稱對應 Filteredlist 的類別。
    plot_pct_change: bool = True           # 是否計算期增率 (%)
    plot_dpi: int = 150

    # ====================================================================== #
    # 8. 機器學習回歸比較（stage 9）── 已修正為真實資料
    # ====================================================================== #
    # 任務定義：用 ml_features 預測 ml_target。
    #   - 單一特徵 (len==1)：會額外畫出 Response Plot（特徵→預測曲線）。
    #   - 多重特徵 (len>1) ：自動略過 Response Plot（高維無法畫單軸曲線）。
    ml_target: str = "總指數"
    ml_features: List[str] = field(default_factory=lambda: ["一.食物類"])
    ml_test_size: float = 0.2
    ml_random_state: int = 42
    # 想關掉哪個模型，把它從這份清單刪掉；想加參數，改 ml_model_overrides。
    ml_models_enabled: List[str] = field(default_factory=lambda: [
        "GPR (高斯過程回歸)",
        "NN (神經網絡模型)",
        "SVM (向量機模型)",
        "Tree (樹模型)",
        "Ensemble (隨機森林)",
        "Ensemble (梯度提升)",
        "Kernel (核嶺回歸)",
    ])

    # ====================================================================== #
    # 9. HTML 報告（stage 10）
    # ====================================================================== #
    report_title: str = "台灣消費者物價指數（CPI）因果分析報告"
    report_subtitle: str = "資料來源：行政院主計總處 DGBAS 開放資料平台"
    report_open_after_build: bool = False  # 產生後是否自動開啟瀏覽器

    # ====================================================================== #
    # 便利方法
    # ====================================================================== #
    def ensure_dirs(self) -> None:
        """建立所有輸出資料夾（若不存在）。"""
        for d in (self.results_dir, self.logs_dir, self.plots_dir,
                  self.ml_dir, self.report_dir):
            d.mkdir(parents=True, exist_ok=True)

    def api_url(self) -> str:
        return self.api_url_template.format(
            start=self.api_start_year,
            end=self.api_end_year,
            month=self.api_end_month,
        )


# 全域單例：其他模組統一 `from config import CONFIG`
CONFIG = Config()
