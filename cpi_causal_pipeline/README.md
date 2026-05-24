# 台灣 CPI 因果分析工作流程

一條把「行政院主計總處消費者物價指數（CPI）」從**抓取 → 預測 → 因果探索 → 視覺化 → 一頁式 HTML 報告**完整串接起來的自動化管線。

本專案由原本散落在 `main.ipynb` 的多個區塊重構而成，目標是**人類友善**與**最大化自定義**：所有可調參數集中在一個檔案、每個階段都能單獨開關、缺少套件會優雅跳過而非整條崩潰。

---

## 一、專案結構

```
cpi_causal_pipeline/
├── config.py          ★ 全域設定檔（唯一需要修改的地方，所有參數都在這）
├── tools.py             可重用工具箱（抓資料、解析、繪圖、轉檔、訓練模型）
├── causal_engine.py     因果探索引擎（DoWhy + PC 演算法 + EMD），由原命令列工具重構
├── build_report.py      把所有有效資料彙整成單一 index.html（可獨立執行，也被主流程呼叫）
├── pipeline.py        ★ 主流程：依序串接所有階段
├── requirements.txt     套件需求清單（分核心／選用）
├── README.md            本說明
├── assets/
│   └── TaipeiSansTCBeta-Regular.ttf   中文字體（繪圖防亂碼）
│
├── data.json            （內附）API 原始資料，可離線立即執行
├── cpi_data.csv / .xlsx （內附）整理後的 CPI 寬表
├── Filteredlist.csv     （內附）強相關配對清單
├── results/             （內附）因果引擎的有效產出（PC／估計／反駁／因果圖）
│
└── report/index.html  ★ 最終報告（執行後產生，單檔可離線開啟）
```

> 內附的資料檔讓你**不需連網、不需安裝重型套件**就能立刻跑出報告。

---

## 二、快速開始

```bash
# 1. 安裝核心套件（最少可運作集合）
pip install pandas numpy scipy matplotlib openpyxl requests scikit-learn statsmodels

# 2. 執行主流程
python pipeline.py

# 3. 打開報告
#    用瀏覽器開啟 report/index.html 即可
```

想完整重現因果探索階段（DoWhy／PC／EMD），再安裝選用套件：

```bash
pip install -r requirements.txt
# pygraphviz 需要系統套件，Ubuntu/Debian：
sudo apt-get install -y graphviz libgraphviz-dev pkg-config && pip install pygraphviz
```

---

## 三、流程的十個階段

| # | 階段名稱 | 做什麼 | 主要產出 |
|---|----------|--------|----------|
| 1 | `fetch_cpi` | 從主計總處 API 抓 CPI | `data.json` |
| 2 | `parse_and_plot` | 解析成寬表並畫總覽折線圖 | `cpi_data.csv`、`cpi_plot.png` |
| 3 | `arima_forecast` | ARIMA 長期預測（5 年） | `arima_forecast_*.{png,csv}` |
| 4 | `causal_impact` | CausalImpact 介入分析（預設關閉） | `report/causal_impact_report.txt` |
| 5 | `csv_to_xlsx` | 轉成 Excel 供因果引擎讀入 | `cpi_data.xlsx` |
| 6 | `causal_engine` | DoWhy + PC + EMD 因果探索 | `results/01~06_*.{csv,png}` |
| 7 | `filter_estimation` | 篩出強相關配對清單 | `Filteredlist.csv` |
| 8 | `correlated_plots` | 強相關配對三聯圖（**真實資料**） | `correlated_plots/*.png` |
| 9 | `ml_regression` | 多模型回歸比較（**真實資料**） | `ml_plots/*.png`、`ml_results.csv` |
| 10 | `build_report` | 彙整所有有效資料成一頁 HTML | `report/index.html` |

---

## 四、人類友善的設計

- **缺套件不崩潰**：沒裝 `statsmodels` → 跳過 ARIMA；沒裝 `dowhy` 等 → 因果引擎自動沿用 `results/` 既有有效結果。核心階段（解析、報告）才會在失敗時中止。
- **離線可跑**：API 連不上時，自動沿用內附的 `data.json`。
- **清楚的進度與摘要**：終端與 `logs/` 同時記錄每個階段的成功／跳過／失敗。
- **單檔報告**：`index.html` 內嵌所有圖片，可直接寄出或離線開啟。

---

## 五、最大化自定義

**所有調整都在 `config.py`，不需碰邏輯程式碼。** 幾個常見例子：

```python
# 只跑某幾段（其餘設 False）
stages = { "fetch_cpi": False, "arima_forecast": True, "build_report": True, ... }

# 換 ARIMA 預測目標與階數
arima_target = "一.食物類"
arima_order  = (2, 1, 2)
arima_forecast_steps = 36          # 改成預測 3 年

# 機器學習：換目標、換特徵（多特徵會自動略過 Response Plot）
ml_target   = "總指數"
ml_features = ["一.食物類", "三.居住類"]   # 多特徵
ml_models_enabled = ["Ensemble (隨機森林)", "SVM (向量機模型)"]  # 只比這兩個

# 強相關篩選門檻
filter_metric_col = "LR_Effect"
filter_threshold  = 1.0

# 因果引擎參數（透傳給 causal_engine）
ce_max_imf = 3
ce_ci_tests = "pearsonr,chi_square"
ce_iv_refute_band = "0.01,0.05"
```

也可用命令列臨時覆寫階段，不必改檔：

```bash
python pipeline.py --only build_report                 # 只重建報告
python pipeline.py --skip fetch_cpi arima_forecast     # 跳過抓資料與預測
```

---

## 六、關於「模擬資料 → 真實資料」的修正

原 `main.ipynb` 有兩個區塊使用 `numpy` 隨機／合成資料當佔位，未與真實 CPI 串接：

- **強相關配對視覺化**（原 Cell 32）：曾用 `np.linspace + np.random.normal` 生成假的 60 點序列。
  → 現於 `tools.build_correlated_plots()` 改讀 `cpi_data.csv` 的**真實**時間序列（1999–2026 共 328 期）。
- **機器學習回歸比較**（原 Cell 34）：曾用 `0.5x + 10·sin(x/5) + noise` 生成假的 X／y。
  → 現於 `tools.run_ml_regression()` 改用**真實** CPI 欄位（預設：以「一.食物類」預測「總指數」），並可在 `config.py` 自由更換特徵與目標。

兩處皆已驗證可正確產圖且中文不亂碼。
