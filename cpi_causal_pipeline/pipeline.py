# -*- coding: utf-8 -*-
"""
================================================================================
 pipeline.py  ─  主流程（Main Pipeline）
================================================================================

把整條工作流程「依序」串接起來。每個階段：
  ▸ 受 config.stages 開關控制（True 才執行）。
  ▸ 由 try/except 包覆 ── 「核心階段」失敗會中止；「選用階段」失敗只警告並續跑。
  ▸ 完成後在終端列印彩色摘要，清楚標示成功 / 跳過 / 失敗。

執行方式：
    python pipeline.py                 # 跑全部開啟的階段
    python pipeline.py --only build_report      # 只跑某一階段
    python pipeline.py --skip fetch_cpi arima_forecast   # 跳過某些階段

所有可調參數都在 config.py，本檔只負責「呼叫 tools / causal_engine / build_report」。
================================================================================
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from config import CONFIG
import tools
import causal_engine
from build_report import build_report


# 階段是否為「核心」：核心失敗 → 整條中止；選用失敗 → 警告續跑。
CORE_STAGES = {"parse_and_plot", "build_report"}


def _run_stage(name: str, fn, logger) -> str:
    """執行單一階段，回傳狀態字串：'ok' | 'skip' | 'fail'。"""
    if not CONFIG.stages.get(name, False):
        logger.info("⏭  跳過階段：%s（config 關閉）", name)
        return "skip"
    logger.info("▶  開始階段：%s", name)
    try:
        fn()
        logger.info("✔  完成階段：%s", name)
        return "ok"
    except Exception as e:
        if name in CORE_STAGES:
            logger.error("✖  核心階段失敗：%s → %s", name, e)
            logger.debug(traceback.format_exc())
            raise
        logger.warning("⚠  選用階段失敗（略過）：%s → %s", name, e)
        logger.debug(traceback.format_exc())
        return "fail"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="台灣 CPI 因果分析主流程（人類友善、高度可自定義）")
    parser.add_argument("--only", nargs="*", default=None,
                        help="只執行指定階段（其餘一律跳過）")
    parser.add_argument("--skip", nargs="*", default=None,
                        help="跳過指定階段")
    args = parser.parse_args(argv)

    # 依命令列調整階段開關（不改 config 檔本身）
    if args.only:
        for k in CONFIG.stages:
            CONFIG.stages[k] = (k in args.only)
    if args.skip:
        for k in args.skip:
            if k in CONFIG.stages:
                CONFIG.stages[k] = False

    CONFIG.ensure_dirs()
    logger = tools.setup_logging(CONFIG.logs_dir)
    tools.setup_chinese_font(CONFIG.font_path, CONFIG.font_family,
                             CONFIG.font_download_url)

    logger.info("=" * 64)
    logger.info(" 台灣 CPI 因果分析管線啟動")
    logger.info("=" * 64)

    status = {}

    # ---------------- 各階段定義（閉包，延後執行）---------------- #
    def s_fetch():
        tools.fetch_cpi_json(CONFIG.api_url(), CONFIG.data_json,
                             CONFIG.allow_offline_fallback)

    def s_parse():
        tools.parse_cpi_json_to_csv(CONFIG.data_json, CONFIG.cpi_csv)
        tools.plot_cpi_overview(CONFIG.cpi_csv, CONFIG.cpi_plot)

    def s_arima():
        tools.run_arima_forecast(
            CONFIG.cpi_csv, CONFIG.arima_target, CONFIG.arima_order,
            CONFIG.arima_forecast_steps, CONFIG.arima_ci_alpha,
            CONFIG.arima_plot, CONFIG.arima_csv)

    def s_causal_impact():
        # 選用、較重；需 pycausalimpact。預設於 config 關閉。
        import pandas as pd
        from causalimpact import CausalImpact
        df = tools.load_cpi_timeseries(CONFIG.cpi_csv)
        cols = [CONFIG.arima_target] + [c for c in CONFIG.ci_covariates
                                        if c in df.columns]
        data = df[cols]
        split = CONFIG.api_end_year - CONFIG.ci_intervention_split_years_before_end
        pre = [df.index.min(), pd.Timestamp(f"{split}-01-01")]
        post = [pd.Timestamp(f"{split}-02-01"), df.index.max()]
        ci = CausalImpact(data, pre, post)
        (Path(CONFIG.report_dir) / "causal_impact_report.txt").write_text(
            ci.summary(output="report"), encoding="utf-8")

    def s_xlsx():
        tools.csv_to_xlsx(CONFIG.cpi_csv, CONFIG.cpi_xlsx)

    def s_engine():
        ok = causal_engine.run(CONFIG)
        if not ok:
            raise RuntimeError("因果引擎無有效產出（缺套件且無既有 results）。")

    def s_filter():
        src = Path(CONFIG.filter_source_csv)
        if not src.is_absolute():
            src = CONFIG.workdir / src
        if not src.exists() and CONFIG.filter_reuse_existing \
                and CONFIG.filtered_csv.exists():
            logger.warning("估計檔不存在 → 沿用既有 %s", CONFIG.filtered_csv)
            return
        tools.filter_estimation_to_list(
            src, CONFIG.filtered_csv,
            CONFIG.filter_metric_col, CONFIG.filter_threshold)

    def s_pairs():
        tools.build_correlated_plots(
            CONFIG.filtered_csv, CONFIG.cpi_csv, CONFIG.plots_dir,
            CONFIG.plot_pct_change, CONFIG.plot_dpi)

    def s_ml():
        tools.run_ml_regression(
            CONFIG.cpi_csv, CONFIG.ml_target, CONFIG.ml_features,
            CONFIG.ml_dir, CONFIG.ml_models_enabled,
            CONFIG.ml_test_size, CONFIG.ml_random_state, CONFIG.plot_dpi)

    def s_report():
        build_report(CONFIG)

    # ---------------- 依序執行 ---------------- #
    plan = [
        ("fetch_cpi", s_fetch),
        ("parse_and_plot", s_parse),
        ("arima_forecast", s_arima),
        ("causal_impact", s_causal_impact),
        ("csv_to_xlsx", s_xlsx),
        ("causal_engine", s_engine),
        ("filter_estimation", s_filter),
        ("correlated_plots", s_pairs),
        ("ml_regression", s_ml),
        ("build_report", s_report),
    ]
    for name, fn in plan:
        status[name] = _run_stage(name, fn, logger)

    # ---------------- 摘要 ---------------- #
    logger.info("=" * 64)
    logger.info(" 執行摘要")
    icon = {"ok": "✔ 成功", "skip": "— 跳過", "fail": "✖ 失敗"}
    for name, _ in plan:
        logger.info("   %-18s %s", name, icon.get(status.get(name, "skip")))
    logger.info("=" * 64)
    report = Path(CONFIG.report_dir) / "index.html"
    if report.exists():
        logger.info("📄 報告位置：%s", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
