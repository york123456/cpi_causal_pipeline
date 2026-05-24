# -*- coding: utf-8 -*-
"""
================================================================================
 causal_engine.py  ─  因果探索引擎（DoWhy + PC 演算法 + EMD 經驗模態分解）
================================================================================

由原 DoWhy_Command_2026.py 重構而來，差異在於：
  ▸ 改為「可匯入、可被 pipeline 呼叫」的 run(cfg) 介面（不再依賴命令列）。
  ▸ 參數全部來自 config.py，可在一處統一調整。
  ▸ 優雅降級：偵測到缺少重型套件（dowhy / pgmpy / PyEMD / pygraphviz）時，
    若既有 results/ 已有有效產出，會自動沿用而不中斷整條管線。

演算流程（與原版相同）：
  1. 讀取 Excel 並前處理（內插補值）            → results/01,02_*.csv
  2. EMD 將每個變數分解為多層 IMF
  3. 每一層跑 PC 演算法找出有向邊                → results/03_*.csv
  4. 對每條邊以 DoWhy 做 IV / 線性回歸估計 + 反駁 → results/04,05_*.csv
  5. 繪製因果圖                                  → results/06_*.png
================================================================================
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger("cpi_pipeline")


# --------------------------------------------------------------------------- #
# 相依套件偵測：決定是否能實際執行因果計算
# --------------------------------------------------------------------------- #
def _heavy_libs_available() -> bool:
    import importlib
    for m in ("dowhy", "pgmpy", "PyEMD"):
        if importlib.util.find_spec(m) is None:
            return False
    return True


def _existing_results_present(results_dir: Path) -> bool:
    """既有 results/ 是否已含足以支撐後續流程的有效產出。"""
    needed = results_dir / "04_Original_Data_DoWhy_Estimation.csv"
    return needed.exists()


# --------------------------------------------------------------------------- #
# 1. 資料前處理
# --------------------------------------------------------------------------- #
def read_and_preprocess(file_path, results_dir: Path):
    logger.info("因果引擎：讀取 %s", file_path)
    try:
        df_raw = pd.read_excel(file_path, engine="calamine")
    except Exception:
        df_raw = pd.read_excel(file_path)
    df_raw.to_csv(results_dir / "01_Original_Data.csv",
                  index=False, encoding="utf-8-sig")

    numeric_df = df_raw.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        raise ValueError("找不到任何數值欄位，請檢查 Excel 內容。")
    for col in numeric_df.columns:
        if numeric_df[col].isnull().sum() > 0:
            numeric_df[col] = numeric_df[col].interpolate(method="spline", order=3)
            numeric_df[col] = numeric_df[col].bfill().ffill()
    numeric_df.to_csv(results_dir / "02_Preprocessed_Data.csv",
                      index=False, encoding="utf-8-sig")
    logger.info("前處理完成，保留 %d 欄。", numeric_df.shape[1])
    return numeric_df


# --------------------------------------------------------------------------- #
# 2. EMD 訊號分解
# --------------------------------------------------------------------------- #
def _compute_emd_for_var(args):
    from PyEMD import EMD
    var, values, max_imf = args
    return var, EMD().emd(values, max_imf=max_imf)


def perform_imf_decomposition(data, max_imf):
    logger.info("EMD 分解（最大層數 %d）…", max_imf)
    var_names = data.columns.tolist()
    imf_layers = [data.copy()]
    tasks = [(v, data[v].values, max_imf) for v in var_names]
    var_imf = {}
    with ProcessPoolExecutor() as ex:
        for v, imfs in ex.map(_compute_emd_for_var, tasks):
            var_imf[v] = imfs
    for idx in range(max_imf):
        layer = {v: var_imf[v][idx] for v in var_names if var_imf[v].shape[0] > idx}
        if layer:
            imf_layers.append(pd.DataFrame(layer))
    logger.info("EMD 完成，共 %d 層有效 IMF。", len(imf_layers) - 1)
    return imf_layers


def check_is_i_map(model):
    try:
        for ind in model.get_independencies().get_assertions():
            X = list(getattr(ind, "event1", []))
            Y = list(getattr(ind, "event2", []))
            Z = list(getattr(ind, "given", [])) if hasattr(ind, "given") else []
            for x in X:
                for y in Y:
                    if model.is_dconnected(x, y, observed=Z):
                        return False
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# 3. 單邊因果估計 + 反駁
# --------------------------------------------------------------------------- #
def _evaluate_single_edge(u, v, imf_data, args):
    from dowhy import CausalModel
    r = {k: None for k in (
        "effect", "linear_effect",
        "refute_p_iv_random_common_cause", "refute_sig_iv_random_common_cause",
        "refute_p_lr_random_common_cause", "refute_sig_lr_random_common_cause",
        "refute_p_iv_data_subset_refuter", "refute_sig_iv_data_subset_refuter",
        "refute_p_lr_data_subset_refuter", "refute_sig_lr_data_subset_refuter",
        "error")}
    try:
        instruments = [z for z in imf_data.columns if z not in (u, v)]
        model = CausalModel(data=imf_data, treatment=u, outcome=v,
                            instruments=instruments)
        est = model.identify_effect()
        e_iv = model.estimate_effect(
            est, method_name="iv.instrumental_variable",
            method_params={"iv_instrument_name": instruments})
        r["effect"] = e_iv.value
        e_lr = model.estimate_effect(est, method_name="backdoor.linear_regression")
        r["linear_effect"] = e_lr.value

        refute_th = [float(t.strip()) for t in args.refute_thresholds.split(",")]
        for method in ("random_common_cause", "data_subset_refuter"):
            try:
                ref_iv = model.refute_estimate(est, e_iv, method_name=method)
                p_iv = ref_iv.refutation_result["p_value"]
                r[f"refute_p_iv_{method}"] = p_iv
                if args.enable_refute_filter:
                    if args.iv_refute_mode == "abs_bandpass":
                        lo, hi = map(float, args.iv_refute_band.split(","))
                        r[f"refute_sig_iv_{method}"] = lo <= np.abs(p_iv) <= hi
                    else:
                        r[f"refute_sig_iv_{method}"] = p_iv > refute_th[0]
                else:
                    r[f"refute_sig_iv_{method}"] = True

                ref_lr = model.refute_estimate(est, e_lr, method_name=method)
                p_lr = ref_lr.refutation_result["p_value"]
                r[f"refute_p_lr_{method}"] = p_lr
                r[f"refute_sig_lr_{method}"] = (
                    (p_lr > refute_th[1]) if args.enable_refute_filter else True)
            except Exception:
                pass
    except Exception as e:
        r["error"] = str(e)
    return (u, v), r


def run_causal_analysis_and_export(layer_name, imf_data, args, results_dir: Path):
    from pgmpy.estimators import PC
    G_edges = {}
    ci_tests = [t.strip() for t in args.ci_tests.split(",")]
    ci_th = [float(t.strip()) for t in args.ci_thresholds.split(",")]

    logger.info("[%s] PC 演算法 …", layer_name)
    pc_records = []
    for ci_test, th in zip(ci_tests, ci_th):
        eff_th = th if args.enable_ci_filter else 1.0
        dag = PC(imf_data).estimate(
            variant="stable", ci_test=ci_test, significance_level=eff_th,
            max_cond_vars=args.pc_max_cond_vars, show_progress=False)
        if not check_is_i_map(dag):
            continue
        for u, v in dag.edges():
            G_edges.setdefault((u, v), {"pc": set()})["pc"].add(ci_test)
            pc_records.append({"Source": u, "Target": v, "Passed_CI_Test": ci_test})
    if pc_records:
        pd.DataFrame(pc_records).to_csv(
            results_dir / f"03_{layer_name}_PC_Detection.csv",
            index=False, encoding="utf-8-sig")

    logger.info("[%s] DoWhy 估計與反駁 …", layer_name)
    with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) + 4)) as ex:
        futs = {ex.submit(_evaluate_single_edge, u, v, imf_data, args): (u, v)
                for (u, v) in G_edges}
        est_rec, ref_rec = [], []
        for fut in as_completed(futs):
            edge, r = fut.result()
            G_edges[edge].update(r)
            u, v = edge
            est_rec.append({"Source": u, "Target": v,
                            "IV_Effect": r.get("effect"),
                            "LR_Effect": r.get("linear_effect")})
            ref_rec.append({
                "Source": u, "Target": v,
                "IV_RandomCause_P": r.get("refute_p_iv_random_common_cause"),
                "IV_RandomCause_Sig": r.get("refute_sig_iv_random_common_cause"),
                "IV_DataSubset_P": r.get("refute_p_iv_data_subset_refuter"),
                "IV_DataSubset_Sig": r.get("refute_sig_iv_data_subset_refuter"),
                "LR_RandomCause_P": r.get("refute_p_lr_random_common_cause"),
                "LR_RandomCause_Sig": r.get("refute_sig_lr_random_common_cause"),
                "LR_DataSubset_P": r.get("refute_p_lr_data_subset_refuter"),
                "LR_DataSubset_Sig": r.get("refute_sig_lr_data_subset_refuter"),
            })
    if est_rec:
        pd.DataFrame(est_rec).to_csv(
            results_dir / f"04_{layer_name}_DoWhy_Estimation.csv",
            index=False, encoding="utf-8-sig")
    if ref_rec:
        pd.DataFrame(ref_rec).to_csv(
            results_dir / f"05_{layer_name}_DoWhy_Refutation.csv",
            index=False, encoding="utf-8-sig")
    return G_edges


# --------------------------------------------------------------------------- #
# 4. 因果圖繪製
# --------------------------------------------------------------------------- #
def draw_causal_graph(G_edges, var_names, output_path, font_family="Noto Sans CJK TC"):
    import pygraphviz as pgv
    A = pgv.AGraph(directed=True)
    for node in var_names:
        A.add_node(node, shape="ellipse", style="filled",
                   fillcolor="skyblue", fontsize=14, fontname=font_family)
    for (u, v), attr in G_edges.items():
        if attr.get("effect") is None:
            continue
        pc_tests = ", ".join(sorted(attr.get("pc", [])))
        label = f"PC: {pc_tests}\nIV: {attr['effect']:.2f}, LR: {attr.get('linear_effect', 0):.2f}"
        robust = (attr.get("refute_sig_iv_random_common_cause")
                  or attr.get("refute_sig_iv_data_subset_refuter"))
        A.add_edge(u, v, color="green" if robust else "gray",
                   penwidth=str(2 + 4 * min(abs(attr["effect"]), 1)),
                   label=label, fontsize=10, fontname=font_family)
    A.layout(prog="dot")
    A.draw(output_path)


# --------------------------------------------------------------------------- #
# 對外主介面：被 pipeline 呼叫
# --------------------------------------------------------------------------- #
def run(cfg) -> bool:
    """
    執行因果引擎。回傳 True 表示「有可用的 results/ 產出」（無論是本次計算
    或沿用既有）。回傳 False 表示既無法計算、也無既有結果。
    """
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not _heavy_libs_available():
        if cfg.ce_reuse_existing_results and _existing_results_present(results_dir):
            logger.warning("缺少 dowhy/pgmpy/PyEMD 等套件 → 沿用既有 results/ 有效產出。")
            return True
        logger.error("缺少因果分析重型套件，且無既有 results/ 可用。"
                     "請依 requirements.txt 安裝後重跑此階段。")
        return False

    # 把 config 的 ce_* 參數打包成原演算法期望的 args 物件
    args = SimpleNamespace(
        max_imf=cfg.ce_max_imf,
        pc_max_cond_vars=cfg.ce_pc_max_cond_vars,
        ci_tests=cfg.ce_ci_tests,
        ci_thresholds=cfg.ce_ci_thresholds,
        enable_ci_filter=cfg.ce_enable_ci_filter,
        refute_thresholds=cfg.ce_refute_thresholds,
        enable_refute_filter=cfg.ce_enable_refute_filter,
        iv_refute_mode=cfg.ce_iv_refute_mode,
        iv_refute_band=cfg.ce_iv_refute_band,
    )

    t0 = time.time()
    data = read_and_preprocess(cfg.cpi_xlsx, results_dir)
    imf_layers = perform_imf_decomposition(data, args.max_imf)
    for i, imf_data in enumerate(imf_layers):
        layer = "Original_Data" if i == 0 else f"IMF_{i}"
        logger.info("====== 層級：%s ======", layer)
        edges = run_causal_analysis_and_export(layer, imf_data, args, results_dir)
        try:
            draw_causal_graph(edges, imf_data.columns.tolist(),
                              str(results_dir / f"06_{layer}_Causal_Graph.png"),
                              font_family=cfg.font_family)
        except Exception as e:
            logger.warning("繪製 %s 因果圖失敗（%s），略過。", layer, e)
    logger.info("因果引擎完成，耗時 %.1f 秒。", time.time() - t0)
    return True
