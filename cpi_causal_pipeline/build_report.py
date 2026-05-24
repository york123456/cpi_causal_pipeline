# -*- coding: utf-8 -*-
"""
================================================================================
 build_report.py  ─  將所有「有效資料」彙整為單一 index.html
================================================================================

把整條管線產出的每一份有效資料 ── CSV 表格、PNG 圖表 ── 收集起來，
組成一份「自帶內容、可離線開啟、可直接分享」的 index.html。

設計重點：
  ▸ 自我包含：所有圖片以 base64 內嵌，整份 HTML 單檔可攜，不依賴外部路徑。
  ▸ 容錯：哪個階段沒跑、哪個檔案不存在，就自動略過該區塊，不會壞掉。
  ▸ 人類友善：清楚的章節導覽、表格預覽（過長自動截斷）、中文說明。
  ▸ 可獨立執行：`python build_report.py`；亦可被 pipeline.py 直接呼叫。

用法：
  from build_report import build_report
  build_report(CONFIG)
================================================================================
"""

from __future__ import annotations

import base64
import html
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger("cpi_pipeline")

MAX_TABLE_ROWS = 30   # 表格預覽最多列數（避免報告過長）


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _img_tag(path: Path, alt: str = "") -> Optional[str]:
    """把圖片讀成 base64 並包成 <img>。檔案不存在則回 None。"""
    p = Path(path)
    if not p.exists():
        return None
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return (f'<img loading="lazy" alt="{html.escape(alt)}" '
            f'src="data:image/png;base64,{b64}">')

'''
def _csv_table(path: Path, max_rows: int = MAX_TABLE_ROWS,
               header: Optional[List[str]] = None) -> Optional[str]:
    """把 CSV 讀成 HTML 表格（截斷過長者並標註）。檔案不存在則回 None。"""
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = (pd.read_csv(p, header=None, names=header)
              if header else pd.read_csv(p))
    except Exception as e:
        logger.warning("讀取表格 %s 失敗：%s", p, e)
        return None

    total = len(df)
    truncated = total > max_rows
    view = df.head(max_rows)

    thead = "".join(f"<th>{html.escape(str(c))}</th>" for c in view.columns)
    rows = []
    for _, r in view.iterrows():
        tds = "".join(
            f"<td>{html.escape(f'{v:.4f}' if isinstance(v, float) else str(v))}</td>"
            for v in r)
        rows.append(f"<tr>{tds}</tr>")
    note = (f'<p class="note">＊僅顯示前 {max_rows} 列，共 {total} 列。完整資料見原始 CSV。</p>'
            if truncated else "")
    return (f'<div class="tablewrap"><table><thead><tr>{thead}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>{note}')
'''

def _csv_table(path: Path, max_rows: int = MAX_TABLE_ROWS,
               header: Optional[List[str]] = None,
               tail: bool = False,                # 新增：是否顯示倒數幾列
               sort_by: Optional[str] = None,     # 新增：排序的欄位名稱
               ascending: bool = True             # 新增：遞增(True)或遞減(False)排序
               ) -> Optional[str]:
    """把 CSV 讀成 HTML 表格（截斷過長者並標註，支援排序與顯示末尾）。檔案不存在則回 None。"""
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = (pd.read_csv(p, header=None, names=header)
              if header else pd.read_csv(p))
    except Exception as e:
        logger.warning("讀取表格 %s 失敗：%s", p, e)
        return None

    # --- 1. 處理排序 ---
    if sort_by:
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)
        else:
            logger.warning(f"找不到指定的排序欄位 '{sort_by}'，將維持原始順序。")

    total = len(df)
    truncated = total > max_rows

    # --- 2. 處理截斷 (Head vs Tail) ---
    if tail:
        view = df.tail(max_rows)
        row_pos_text = "後"
    else:
        view = df.head(max_rows)
        row_pos_text = "前"

    thead = "".join(f"<th>{html.escape(str(c))}</th>" for c in view.columns)
    rows = []
    for _, r in view.iterrows():
        tds = "".join(
            f"<td>{html.escape(f'{v:.4f}' if isinstance(v, float) else str(v))}</td>"
            for v in r)
        rows.append(f"<tr>{tds}</tr>")
        
    note = (f'<p class="note">＊僅顯示{row_pos_text} {max_rows} 列，共 {total} 列。完整資料見原始 CSV。</p>'
            if truncated else "")
    
    return (f'<div class="tablewrap"><table><thead><tr>{thead}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>{note}')




def _section(num: str, sid: str, title: str, body_parts: List[str],
             desc: str = "") -> Optional[str]:
    """組裝一個章節；若 body 全空則回傳 None（整段略過）。"""
    body = [b for b in body_parts if b]
    if not body:
        return None
    desc_html = f'<p class="desc">{desc}</p>' if desc else ""
    return (f'<section id="{sid}"><div class="sec-head">'
            f'<span class="sec-num">{num}</span>'
            f'<h2>{html.escape(title)}</h2></div>{desc_html}'
            f'{"".join(body)}</section>')


def _gallery(images: List[str], captions: List[str]) -> str:
    cells = []
    for img, cap in zip(images, captions):
        cells.append(f'<figure>{img}<figcaption>{html.escape(cap)}</figcaption></figure>')
    return f'<div class="gallery">{"".join(cells)}</div>'


# --------------------------------------------------------------------------- #
# 主函式
# --------------------------------------------------------------------------- #
def build_report(cfg) -> Path:
    """掃描所有有效產出並輸出 report/index.html，回傳其路徑。"""
    report_dir = Path(cfg.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    out_html = report_dir / "index.html"

    sections: List[str] = []
    nav: List[tuple] = []   # (sid, 顯示名)

    def add(sec, sid, name):
        if sec:
            sections.append(sec)
            nav.append((sid, name))

    # ---- 1. CPI 總覽 ----
    add(_section(
        "01", "overview", "CPI 資料總覽",
        [_img_tag(cfg.cpi_plot, "CPI 折線圖"),
         _csv_table(cfg.cpi_csv, tail=True)],
        "行政院主計總處消費者物價基本分類指數，涵蓋總指數與七大分類的長期月資料。"
    ), "overview", "資料總覽")

    # ---- 2. ARIMA 預測 ----
    add(_section(
        "02", "arima", "ARIMA 長期預測",
        [_img_tag(cfg.arima_plot, "ARIMA 預測圖"),
         _csv_table(cfg.arima_csv)],
        f"以 ARIMA 模型對「{cfg.arima_target}」進行長期外推預測，含信賴區間。"
    ), "arima", "長期預測")

    # ---- 3. 因果探索結果（results/ 內的 PC/估計/反駁/因果圖）----
    res = Path(cfg.results_dir)
    causal_parts: List[str] = []
    # 因果圖
    graph_imgs, graph_caps = [], []
    for png in sorted(res.glob("06_*_Causal_Graph.png")):
        tag = _img_tag(png, png.stem)
        if tag:
            graph_imgs.append(tag)
            graph_caps.append(png.stem.replace("06_", "").replace("_", " "))
    if graph_imgs:
        causal_parts.append("<h3>因果關係圖</h3>")
        causal_parts.append(_gallery(graph_imgs, graph_caps))
    # 估計表（原始層）
    est = res / "04_Original_Data_DoWhy_Estimation.csv"
    est_tbl = _csv_table(est)
    if est_tbl:
        causal_parts.append("<h3>因果效應估計（原始層）</h3>")
        causal_parts.append(est_tbl)
    add(_section(
        "03", "causal", "因果探索（DoWhy + PC + EMD）", causal_parts,
        "透過 EMD 多層分解、PC 演算法找出有向邊，再以 DoWhy 進行 IV／線性回歸"
        "效應估計與穩健性反駁檢定。"
    ), "causal", "因果探索")

    # ---- 4. 強相關清單 ----
    add(_section(
        "04", "filtered", "強相關配對清單",
        [_csv_table(cfg.filtered_csv,
                    header=["類別 1", "類別 2", "相關係數", "效應/距離"])],
        f"由因果估計結果中，篩選出「{cfg.filter_metric_col} ≥ {cfg.filter_threshold}」"
        "的顯著配對。"
    ), "filtered", "強相關清單")

    # ---- 5. 強相關配對視覺化（真實資料）----
    plot_imgs, plot_caps = [], []
    for png in sorted(Path(cfg.plots_dir).glob("*.png")):
        tag = _img_tag(png, png.stem)
        if tag:
            plot_imgs.append(tag)
            plot_caps.append(png.stem.replace("plot_", "").replace("_", " "))
    add(_section(
        "05", "pairs", "強相關配對視覺化（真實資料）",
        [_gallery(plot_imgs, plot_caps)] if plot_imgs else [],
        "每組強相關類別的三聯圖：原始趨勢、期增率、雙軸綜合比較。"
    ), "pairs", "配對視覺化")

    # ---- 6. 機器學習回歸比較（真實資料）----
    ml_imgs, ml_caps = [], []
    for fn in ("01_Compare_Bar.png", "02_Pred_vs_Actual.png",
               "03_Bland_Altman.png", "04_Response_Plot.png"):
        tag = _img_tag(Path(cfg.ml_dir) / fn, fn)
        if tag:
            ml_imgs.append(tag)
            ml_caps.append(fn.replace(".png", "").replace("_", " "))
    ml_tbl = _csv_table(Path(cfg.ml_dir) / "ml_results.csv")
    ml_body = ([ml_tbl] if ml_tbl else []) + \
              ([_gallery(ml_imgs, ml_caps)] if ml_imgs else [])
    add(_section(
        "06", "ml", "機器學習回歸模型比較（真實資料）", ml_body,
        f"以真實 CPI 欄位（特徵：{ '、'.join(cfg.ml_features) }；目標：{cfg.ml_target}）"
        "比較多種回歸模型的擬合表現。"
    ), "ml", "模型比較")

    # ---- 組裝 HTML ----
    nav_html = "".join(
        f'<a href="#{sid}">{html.escape(name)}</a>' for sid, name in nav)
    body_html = "".join(sections)
    generated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    page = _HTML_TEMPLATE.format(
        title=html.escape(cfg.report_title),
        subtitle=html.escape(cfg.report_subtitle),
        nav=nav_html,
        body=body_html,
        generated=generated,
    )
    out_html.write_text(page, encoding="utf-8")
    logger.info("HTML 報告已產生：%s（%d 個章節）", out_html, len(nav))
    return out_html


# --------------------------------------------------------------------------- #
# HTML / CSS 模板（精煉編輯式學術風格，淺色、襯線標題、銳利強調色）
# --------------------------------------------------------------------------- #
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --ink:#1a1a1a; --paper:#f7f5f0; --card:#ffffff;
    --accent:#9b2226; --accent2:#005f73; --line:#e0ddd5; --muted:#6b6b6b;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--paper); color:var(--ink);
    font-family:"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif;
    line-height:1.7; -webkit-font-smoothing:antialiased;
  }}
  header.hero {{
    padding:64px 28px 40px; border-bottom:3px solid var(--ink);
    background:linear-gradient(180deg,#fffdf8,var(--paper));
  }}
  .hero .kicker {{ letter-spacing:.35em; font-size:.72rem; color:var(--accent);
    text-transform:uppercase; margin-bottom:14px; }}
  .hero h1 {{
    font-family:"Georgia","Songti TC",serif; font-weight:700;
    font-size:clamp(1.8rem,4.5vw,3.1rem); line-height:1.15; margin:0 0 12px;
    max-width:18ch;
  }}
  .hero p {{ color:var(--muted); margin:0; max-width:60ch; }}
  .hero .meta {{ margin-top:18px; font-size:.8rem; color:var(--muted); }}

  nav {{ position:sticky; top:0; z-index:10; display:flex; flex-wrap:wrap;
    gap:4px; padding:10px 20px; background:rgba(247,245,240,.92);
    backdrop-filter:blur(8px); border-bottom:1px solid var(--line); }}
  nav a {{ font-size:.82rem; color:var(--ink); text-decoration:none;
    padding:5px 12px; border:1px solid var(--line); border-radius:999px;
    transition:.18s; }}
  nav a:hover {{ background:var(--ink); color:var(--paper); border-color:var(--ink); }}

  main {{ max-width:1080px; margin:0 auto; padding:8px 24px 80px; }}
  section {{ padding:44px 0; border-bottom:1px solid var(--line); }}
  .sec-head {{ display:flex; align-items:baseline; gap:16px; margin-bottom:6px; }}
  .sec-num {{ font-family:"Georgia",serif; font-size:1.1rem; color:var(--accent);
    font-weight:700; }}
  section h2 {{ font-family:"Georgia","Songti TC",serif; font-size:1.7rem;
    margin:0; }}
  section h3 {{ font-size:1.05rem; margin:30px 0 12px; color:var(--accent2); }}
  .desc {{ color:var(--muted); margin:6px 0 22px; max-width:70ch; }}
  .note {{ color:var(--muted); font-size:.8rem; margin:8px 0 0; }}

  figure {{ margin:0; background:var(--card); border:1px solid var(--line);
    border-radius:10px; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,.04); }}
  figure img {{ width:100%; height:auto; display:block; }}
  figcaption {{ padding:10px 14px; font-size:.82rem; color:var(--muted);
    border-top:1px solid var(--line); }}
  section > img {{ width:100%; border:1px solid var(--line); border-radius:10px;
    background:var(--card); }}

  .gallery {{ display:grid; gap:18px;
    grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }}

  .tablewrap {{ overflow:auto; border:1px solid var(--line); border-radius:10px;
    background:var(--card); margin:8px 0; }}
  table {{ border-collapse:collapse; width:100%; font-size:.86rem; }}
  thead th {{ background:var(--ink); color:var(--paper); text-align:left;
    padding:9px 12px; position:sticky; top:0; white-space:nowrap; }}
  tbody td {{ padding:8px 12px; border-top:1px solid var(--line);
    white-space:nowrap; }}
  tbody tr:nth-child(even) {{ background:#fbfaf6; }}

  footer {{ text-align:center; color:var(--muted); font-size:.8rem;
    padding:30px; }}
</style>
</head>
<body>
  <header class="hero">
    <div class="kicker">Causal Analysis Report</div>
    <h1>{title}</h1>
    <p>{subtitle}</p>
    <div class="meta">報告產生時間：{generated}　·　本報告所有圖表均內嵌於單一檔案，可離線開啟與分享。</div>
  </header>
  <nav>{nav}</nav>
  <main>
    {body}
  </main>
  <footer>由 CPI 因果分析管線自動生成　·　所有資料源自真實有效輸出</footer>
</body>
</html>"""


if __name__ == "__main__":
    # 獨立執行：直接用預設 config 產報告
    import logging as _lg
    _lg.basicConfig(level=_lg.INFO, format="%(levelname)s %(message)s")
    from config import CONFIG
    path = build_report(CONFIG)
    print(f"已輸出：{path}")
