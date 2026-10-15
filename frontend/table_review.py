"""Logical table, physical grid and cell support controls for the quality page."""

import streamlit as st

from app.pdf.tables.review import review_tables, structure_rows, select_table_evidence
from app.quality.tables import table_usage

USAGE = {"blocked": "禁止用于数值问答", "reference_only": "仅供原文参考，数值需复核", "traceable": "规则通过，可追溯原文（不代表识别准确率）"}
ROLES = {"value": "所选值", "row_header": "行标识", "column_header": "列表头", "unit": "单位"}


def render_table_review(run, report, run_id, reviews=()):
    tables, fragments = review_tables(run)
    if not tables:
        return None
    summaries = {t.id: t for t in report.tables}
    policies = {t.id: table_usage(report, t, reviews) for t in report.tables}
    st.write("**表格结构与单元格复核**")
    st.dataframe([{"表格": t.get("title") or identifier, "物理页": "、".join(str(fragments[i]["span"]["page_no"]) for i in t["fragment_ids"] if i in fragments),
                   "使用状态": USAGE[policies[identifier]["usage"]] if identifier in policies else "尚无表格级报告，请更新自动检查"}
                  for identifier, t in tables.items()], hide_index=True, width="stretch")
    table_id = st.selectbox("定位逻辑表", ["", *tables], format_func=lambda i:
                           f"{tables[i].get('title') or '表格'} · {i}" if i else "不选择表格", key=f"quality_table_{run_id}")
    if not table_id:
        return None
    table = tables[table_id]
    summary = summaries.get(table_id)
    if summary:
        st.caption("表格使用状态：" + USAGE[policies[table_id]["usage"]])
    ids = [i for i in table["fragment_ids"] if i in fragments]
    if len(ids) != len(table["fragment_ids"]):
        st.error("逻辑表引用了缺失片段，当前仅展示仍可读取的部分。")
    if not ids:
        return None
    fragment_id = st.selectbox("物理表格片段", ids, format_func=lambda i:
        f"第 {fragments[i]['span']['page_no']} 页 · {i}", key=f"quality_fragment_{run_id}_{table_id}")
    fragment = fragments[fragment_id]
    st.caption(f"后端：{fragment.get('backend', 'unknown')} · 结构：{fragment.get('structure_status', 'unverified')} · 来源：{fragment.get('provenance_status', 'unverified')}")
    st.dataframe(structure_rows(fragment), hide_index=True, width="stretch")
    if fragment["row_count"] > 200 or fragment["column_count"] > 40:
        st.caption("结构预览仅显示前 200 行、40 列；单元格选择仍可定位完整记录。")
    local_cells = {c["id"]: c for c in fragment["cells"]}
    cell_id = st.selectbox("定位表格单元格", ["", *local_cells], format_func=lambda i:
        f"第 {local_cells[i]['row'] + 1} 行第 {local_cells[i]['column'] + 1} 列 · {str(local_cells[i].get('raw_text') or '（无文字）')[:80]}" if i else "整个物理表格",
        key=f"quality_cell_{run_id}_{fragment_id}")
    selection = select_table_evidence(run, table_id, fragment_id, cell_id)
    if cell_id:
        cell = local_cells[cell_id]
        st.caption(f"状态：{cell.get('state', 'unknown')} · 跨行：{cell.get('row_span') or '未知'} · 跨列：{cell.get('col_span') or '未知'} · 文字来源：{cell.get('text_origin', 'unknown')}")
        st.dataframe([{"角色": ROLES[s["role"]], "原文": s["text"], "物理页": "、".join(str(p["page_no"]) for p in s["spans"])}
                      for s in selection["support"]], hide_index=True, width="stretch")
        st.caption("原页高亮：红色＝所选值，蓝色＝行标识，绿色＝列表头，紫色＝单位；缺少位置的证据不绘制。")
        if selection["missing"]:
            st.warning("部分支持单元格引用已失效，不能据此确认数值归属。")
    if table.get("unit_text"):
        st.write("表级单位声明：" + table["unit_text"])
    for note in table.get("footnotes", []):
        st.text("表级注释：" + note)
    with st.expander("续表与来源详情"):
        st.json({"片段": table["fragment_ids"], "续表判定": fragment.get("continuation_decision", "未记录"),
                 "待确认续接": table.get("continuation_candidate_ids", []), "重复表头来源": table.get("repeated_header_cell_ids", [])})
    return {**selection, "table_id": table_id, "fragment_id": fragment_id, "cell_id": cell_id,
            "reviewable": summary is not None, "cell_reviewable": bool(summary and cell_id in summary.cell_ids)}
