"""Read-only table review projections, including old assets and broken references."""


def review_tables(run):
    fragments = {f["id"]: f for f in run.get("table_fragments", [])}
    if "table_fragments" not in run:
        for element in run["elements"]:
            if element["kind"] != "table":
                continue
            payload = element["payload"]
            fragments[element["id"]] = {"id": element["id"], "span": element["span"],
                "row_count": payload["row_count"], "column_count": payload["column_count"],
                "backend": "legacy", "structure_status": "unverified", "provenance_status": "unverified",
                "cells": [{**c, "spans": [c["span"]] if c.get("span") else [],
                           "state": c.get("state", "text" if c.get("raw_text") else "unknown")}
                          for c in payload["cells"]]}
    tables = {t["id"]: t for t in run.get("logical_tables", [])}
    assigned = {identifier for t in tables.values() for identifier in t["fragment_ids"]}
    for identifier in fragments:
        if identifier not in assigned:
            tables[identifier] = {"id": identifier, "fragment_ids": [identifier], "title": "页内表格"}
    return tables, fragments


def structure_rows(fragment, max_rows=200, max_columns=40):
    height, width = min(fragment["row_count"], max_rows), min(fragment["column_count"], max_columns)
    rows = [{"行": r + 1, **{f"列 {c + 1}": "未知位置" for c in range(width)}} for r in range(height)]
    for cell in fragment["cells"]:
        row, column = cell["row"], cell["column"]
        if not 0 <= row < height or not 0 <= column < width:
            continue
        state = cell.get("state", "unknown")
        text = cell.get("raw_text")
        rows[row][f"列 {column + 1}"] = text if text is not None and text != "" else {
            "blank": "（空白）", "covered": "↳ 合并占位", "unreadable": "（未识别）"}.get(state, "（未知）")
        if state == "covered":
            continue
        for r in range(row, min(height, row + (cell.get("row_span") or 1))):
            for c in range(column, min(width, column + (cell.get("col_span") or 1))):
                if (r, c) != (row, column):
                    rows[r][f"列 {c + 1}"] = f"↳ 第 {row + 1} 行第 {column + 1} 列"
    return rows


def select_table_evidence(run, table_id, fragment_id, cell_id=""):
    tables, fragments = review_tables(run)
    table = tables[table_id]
    if fragment_id not in table["fragment_ids"] or fragment_id not in fragments:
        raise ValueError("所选片段不属于逻辑表或已缺失")
    fragment = fragments[fragment_id]
    local_cells = {c["id"]: c for c in fragment["cells"]}
    all_cells = {c["id"]: c for i in table["fragment_ids"] if i in fragments for c in fragments[i]["cells"]}
    if not cell_id:
        return {"spans": [fragment["span"]], "support": [], "missing": [], "page_no": fragment["span"]["page_no"]}
    cell = local_cells[cell_id]
    if cell.get("state") == "covered":
        cell = local_cells.get(cell.get("covered_by_cell_id"), cell)
    requests = [(cell["id"], "value")]
    for name, role in (("row_header_cell_ids", "row_header"), ("column_header_cell_ids", "column_header"), ("unit_cell_ids", "unit")):
        requests.extend((identifier, role) for identifier in cell.get(name, []))
    support, missing, highlights = [], [], {}
    for identifier, role in requests:
        reference = all_cells.get(identifier)
        if reference is None:
            missing.append(identifier)
            continue
        support.append({"role": role, "cell_id": identifier, "text": reference.get("raw_text"), "spans": reference.get("spans", [])})
        for span in reference.get("spans", []):
            key = (span["page_no"], tuple(span["bbox"]))
            # A value always stays red; a unit may also be a column header.
            if key not in highlights or (role == "unit" and highlights[key]["highlight_role"] != "value"):
                highlights[key] = {**span, "highlight_role": role}
    for span in table.get("unit_spans", []):
        highlights.setdefault((span["page_no"], tuple(span["bbox"])), {**span, "highlight_role": "unit"})
    return {"spans": list(highlights.values()), "support": support, "missing": missing,
            "page_no": next((s["page_no"] for s in cell.get("spans", [])), fragment["span"]["page_no"])}
