"""原始表格单元格的只读索引，兼容旧父对象内嵌证据。"""

RELATION_ROLES = (("row_header_cell_ids", "row_header"),
                  ("column_header_cell_ids", "column_header"), ("unit_cell_ids", "unit"))


def index_cells(cells):
    result = {}
    for cell in cells:
        if cell["id"] in result:
            raise ValueError("duplicate evidence cell")
        result[cell["id"]] = cell
    return result


def read_table_cells(run, payload):
    version = payload.get("evidence_schema_version")
    if version == 1:
        return index_cells(payload["evidence_cells"])
    if version != 2:
        raise ValueError("unsupported table evidence schema")
    tables = {t["id"]: t for t in run.get("logical_tables", [])}
    table = tables.get(payload["table_id"])
    if table is None or table["fragment_ids"] != payload["fragment_ids"]:
        raise ValueError("mismatched logical table")
    fragments = {f["id"]: f for f in run["table_fragments"]}
    if len(fragments) != len(run["table_fragments"]):
        raise ValueError("duplicate table fragment")
    cells = []
    for identifier in table["fragment_ids"]:
        for cell in fragments[identifier]["cells"]:
            if cell["fragment_id"] != identifier:
                raise ValueError("mismatched cell owner")
            cells.append(cell)
    return index_cells(cells)


def supporting_cells(cell, cells):
    """同页原始关联；不从生成上下文恢复证据。"""
    identifiers = dict.fromkeys(i for name, _ in RELATION_ROLES for i in cell.get(name, []))
    result = []
    for identifier in identifiers:
        support = cells[identifier]
        if support["fragment_id"] != cell["fragment_id"]:
            raise ValueError("cross-fragment support")
        result.append({"cell_id": identifier, "text": support["raw_text"], "spans": support["spans"]})
    return result
