"""Bounded logical-table row/column groups with cell-level source intervals."""

import json


def table_chunks(parent, size, target, split_ranges):
    represented_headers = set()
    for text, mappings, extra in _table_chunks(parent, size, target, split_ranges):
        represented_headers.update(m["cell_id"] for m in mappings if m["role"] == "header" and m["end"] > m["start"])
        yield text, mappings, extra
    # If a header alone exceeded the row/column budget, it remains an indexed
    # evidence entry rather than being lost in truncated embedding context.
    headers = parent.payload["header_rows"] or []
    for cell in parent.payload["cells"]:
        if cell["row"] not in headers or cell["id"] in represented_headers or not cell["raw_text"]:
            continue
        for start, end in split_ranges(cell["raw_text"], size, target):
            yield cell["raw_text"][start:end], [{"start": 0, "end": end-start, "cell_id": cell["id"],
                "fragment_id": cell["fragment_id"], "source_char_start": start, "source_char_end": end,
                "span": cell["span"], "role": "header"}], {"table_split": "header"}


def _table_chunks(parent, size, target, split_ranges):
    rows = parent.payload["rows"]
    cells = {(c["row"], c["column"]): c for c in parent.payload["cells"]}
    headers = parent.payload["header_rows"] or []
    columns = list(range(len(rows[0])))
    data = [r for r in range(len(rows)) if r not in headers]

    def render(indices, selected):
        text, mappings = "", []
        for row in indices:
            if text:
                text += "\n"
            for index, column in enumerate(selected):
                if index:
                    text += " | "
                value = rows[row][column] or ""
                cell = cells[row, column]
                start = len(text)
                text += value
                mappings.append({"start": start, "end": len(text), "cell_id": cell["id"],
                    "fragment_id": cell["fragment_id"],
                    "source_char_start": 0, "source_char_end": len(value),
                    "span": cell["span"], "role": "header" if row in headers else "source"})
        return text, mappings

    def group(indices, selected, kind):
        text, mappings = render([*headers, *indices], selected)
        return text, mappings, {"table_split": kind, "table_rows": json.dumps(indices),
                                "table_columns": json.dumps(selected)}

    pending = []
    for row in data:
        text, _, _ = group([row], columns, "row_group")
        if len(text) <= size:
            combined, _, _ = group([*pending, row], columns, "row_group")
            if pending and len(combined) > target:
                yield group(pending, columns, "row_group")
                pending = []
            pending.append(row)
            continue
        if pending:
            yield group(pending, columns, "row_group")
            pending = []
        selected = []
        for column in columns:
            combined, _, _ = group([row], [*selected, column], "column_group")
            if selected and len(combined) > target:
                yield group([row], selected, "column_group")
                selected = []
            single, _, _ = group([row], [column], "column_group")
            if len(single) <= size:
                selected.append(column)
                continue
            # A single value/header exceeds the budget. Keep raw cell intervals;
            # the complete semantic header lives in separately referenced support.
            cell = cells[row, column]
            value = rows[row][column] or ""
            header = " / ".join(dict.fromkeys(rows[r][column] for r in headers if rows[r][column]))
            for start, end in split_ranges(value, size, target):
                yield value[start:end], [{"start": 0, "end": end-start, "cell_id": cell["id"],
                    "fragment_id": cell["fragment_id"],
                    "source_char_start": start, "source_char_end": end, "span": cell["span"], "role": "source"}], {
                    "table_split": "cell", "table_row": row, "table_column": column,
                    "table_rows": json.dumps([row]), "table_columns": json.dumps([column]),
                    "table_header": header, "table_row_label": rows[row][0] or ""}
        if selected:
            yield group([row], selected, "column_group")
    if pending:
        yield group(pending, columns, "row_group")
    if not data:
        text, mappings = render(headers, columns)
        for start, end in split_ranges(text, size, target):
            cropped = [{**m, "start": max(start, m["start"])-start, "end": min(end, m["end"])-start,
                        "source_char_start": max(start, m["start"])-m["start"],
                        "source_char_end": min(end, m["end"])-m["start"]}
                       for m in mappings if min(end, m["end"]) > max(start, m["start"])]
            yield text[start:end], cropped, {"table_split": "header"}
