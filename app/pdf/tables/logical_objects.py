"""Project logical evidence into the existing row-based chunk interface."""

from dataclasses import asdict

from .base import to_element


def logical_payload(parsed, logical):
    fragments = {f.id: f for f in parsed.table_fragments}
    rows, cells, header_indices, spans = [], [], [], []
    relations = {}
    fragment_regions = {}
    first = fragments[logical.fragment_ids[0]]
    properties = to_element(first, parsed.content_hash).payload
    for index, identifier in enumerate(logical.fragment_ids):
        fragment = fragments[identifier]
        projection = properties if index == 0 else to_element(fragment, parsed.content_hash).payload
        fragment_regions[identifier] = {"fragment_id": identifier, "span": asdict(fragment.span),
                                        "image_id": projection["image_id"]}
        projected = {(c["row"], c["column"]): c for c in projection["cells"]}
        spans.append(fragment.span)
        for cell in fragment.cells:
            relations[cell.id] = {
                "row_header_cell_ids": cell.row_header_cell_ids,
                "column_header_cell_ids": cell.column_header_cell_ids,
                "unit_cell_ids": cell.unit_cell_ids,
            }
        for row in range(fragment.row_count):
            is_header = row in (fragment.header_rows or [])
            if index and is_header:
                continue  # Repeated header references remain in LogicalTable, not data rows.
            new_row = len(rows)
            rows.append(projection["rows"][row])
            if is_header:
                header_indices.append(new_row)
            for column in range(fragment.column_count):
                cells.append({**projected[row, column], "row": new_row})
    payload = {**properties, "rows": rows, "cells": cells, "row_count": len(rows),
               "evidence_schema_version": 2,
               "fragment_regions": fragment_regions,
               "related_evidence": [
                   *([{"role": "title", "text": logical.title, "spans": [asdict(logical.title_spans[0])]}] if logical.title_spans else []),
                   *([{"role": "unit", "text": logical.unit_text, "spans": [asdict(s) for s in logical.unit_spans]}] if logical.unit_text else []),
                   *[{"role": "footnote", "text": text, "spans": [asdict(span)]}
                     for text, span in zip(logical.footnotes, logical.footnote_spans)]],
               "properties": {**properties["properties"], "row_count": len(rows)},
               "header_rows": header_indices, "header_status": "logical_header_paths" if first.header_rows is not None else "unknown",
               "table_id": logical.id, "fragment_ids": logical.fragment_ids, "cell_relations": relations,
               "section_path": logical.section_path, "caption": logical.title,
               "units": [logical.unit_text] if logical.unit_text else [], "notes": logical.footnotes,
               "related_spans": [asdict(s) for s in [*logical.title_spans, *logical.unit_spans, *logical.footnote_spans]],
               "requires_review": logical.structure_status != "complete"}
    text = "\n".join(" | ".join(v or "" for v in row) for row in rows)
    return text, spans, payload
