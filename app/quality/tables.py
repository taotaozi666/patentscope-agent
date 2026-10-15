"""Check table evidence independently of backend success and flattened text."""

from collections import Counter, defaultdict
from dataclasses import asdict
import json

from app.schemas.quality import TableQuality


def table_groups(parsed):
    groups = [(t.id, list(t.fragment_ids)) for t in parsed.logical_tables]
    assigned = {i for _, ids in groups for i in ids}
    return [*groups, *((f.id, [f.id]) for f in parsed.table_fragments if f.id not in assigned)]


def check_tables(parsed, chunks, issue, valid_span):
    groups = table_groups(parsed)
    owner = {i: tid for tid, ids in groups for i in ids}
    fragments = {f.id: f for f in parsed.table_fragments}
    cells = {c.id: c for f in parsed.table_fragments for c in f.cells}
    duplicates = Counter(c.id for f in parsed.table_fragments for c in f.cells)
    grids, checked_grids = {}, set()

    def emit(code, message, fragment=None, ids=(), status="review", table_id=None, chunk_id=None):
        spans = [asdict(s) for i in ids if i in cells for s in cells[i].spans]
        if not spans and fragment:
            spans = [asdict(fragment.span)]
        issue(code, message, fragment.span.page_no if fragment else None,
              fragment.id if fragment else None, chunk_id, status,
              table_id=table_id or (owner.get(fragment.id) if fragment else None),
              cell_ids=list(ids), source_spans=spans)

    if len(fragments) != len(parsed.table_fragments):
        for fragment in parsed.table_fragments:
            emit("table_fragment_id", "表格片段标识重复，无法确定来源。", fragment, status="blocked")
    for fragment in parsed.table_fragments:
        if not valid_span(asdict(fragment.span), parsed.page_count):
            emit("table_region", "表格区域或页码无效。", fragment, status="blocked")
        if fragment.header_rows is None:
            emit("table_header_unknown", "表头证据不足，所有行按原文保留，表头关联需要复核。", fragment)
        if fragment.structure_status != "complete" or fragment.provenance_status != "complete":
            emit("table_structure_uncertain", "表格结构或单元格来源尚未完整验证。", fragment)
        if fragment.backend == "docling" or any(c.raw_text and c.text_origin != "native" for c in fragment.cells):
            emit("table_recognition_review", "表格含 OCR、模型或未知文字通道，识别与结构结果需要对照原件核验。", fragment)
        dimensions_valid = type(fragment.row_count) is int and type(fragment.column_count) is int and min(fragment.row_count, fragment.column_count) > 0
        bounded = dimensions_valid and fragment.row_count * fragment.column_count <= 100000
        if not bounded:
            emit("table_grid_budget", "表格维度无效或超过完整网格检查预算，需要复核。", fragment,
                 status="review" if dimensions_valid else "blocked")
        else:
            checked_grids.add(fragment.id)
        grid = {}
        for cell in fragment.cells:
            if duplicates[cell.id] > 1 or cell.fragment_id != fragment.id:
                emit("table_cell_id", "单元格标识重复或片段归属错误。", fragment, [cell.id], "blocked")
            if cell.row_span is None or cell.col_span is None:
                emit("table_span_unknown", "单元格合并跨度未知，不按已确认跨度使用。", fragment, [cell.id])
            values = (cell.row, cell.column, cell.row_span if cell.row_span is not None else 1,
                      cell.col_span if cell.col_span is not None else 1)
            legal = all(type(n) is int for n in values) and min(values[:2]) >= 0 and min(values[2:]) >= 1
            legal = legal and dimensions_valid and cell.row + values[2] <= fragment.row_count and cell.column + values[3] <= fragment.column_count
            if not legal:
                emit("table_span_bounds", "单元格行列或跨度越界。", fragment, [cell.id], "blocked")
                continue
            if cell.state in {"unknown", "unreadable"}:
                emit("table_cell_unreadable", "单元格内容未知或未识别；不能作为零值或已确认空白。", fragment, [cell.id])
            if not cell.spans and cell.state != "covered":
                emit("table_cell_location", "单元格缺少独立来源位置。", fragment, [cell.id])
            for span in cell.spans:
                box, table_box = span.bbox, fragment.span.bbox
                valid = valid_span(asdict(span), parsed.page_count) and valid_span(asdict(fragment.span), parsed.page_count)
                inside = (valid and table_box[0] - .01 <= box[0] < box[2] <= table_box[2] + .01
                          and table_box[1] - .01 <= box[1] < box[3] <= table_box[3] + .01)
                if not valid or span.page_no != fragment.span.page_no or not inside:
                    emit("table_cell_location", "单元格来源无效、位于其他页或超出表格区域。", fragment, [cell.id], "blocked")
            if cell.state == "covered":
                anchor = cells.get(cell.covered_by_cell_id)
                if (not anchor or anchor.fragment_id != fragment.id or anchor.state == "covered" or anchor.id == cell.id
                        or not (anchor.row <= cell.row < anchor.row + (anchor.row_span or 1)
                                and anchor.column <= cell.column < anchor.column + (anchor.col_span or 1))):
                    emit("table_covered_anchor", "合并占位没有有效的覆盖锚点。", fragment, [cell.id], "blocked")
            elif bounded:
                for row in range(cell.row, cell.row + values[2]):
                    for col in range(cell.column, cell.column + values[3]):
                        previous = grid.get((row, col))
                        if previous and previous != cell.id:
                            emit("table_span_overlap", "两个单元格覆盖同一行列位置。", fragment, [previous, cell.id], "blocked")
                        grid[row, col] = cell.id
            for name in ("row_header_cell_ids", "column_header_cell_ids", "unit_cell_ids"):
                for identifier in getattr(cell, name):
                    reference = cells.get(identifier)
                    if not reference or reference.fragment_id != fragment.id or identifier == cell.id:
                        emit("table_reference", "表格行列表头或单位引用失效。", fragment, [cell.id], "blocked")
                        continue
                    if name == "column_header_cell_ids":
                        aligned = reference.row in (fragment.header_rows or []) and reference.column < cell.column + (cell.col_span or 1) and cell.column < reference.column + (reference.col_span or 1)
                    elif name == "row_header_cell_ids":
                        aligned = reference.column < cell.column and reference.row <= cell.row < reference.row + (reference.row_span or 1)
                    else:
                        aligned = identifier in cell.column_header_cell_ids
                    if not aligned:
                        emit("table_reference_alignment", "表头或单位引用与该单元格行列不一致。", fragment, [cell.id, identifier], "blocked")
        if bounded and len(grid) != fragment.row_count * fragment.column_count:
            emit("table_grid_gap", "表格存在没有单元格锚点覆盖的位置。", fragment,
                 status="blocked" if fragment.structure_status == "complete" and all(
                     c.row_span is not None and c.col_span is not None for c in fragment.cells) else "review")
        grids[fragment.id] = grid
        for diagnosis in parsed.page_inspections:
            if diagnosis.page_no == fragment.span.page_no and any(a.get("status") == "failed" for a in diagnosis.metrics.get("table_backends", [])):
                emit("table_backend_fallback", "本页表格后端曾失败或回退，请核对保留的表结构。", fragment)

    # Detect lost/duplicated data and mismatched values after slicing. Header
    # repetition is allowed; data intervals must refer to the original cell.
    coverage = defaultdict(list)
    for chunk in chunks:
        if chunk.metadata.get("doc_type_kwd") != "table":
            continue
        try:
            mappings = json.loads(chunk.metadata["pdf_serialized_map"])
            for mapping in mappings:
                if mapping["end"] <= mapping["start"]:
                    continue
                cell = cells.get(mapping.get("cell_id"))
                if not cell:
                    issue("table_chunk_cell", "切片引用了不存在的表格单元格。", chunk=chunk.metadata["pdf_chunk_id"], status="blocked")
                    continue
                fragment = fragments[cell.fragment_id]
                if chunk.metadata.get("table_id") != owner.get(fragment.id):
                    emit("table_chunk_owner", "切片所属逻辑表与单元格归属不一致。", fragment, [cell.id], "blocked", chunk_id=chunk.metadata["pdf_chunk_id"])
                if fragment.header_rows is not None and ((mapping.get("role") == "header") != (cell.row in fragment.header_rows)):
                    emit("table_chunk_role", "切片将表头或数据格标记为错误角色。", fragment, [cell.id], "blocked", chunk_id=chunk.metadata["pdf_chunk_id"])
                text = cell.raw_text or ""
                start, end = mapping.get("source_char_start", 0), mapping.get("source_char_end", len(text))
                valid_range = type(start) is int and type(end) is int and 0 <= start <= end <= len(text)
                same_location = any(json.dumps(mapping.get("span"), sort_keys=True) == json.dumps(asdict(s), sort_keys=True) for s in cell.spans)
                if not valid_range or (cell.spans and not same_location) or chunk.page_content[mapping["start"]:mapping["end"]] != text[start:end]:
                    emit("table_value_mapping", "切片数值或来源与原单元格不一致。", fragment, [cell.id], "blocked", chunk_id=chunk.metadata["pdf_chunk_id"])
                if valid_range and mapping.get("role") != "header":
                    coverage[cell.id].append((start, end))
        except (ValueError, TypeError, KeyError, IndexError):
            continue  # Generic character-map validation reports malformed maps.
    for fragment in parsed.table_fragments:
        for cell in fragment.cells:
            if not cell.raw_text or cell.row in (fragment.header_rows or []) or cell.state == "covered":
                continue
            intervals = sorted(coverage[cell.id])
            position, repeated = 0, False
            for start, end in intervals:
                if start < position:
                    repeated = True
                if start > position:
                    break
                position = max(position, end)
            if position != len(cell.raw_text):
                emit("table_unindexed_cell", "单元格原文未完整进入当前检索切片。", fragment, [cell.id])
            if repeated:
                emit("table_data_duplicate", "同一数据格的字符被重复索引。", fragment, [cell.id])

    counts = Counter(i for table in parsed.logical_tables for i in table.fragment_ids)
    for table in parsed.logical_tables:
        found = [fragments[i] for i in table.fragment_ids if i in fragments]
        first = found[0] if found else None
        if len(found) != len(table.fragment_ids) or any(counts[i] > 1 for i in table.fragment_ids):
            emit("table_logical_reference", "逻辑表引用缺失片段或多个逻辑表共用了片段。", first, status="blocked", table_id=table.id)
        if table.continuation_candidate_ids:
            emit("table_continuation_candidate", "相邻表格可能续接，证据不足，当前保持独立，请复核。", first, table_id=table.id)
        valid_ids = {c.id for f in found for c in f.cells}
        used = [i for row in table.data_rows for i in row] + list(table.header_paths) + [i for path in table.header_paths.values() for i in path] + table.repeated_header_cell_ids
        if any(i not in valid_ids for i in used):
            emit("table_logical_reference", "逻辑数据行或表头路径引用了表外单元格。", first, status="blocked", table_id=table.id)
        if any(i in cells and path != cells[i].column_header_cell_ids for i, path in table.header_paths.items()):
            emit("table_header_path", "逻辑表头路径与单元格的列表头引用不一致。", first, status="blocked", table_id=table.id)
        complete_grids = all(f.id in checked_grids and len(grids.get(f.id, {})) == f.row_count * f.column_count and all(
            c.row_span is not None and c.col_span is not None for c in f.cells) for f in found)
        expected_rows = [list(dict.fromkeys(grids[f.id][r, c] for c in range(f.column_count)))
                         for f in found for r in range(f.row_count) if r not in (f.header_rows or [])] if complete_grids else None
        if expected_rows is not None and table.data_rows != expected_rows and (table.data_rows or "logical-tables-" in parsed.parser_version):
            emit("table_data_rows", "逻辑数据行的单元格归属、顺序或覆盖不完整。", first, status="blocked", table_id=table.id)
        for row in table.data_rows:
            if any(i in cells and cells[i].row in (fragments[cells[i].fragment_id].header_rows or []) for i in row):
                emit("table_header_as_data", "重复表头被作为逻辑数据行使用。", first, row, "blocked", table.id)
        for prior, current in zip(found, found[1:]):
            from app.pdf.tables.assembly import _neighbors, _LABEL
            prior_caption, prior_units, _ = _neighbors(parsed, prior)
            current_caption, current_units, _ = _neighbors(parsed, current)
            labels: list[str] = []
            for caption in (prior_caption, current_caption):
                label_match = _LABEL.match(caption.text) if caption is not None else None
                labels.append(label_match.group(1).casefold() if label_match is not None else "")
            labels_conflict = bool(all(labels) and labels[0] != labels[1])
            units_conflict = bool(prior_units and current_units and "".join("".join(e.text.split()) for e in prior_units)
                                  != "".join("".join(e.text.split()) for e in current_units))
            def signature(f):
                grid = grids.get(f.id, {})
                return tuple(tuple("".join((cells[i].raw_text or "").split()) for i in dict.fromkeys(
                    grid[r, col] for r in (f.header_rows or []) if (r, col) in grid)) for col in range(f.column_count))
            if (current.span.page_no != prior.span.page_no + 1 or current.column_count != prior.column_count
                    or current.section_path != prior.section_path or signature(prior) != signature(current)):
                emit("table_continuation_conflict", "续表的页序、列数、章节或表头（含单位）不一致。", current, status="blocked", table_id=table.id)
            if labels_conflict or units_conflict:
                emit("table_continuation_conflict", "续表的表号或显式单位声明冲突。", current, status="blocked", table_id=table.id)
            if not prior.header_rows or not current.header_rows:
                emit("table_continuation_unverified", "续表缺少已确认的表头关系。", current, table_id=table.id)
            expected = {c.id for c in current.cells if c.row in (current.header_rows or [])}
            if not expected.issubset(set(table.repeated_header_cell_ids)):
                emit("table_repeated_header", "续页重复表头缺少来源引用。", current, status="blocked", table_id=table.id)


def summarize_tables(parsed, issues):
    fragments = {f.id: f for f in parsed.table_fragments}
    result = []
    for identifier, ids in table_groups(parsed):
        relevant = [i for i in issues if i.table_id == identifier or i.element_id in ids]
        status = "blocked" if any(i.status == "blocked" for i in relevant) else "review" if relevant else "passed"
        found = [fragments[i] for i in ids if i in fragments]
        result.append(TableQuality(identifier, ids, [c.id for f in found for c in f.cells],
            sorted({f.span.page_no for f in found}), status,
            {"blocked": "blocked", "review": "reference_only", "passed": "traceable"}[status], [i.id for i in relevant]))
    return result


def table_usage(report, table, reviews):
    """Shared effective usage for review UI and retrieval; rules remain immutable."""
    targets = {"document", f"table:{table.id}", *(f"page:{p}" for p in table.pages),
               *(f"cell:{table.id}:{i}" for i in table.cell_ids), *(f"issue:{i}" for i in table.issue_ids)}
    selected = [r for r in reviews if r.target in targets]
    manual = "problem" if any(r.decision == "problem" for r in selected) else "passed" if any(
        r.decision == "passed" and r.target in {"document", f"table:{table.id}"} for r in selected) else "pending"
    usage = "blocked" if report.status == "blocked" else table.usage
    if manual == "problem" and usage != "blocked":
        usage = "reference_only"
    return {"status": table.status, "usage": usage, "manual_review": manual,
            "report_id": report.id, "rule_version": report.rule_version}
