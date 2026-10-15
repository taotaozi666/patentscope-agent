"""Conservative logical tables: evidence references, header paths and continuation gates."""

from hashlib import sha256
import re

from app.schemas.table import LogicalTable

ASSEMBLY_VERSION = "logical-tables-v1"
_NUMBER = re.compile(r"^[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*%?$")
_LABEL = re.compile(r"^\s*(?:表|Table)\s*([0-9]+(?:[.\-][0-9]+)*[a-zA-Z]?)(?![0-9a-zA-Z])", re.I)
_UNIT = re.compile(r"(?:单位\s*[:：]|(?<![a-zA-Z])(?:MPa|kPa|Pa|GPa|mm|cm|kg|mg|g|mol|mL|Hz)(?![a-zA-Z])|[%％℃°])")


def _key(text):
    return _compact(text).casefold()


def _compact(text):
    # SI prefixes are case sensitive: MPa and mPa must never compare equal.
    return re.sub(r"\s+", "", text or "")


def occupancy(fragment):
    slots = {}
    for cell in fragment.cells:
        if cell.state == "covered":
            continue
        for row in range(cell.row, cell.row + (cell.row_span or 1)):
            for column in range(cell.column, cell.column + (cell.col_span or 1)):
                if (row, column) in slots:
                    raise ValueError("逻辑表存在单元格覆盖冲突")
                slots[row, column] = cell
    return slots


def _headers(fragment, slots):
    if fragment.header_rows is not None:
        return list(fragment.header_rows), fragment.header_method if fragment.header_method != "unknown" else "backend"
    first_stub = slots.get((0, 0))
    if first_stub and _key(first_stub.raw_text) in {"metric", "sample", "实施例", "指标", "样品", "项目"}:
        if any(_NUMBER.fullmatch((slots[1, col].raw_text or "").strip())
               for col in range(1, fragment.column_count) if (1, col) in slots):
            return [0], "explicit_stub_and_numeric_body"
    # Only a leading nonnumeric band followed by numeric data establishes a header.
    # Text-only tables remain unknown instead of treating their first record as a header.
    for row in range(fragment.row_count):
        numeric = any(_NUMBER.fullmatch((slots[row, col].raw_text or "").strip())
                      for col in range(1, fragment.column_count) if (row, col) in slots)
        if numeric:
            if not row:
                return None, "unknown"
            band = list(range(row))
            if all(any((r, c) in slots and slots[r, c].raw_text for r in band)
                   for c in range(fragment.column_count)):
                return band, "numeric_body_transition"
            break
    return None, "unknown"


def _neighbors(parsed, fragment):
    # Assign each nearby paragraph to its closest table, preventing two adjacent
    # tables from sharing an unrelated caption, unit declaration or footnote.
    found = []
    others = [f for f in parsed.table_fragments if f.span.page_no == fragment.span.page_no]
    for element in parsed.elements:
        if element.kind not in {"paragraph", "heading"} or element.span.page_no != fragment.span.page_no:
            continue
        if len(element.text) > 250:
            continue
        def distance(f):
            if min(f.span.bbox[2], element.span.bbox[2]) <= max(f.span.bbox[0], element.span.bbox[0]):
                return 10
            return min(abs(element.span.bbox[3] - f.span.bbox[1]),
                       abs(element.span.bbox[1] - f.span.bbox[3]))
        if distance(fragment) <= .08 and min(others, key=distance).id == fragment.id:
            found.append(element)
    above = sorted((e for e in found if e.span.bbox[3] <= fragment.span.bbox[1] + .01),
                   key=lambda e: -e.span.bbox[3])
    caption = next((e for e in above if _LABEL.match(e.text)), None)
    units = [e for e in found if re.match(r"^\s*单位\s*[:：]", e.text)]
    notes = [e for e in found if e.span.bbox[1] >= fragment.span.bbox[3] - .01
             and re.match(r"^\s*(?:注|备注|Note)\s*[:：]", e.text, re.I)]
    return caption, units, notes


def assemble_tables(parsed):
    """Rebuild semantic links without changing raw text, geometry or backend output."""
    tables = []
    previous = None
    previous_info = None
    for fragment in sorted(parsed.table_fragments, key=lambda f: (f.span.page_no, f.span.bbox[1], f.span.bbox[0])):
        slots = occupancy(fragment)
        headers, method = _headers(fragment, slots)
        fragment.header_rows = headers
        fragment.header_method = method
        caption, units, notes = _neighbors(parsed, fragment)
        label_match = _LABEL.match(caption.text) if caption else None
        label = label_match[1].casefold() if label_match else ""
        continued = bool(caption and re.search(r"continued|续", caption.text, re.I))
        header_ids = {cell.id for cell in fragment.cells if cell.row in (headers or [])}
        paths = {}
        for column in range(fragment.column_count):
            paths[column] = list(dict.fromkeys(slots[row, column].id for row in (headers or [])
                                                if (row, column) in slots))
        by_id = {c.id: c for c in fragment.cells}
        for cell in fragment.cells:
            cell.column_header_cell_ids = []
            cell.row_header_cell_ids = []
            cell.unit_cell_ids = []
            if cell.id in header_ids or cell.state == "covered":
                continue
            cell.column_header_cell_ids = list(dict.fromkeys(
                identifier for col in range(cell.column, cell.column + (cell.col_span or 1))
                for identifier in paths[col]))
            # Explicit leading textual stub(s), including a rowspan anchor.
            for col in range(cell.column):
                stub = slots.get((cell.row, col))
                if not stub or not stub.raw_text or _NUMBER.fullmatch(stub.raw_text.strip()):
                    break
                if stub.id != cell.id and stub.id not in cell.row_header_cell_ids:
                    cell.row_header_cell_ids.append(stub.id)
            cell.unit_cell_ids = [identifier for identifier in cell.column_header_cell_ids
                                  if _UNIT.search(by_id[identifier].raw_text or "")]
        rows = [list(dict.fromkeys(slots[row, col].id for col in range(fragment.column_count)
                                  if (row, col) in slots))
                for row in range(fragment.row_count) if row not in (headers or [])]
        signature = tuple(tuple(_compact(by_id[i].raw_text) for i in paths[col]) for col in paths)
        info = dict(label=label, signature=signature, unit="\n".join(e.text for e in units),
                    continued=continued, fragment=fragment)
        decision = "single_fragment"
        merge = False
        candidate = False
        if previous_info:
            prior = previous_info["fragment"]
            adjacent = fragment.span.page_no == prior.span.page_no + 1
            same_columns = fragment.column_count == prior.column_count
            same_headers = bool(headers) and signature == previous_info["signature"]
            same_section = fragment.section_path == prior.section_path
            aligned = abs(fragment.span.bbox[0] - prior.span.bbox[0]) <= .035 and abs(fragment.span.bbox[2] - prior.span.bbox[2]) <= .035
            labels_conflict = bool(label and previous_info["label"] and label != previous_info["label"])
            units_conflict = bool(info["unit"] and previous_info["unit"] and
                                  _compact(info["unit"]) != _compact(previous_info["unit"]))
            # Same header text includes units; equal column count alone is never sufficient.
            compatible = adjacent and same_columns and same_headers and same_section and aligned and not labels_conflict and not units_conflict
            same_label = bool(label and label == previous_info["label"])
            edge_flow = prior.span.bbox[3] >= .8 and fragment.span.bbox[1] <= .25
            merge = compatible and ((same_label and continued) or (edge_flow and not label))
            candidate = compatible and not merge
            decision = "joined_explicit_continuation" if merge and continued else "joined_page_edges" if merge else "candidate_only" if candidate else "incompatible_or_insufficient_evidence"
        fragment.continuation_decision = decision
        if not merge:
            logical = LogicalTable("lt_" + sha256(f"{parsed.content_hash}:{fragment.id}".encode()).hexdigest()[:20], [fragment.id],
                title=caption.text if caption else "表格", table_label=label, section_path=fragment.section_path,
                structure_status="partial")
            if candidate:
                previous.continuation_candidate_ids.append(logical.id)
                logical.continuation_candidate_ids.append(previous.id)
            tables.append(logical)
        else:
            logical = previous
            logical.fragment_ids.append(fragment.id)
            logical.repeated_header_cell_ids.extend(sorted(header_ids))
        logical.data_rows.extend(rows)
        if caption:
            logical.title_spans.append(caption.span)
        for cell in fragment.cells:
            logical.header_paths[cell.id] = list(cell.column_header_cell_ids)
        if units:
            logical.unit_text = "\n".join(dict.fromkeys(filter(None, [logical.unit_text, *(e.text for e in units)])))
            logical.unit_spans.extend(e.span for e in units)
        logical.footnotes.extend(e.text for e in notes)
        logical.footnote_spans.extend(e.span for e in notes)
        # Carry explicit table identity across intermediate pages that omit it.
        info["label"] = logical.table_label or label
        info["unit"] = logical.unit_text or info["unit"]
        previous, previous_info = logical, info
    parsed.logical_tables = tables
    return tables
