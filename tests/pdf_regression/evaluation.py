"""Compare current parser output with fixed gold cells, never infer gold from it."""

import json
from pathlib import Path
from dataclasses import asdict

from app.schemas.pdf import ParsedPdf
from app.schemas.inspection import PageInspection


def load_expectations() -> dict:
    return json.loads(Path(__file__).with_name("expected.json").read_text(encoding="utf-8"))


def _same_box(actual, expected, tolerance=.006) -> bool:
    return (isinstance(actual, (list, tuple)) and len(actual) == 4
            and all(abs(a - b) <= tolerance for a, b in zip(actual, expected)))


def _text(value) -> str:
    return "" if value is None else str(value).strip()


def _inside(actual, expected, tolerance=.006) -> bool:
    return (isinstance(actual, (list, tuple)) and len(actual) == 4 and
            expected[0] - tolerance <= actual[0] < actual[2] <= expected[2] + tolerance and
            expected[1] - tolerance <= actual[1] < actual[3] <= expected[3] + tolerance)


def evaluate(parsed: ParsedPdf | None, expected: dict, *, error: str | None = None,
             page_inspections: list[PageInspection] | None = None) -> dict:
    """Report target gaps as data; a successful parse is not a successful table.

    Physical cells are matched by independent positions; semantic links and
    logical groups are then compared by mapped evidence IDs, never by numbers alone.
    """
    tables = [item for item in parsed.elements if item.kind == "table"] if parsed else []
    metrics = {name: {"matched": 0, "expected": 0} for name in (
        "table_regions", "cell_text", "cell_location", "cell_topology", "paragraph_once",
    )}
    mismatches = []
    inspections = parsed.page_inspections if parsed is not None else (page_inspections or [])
    unavailable = ["unit_normalization_and_footnote_semantics"]
    fragment_pairs, cell_pairs = {}, {}
    if inspections:
        source_metric = {"matched": 0, "expected": len(expected["page_sources"])}
        for page_no, source in enumerate(expected["page_sources"], 1):
            found = [item for item in inspections if item.page_no == page_no]
            matched = len(found) == 1 and found[0].text_source == source
            source_metric["matched"] += int(matched)
            if not matched:
                mismatches.append({"check": "page_source_classification", "page_no": page_no,
                                   "expected_source": source})
        metrics["page_source_classification"] = source_metric
        route_metric = {"matched": 0, "expected": 0}
        for inspection in inspections:
            for region in inspection.regions:
                routes = [route for route in inspection.routes if route.region_id == region.id]
                matched = len(routes) == 1 and bool(routes[0].planned_operations)
                route_metric["expected"] += 1
                route_metric["matched"] += int(matched)
                if not matched:
                    mismatches.append({"check": "routing_plan_coverage", "region_id": region.id})
        metrics["routing_plan_coverage"] = route_metric
    else:
        unavailable.append("page_source_routing")
    for fragment in expected["fragments"]:
        metrics["table_regions"]["expected"] += 1
        matches = [item for item in tables if item.span.page_no == fragment["page_no"]
                   and (_same_box(item.span.bbox, fragment["bbox"]) or
                        (item.payload.get("table_backend") in {"pymupdf-text", "docling"}
                         and item.payload.get("row_count") == fragment["row_count"]
                         and item.payload.get("column_count") == fragment["column_count"]
                         and _inside(item.span.bbox, fragment["bbox"])))]
        table = matches[0] if len(matches) == 1 else None
        metrics["table_regions"]["matched"] += int(table is not None)
        if table is None:
            mismatches.append({"fragment_id": fragment["id"], "check": "table_region"})
        actual_cells = table.payload.get("cells", []) if table else []
        if table:
            fragment_pairs[fragment["id"]] = table.id
        for cell in fragment["cells"]:
            # Pair by physical position, not text, to catch swapped numbers and
            # identical values appearing in separate columns or pages.
            located = [actual for actual in actual_cells
                       if actual.get("state") != "covered" and actual.get("span") and
                       (_same_box(actual["span"].get("bbox"), cell["bbox"]) or
                        (actual["span"].get("granularity") == "cell_content" and
                         _inside(actual["span"].get("bbox"), cell["bbox"]))) and
                       actual["span"].get("page_no") == fragment["page_no"]]
            actual = located[0] if len(located) == 1 else None
            if actual:
                cell_pairs[fragment["id"], cell["id"]] = actual["id"]
            checks = {
                "cell_location": actual is not None,
                "cell_text": actual is not None and _text(actual.get("raw_text")) == cell["raw_text"],
                "cell_topology": actual is not None and all(
                    actual.get(key, 1 if key in {"row_span", "col_span"} else None) == cell[key]
                    for key in ("row", "column", "row_span", "col_span")
                ),
            }
            for name, matched in checks.items():
                metrics[name]["expected"] += 1
                metrics[name]["matched"] += int(matched)
                if not matched:
                    mismatches.append({"fragment_id": fragment["id"], "cell_id": cell["id"],
                                       "check": name, "expected_text": cell["raw_text"]})
    actual_by_id = {cell.id: cell for f in parsed.table_fragments for cell in f.cells} if parsed else {}
    relation_metric = {"matched": 0, "expected": len(expected["numeric_facts"])}
    for fact in expected["numeric_facts"]:
        key = fact["fragment_id"], fact["cell_id"]
        actual = actual_by_id.get(cell_pairs.get(key))
        matched = actual is not None and actual.raw_text == fact["raw_value"]
        for name in ("row_header_cell_ids", "column_header_cell_ids"):
            desired = [cell_pairs.get((fact["fragment_id"], i)) for i in fact[name]]
            matched = matched and None not in desired and getattr(actual, name) == desired
        relation_metric["matched"] += int(bool(matched))
        if not matched:
            mismatches.append({"check": "header_association", "fragment_id": key[0], "cell_id": key[1]})
    metrics["header_association"] = relation_metric
    groups = [set(t.fragment_ids) for t in parsed.logical_tables] if parsed else []
    desired_groups = [{fragment_pairs.get(i) for i in group} for group in expected["logical_tables"]]
    matched_groups = sum(None not in group and group in groups for group in desired_groups)
    metrics["logical_table_grouping"] = {"matched": matched_groups, "expected": len(desired_groups)}
    if matched_groups != len(desired_groups) or len(groups) != len(desired_groups):
        mismatches.append({"check": "logical_table_grouping"})
    for paragraph in expected["paragraphs"]:
        count = sum(item.text.count(paragraph["text"]) for item in parsed.elements
                    if item.span.page_no == paragraph["page_no"]) if parsed else 0
        metrics["paragraph_once"]["expected"] += 1
        metrics["paragraph_once"]["matched"] += int(count == 1)
        if count != 1:
            mismatches.append({"check": "paragraph_once", "occurrences": count})
    measured_pass = not mismatches and error is None and len(tables) == len(expected["fragments"])
    return {
        "case_id": expected["id"], "parser_version": parsed.parser_version if parsed else None,
        "parse_error": error, "expected_table_count": len(expected["fragments"]),
        "actual_table_count": len(tables), "metrics": metrics,
        "measured_checks_passed": measured_pass,
        "unavailable_checks": unavailable,
        "target_complete": False, "mismatches": mismatches,
        "warnings": parsed.warnings if parsed else [],
        "page_inspections": [asdict(item) for item in inspections],
    }
