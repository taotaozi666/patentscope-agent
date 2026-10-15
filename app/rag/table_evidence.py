"""Resolve table support from the exact saved parse, never generated context."""

from dataclasses import replace
import json
import math
import sqlite3

from app.schemas import Hit
from app.pdf.tables.evidence import read_table_cells, RELATION_ROLES

EVIDENCE_VERSION = "table-evidence-v1"


def _valid_spans(spans, page_count):
    return isinstance(spans, list) and bool(spans) and all(
        isinstance(s, dict) and type(s.get("page_no")) is int and 1 <= s["page_no"] <= page_count
        and isinstance(s.get("bbox"), (list, tuple)) and len(s["bbox"]) == 4
        and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in s["bbox"])
        and s["bbox"][0] < s["bbox"][2] and s["bbox"][1] < s["bbox"][3] for s in spans)


def expand_table_hits(hits, repository, *, max_cells=128, max_characters=6000):
    """Bound support within each hit; preserve top-k, ranking and original text."""
    output, cache = [], {}
    for hit in hits:
        if hit.metadata.get("doc_type_kwd") != "table":
            output.append(hit)
            continue
        metadata = dict(hit.metadata)
        evidence = {"version": EVIDENCE_VERSION, "status": "unavailable", "cells": [],
                    "associations": {}, "related": [], "issues": []}
        try:
            _expand(hit, repository, evidence, cache, max_cells, max_characters)
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, IndexError):
            # Do not expose database paths or accept partial data from a failed
            # identity check. This only changes the returned retrieval object.
            evidence.update(status="unavailable", cells=[], associations={}, related=[],
                            issues=["saved_evidence_unavailable_or_inconsistent"])
        metadata["table_evidence"] = evidence
        output.append(replace(hit, metadata=metadata))
    return output


def _expand(hit, repository, evidence, cache, max_cells, max_characters):
    if repository is None:
        raise ValueError("asset directory unavailable")
    def cached(kind, identifier, read):
        key = kind, identifier
        if key not in cache:
            cache[key] = read()
        return cache[key]

    identity = ("pdf_chunk_id", "pdf_parse_id", "pdf_version_id", "parent_id", "table_id")
    if any(not isinstance(hit.metadata.get(k), str) or not hit.metadata[k] for k in identity):
        raise ValueError("missing parse identity")
    saved = cached("chunk", hit.metadata["pdf_chunk_id"], lambda: repository.get_chunk(hit.metadata["pdf_chunk_id"]))
    if not saved or saved.get("text") != hit.text or saved["metadata"].get("source") != hit.source:
        raise ValueError("mismatched chunk")
    meta = saved["metadata"]
    if any(meta.get(k) != hit.metadata[k] for k in identity):
        raise ValueError("mismatched parse")
    run_id, object_id = meta["pdf_parse_id"], meta["parent_id"]
    run = cached("run", run_id, lambda: repository.get_run(run_id))
    if not run or run["content_hash"] != meta["pdf_version_id"] or run["source"] != hit.source:
        raise ValueError("mismatched document")
    evidence["quality"] = cached("quality", (run_id, meta["pdf_chunk_id"], meta["table_id"]),
        lambda: repository.get_table_quality(run_id, meta["pdf_chunk_id"], meta["table_id"]))
    parent = cached("object", (run_id, object_id), lambda: repository.get_object(object_id, run_id=run_id))
    if not parent or parent.get("kind") != "table":
        raise ValueError("missing table object")
    payload = parent["payload"]
    if payload.get("evidence_schema_version") not in {1, 2} or payload.get("table_id") != meta["table_id"]:
        raise ValueError("legacy or different table")
    cells = cached("table_cells", (run_id, object_id), lambda: read_table_cells(run, payload))
    roots = []
    for mapping in json.loads(meta["pdf_serialized_map"]):
        if mapping.get("role") == "header" or mapping["end"] <= mapping["start"]:
            continue
        cell = cells[mapping["cell_id"]]
        raw = cell["raw_text"]
        left, right = mapping.get("source_char_start", 0), mapping.get("source_char_end", len(raw))
        if (type(left) is not int or type(right) is not int or not 0 <= left <= right <= len(raw)
                or not 0 <= mapping["start"] < mapping["end"] <= len(hit.text)
                or hit.text[mapping["start"]:mapping["end"]] != raw[left:right]
                or mapping.get("span") not in cell["spans"]):
            raise ValueError("mismatched raw cell")
        if cell["id"] not in roots:
            roots.append(cell["id"])
    if not roots:
        evidence["issues"].append("no_data_cells")
    selected, characters = {}, 0

    def add(identifier, role):
        nonlocal characters
        cell = cells.get(identifier)
        if cell is None or not _valid_spans(cell["spans"], run["page_count"]):
            evidence["issues"].append("missing_support_location")
            return False
        text = cell["raw_text"]
        if not isinstance(text, str) or not text.strip() or cell["state"] != "text":
            evidence["issues"].append("unreadable_support_cell")
            return False
        if identifier not in selected:
            if len(selected) >= max_cells or characters + len(text) > max_characters:
                evidence["issues"].append("evidence_budget")
                return False
            selected[identifier] = {"id": identifier, "text": text, "roles": [], "spans": cell["spans"]}
            characters += len(text)
        if role not in selected[identifier]["roles"]:
            selected[identifier]["roles"].append(role)
        return True

    for identifier in roots:
        if not add(identifier, "value"):
            continue
        cell = cells[identifier]
        association = {}
        for name, role in RELATION_ROLES:
            association[name] = []
            for support_id in cell[name]:
                support = cells.get(support_id)
                if support is None or support["fragment_id"] != cell["fragment_id"]:
                    evidence["issues"].append("invalid_support_reference")
                elif add(support_id, role):
                    association[name].append(support_id)
        if not cell["column_header_cell_ids"]:
            evidence["issues"].append("unknown_column_header")
        if cell["column"] > 0 and not cell["row_header_cell_ids"]:
            evidence["issues"].append("unknown_row_header")
        evidence["associations"][identifier] = association
    for item in payload.get("related_evidence", []):
        if not _valid_spans(item["spans"], run["page_count"]):
            evidence["issues"].append("missing_related_location")
        elif characters + len(item["text"]) > max_characters:
            evidence["issues"].append("evidence_budget")
        else:
            evidence["related"].append(item)
            characters += len(item["text"])
    evidence["cells"] = list(selected.values())
    evidence["issues"] = list(dict.fromkeys(evidence["issues"]))
    evidence["status"] = "partial" if evidence["issues"] else "complete"
    evidence["parse_id"] = run_id
    evidence["table_id"] = meta["table_id"]


def table_evidence_text(hit: Hit, *, max_characters=12000) -> str:
    """Only saved raw evidence reaches the answer; embedding context is excluded."""
    evidence = hit.metadata.get("table_evidence")
    if not isinstance(evidence, dict):
        return ("表格支持证据暂不可用，行列表头和单位归属未核实；不能仅凭孤立数值作精确归属。"
                if hit.metadata.get("doc_type_kwd") == "table" else "")
    status = evidence.get("status")
    usage = evidence.get("quality", {}).get("usage", "reference_only")
    if usage == "blocked":
        return "该表的质量检查已阻断，禁止用于数值问答；请对照原件修复结构或来源问题。"
    if status == "unavailable":
        return "表格支持证据暂不可用，行列表头和单位归属未核实；不能仅凭孤立数值作精确归属。"
    cells = {c["id"]: c for c in evidence.get("cells", [])}

    def describe(cell):
        pages = "、".join(str(n) for n in sorted({s["page_no"] for s in cell["spans"]}))
        return f"「{cell['text']}」（第 {pages} 页）"

    lines = ["表格原文支持证据（补齐不代表已人工确认识别正确）："]
    if usage == "reference_only":
        lines.append("表格当前仅供原文参考，数值归属或识别质量尚需复核，不能作为已核实的精确结论。")
    used = len(lines[0])
    omitted = False

    def append(line):
        nonlocal used, omitted
        if used + len(line) + 1 > max_characters:
            omitted = True
        else:
            lines.append(line)
            used += len(line) + 1
    for identifier, relation in evidence.get("associations", {}).items():
        parts = ["命中单元格 " + describe(cells[identifier])]
        for name, label in (("row_header_cell_ids", "行标识"), ("column_header_cell_ids", "列表头路径"), ("unit_cell_ids", "单位来源格")):
            parts.append(label + "：" + (" → ".join(describe(cells[i]) for i in relation[name]) or "未确定"))
        append("；".join(parts))
    labels = {"title": "表题", "unit": "表级单位声明", "footnote": "表级注释"}
    for item in evidence.get("related", []):
        append(labels.get(item["role"], "表格关联原文") + "：" + describe(item))
    if omitted:
        lines.append("展示预算不足，部分支持证据未提供；不得推断缺失关联。")
    if status != "complete":
        lines.append("支持证据未完全补齐；缺失的表头、行标识或单位不得推断。")
    return "\n".join(lines)


def answer_reference_text(hit: Hit) -> str:
    supplement = table_evidence_text(hit)
    if hit.metadata.get("table_evidence", {}).get("quality", {}).get("usage") == "blocked":
        return supplement  # Keep the original hit visible in sources, exclude its numbers from the model.
    return hit.text + ("\n\n" + supplement if supplement else "")
