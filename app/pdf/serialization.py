"""恢复解析结构，兼容没有专利字段或统一表格字段的旧记录。"""

from app.schemas.pdf import ParsedPdf, PdfElement, SourceSpan, SemanticObject
from app.schemas.patent import PatentClaim, PatentField, PatentSection, PatentStructure
from app.schemas.table import TABLE_SCHEMA_VERSION, TableCell, TableFragment, LogicalTable
from .tables.legacy import adapt_basic_grid
from app.schemas.inspection import PageInspection, RegionInspection, RegionRoute


def restore_pdf(data: dict) -> ParsedPdf:
    def span(value: dict) -> SourceSpan:
        return SourceSpan(**{**value, "bbox": tuple(value["bbox"])})

    version = data.get("table_schema_version", 0)
    if type(version) is not int or version not in {0, TABLE_SCHEMA_VERSION}:
        raise ValueError(f"不支持的表格结构版本：{version}")

    patent = data.get("patent")
    structure = None
    if patent:
        structure = PatentStructure(
            extractor_version=patent["extractor_version"],
            sections=[PatentSection(**{**item, "spans": [span(value) for value in item["spans"]]}) for item in patent["sections"]],
            claims=[PatentClaim(**{**item, "spans": [span(value) for value in item["spans"]]}) for item in patent["claims"]],
            warnings=patent["warnings"],
        )
        for name in ("title", "publication_number"):
            item = patent.get(name)
            if item:
                setattr(structure, name, PatentField(item["value"], [span(value) for value in item["spans"]]))
    elements = [PdfElement(**{**item, "span": span(item["span"])}) for item in data["elements"]]
    objects = [SemanticObject(**{**item, "spans": [span(value) for value in item["spans"]]}) for item in data.get("objects", [])]
    if "table_fragments" in data:
        fragments = [TableFragment(**{
            **item, "span": span(item["span"]),
            "cells": [TableCell(**{**cell, "spans": [span(value) for value in cell.get("spans", [])]})
                      for cell in item.get("cells", [])],
        }) for item in data["table_fragments"]]
    else:
        # 只增加内存中的兼容视图，不改旧资产、解析版本或原始 payload。
        fragments = [adapt_basic_grid(element, data["parser_version"])
                     for element in elements if element.kind == "table"]
    tables = [LogicalTable(**{
        **item, "unit_spans": [span(value) for value in item.get("unit_spans", [])],
        "title_spans": [span(value) for value in item.get("title_spans", [])],
        "footnote_spans": [span(value) for value in item.get("footnote_spans", [])],
    }) for item in data.get("logical_tables", [])]
    inspections = [PageInspection(**{
        **item,
        "regions": [RegionInspection(**{**region, "span": span(region["span"]),
                                       "pdf_bbox": tuple(region["pdf_bbox"])})
                    for region in item.get("regions", [])],
        "routes": [RegionRoute(**route) for route in item.get("routes", [])],
    }) for item in data.get("page_inspections", [])]
    return ParsedPdf(**{**data, "elements": elements, "patent": structure, "objects": objects,
                        "table_schema_version": TABLE_SCHEMA_VERSION, "table_fragments": fragments,
                        "logical_tables": tables, "page_inspections": inspections})
