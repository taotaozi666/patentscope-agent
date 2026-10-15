"""Docling adapter; single unrotated page input avoids ambiguous cell page IDs."""

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import pymupdf

from app.schemas.table import TableCell, TableFragment
from ..geometry import source_span
from .base import TableBackendResult, TableOptions


DOCLING_ADAPTER_VERSION = "docling-table-adapter-v1"


class TableBackendUnavailable(RuntimeError):
    pass


class UnsupportedTableOutput(ValueError):
    pass


def package_version() -> str:
    try:
        return version("docling")
    except PackageNotFoundError:
        return "unavailable"


def models_identity(directory: str) -> str:
    if not directory:
        return "unconfigured"
    root = Path(directory)
    if not root.is_dir():
        return "missing"
    files = [(str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
             for path in sorted(root.rglob("*")) if path.is_file()]
    return sha256(json.dumps(files).encode()).hexdigest()


def _bbox(value: dict, height: float) -> tuple[float, float, float, float]:
    if value.get("coord_origin") == "TOPLEFT":
        box = (value["l"], value["t"], value["r"], value["b"])
    elif value.get("coord_origin") == "BOTTOMLEFT":
        box = (value["l"], height - value["t"], value["r"], height - value["b"])
    else:
        raise UnsupportedTableOutput("Docling 输出使用未知坐标原点")
    if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in box) or box[0] >= box[2] or box[1] >= box[3]:
        raise UnsupportedTableOutput("Docling 表格来源区域无效")
    return box


def adapt_document(document: dict, page: pymupdf.Page, *, backend_version: str,
                   text_origin: str) -> list[TableFragment]:
    if document.get("schema_name") != "DoclingDocument" or str(document.get("version", "")).split(".")[0] != "1":
        raise UnsupportedTableOutput("不支持的 Docling 文档 schema")
    pages = document.get("pages", {})
    if set(pages) != {"1"}:
        raise UnsupportedTableOutput("Docling 表格适配只接受独立单页输出")
    bounds = page.rect * page.derotation_matrix
    size = pages["1"]["size"]
    if abs(size["width"] - bounds.width) > 1 or abs(size["height"] - bounds.height) > 1:
        raise UnsupportedTableOutput("Docling 页面尺寸与原 PDF 不一致，不能可靠定位")
    assert page.number is not None
    results = []
    for index, table in enumerate(document.get("tables", [])):
        provenance = table.get("prov", [])
        if len(provenance) != 1 or provenance[0]["page_no"] != 1:
            raise UnsupportedTableOutput("Docling 表格存在多页或缺失来源")
        identifier = f"p{page.number + 1}_docling_table{index}"
        table_box = _bbox(provenance[0]["bbox"], bounds.height)
        if table_box[0] < -.5 or table_box[1] < -.5 or table_box[2] > bounds.width + .5 or table_box[3] > bounds.height + .5:
            raise UnsupportedTableOutput("Docling 表格来源越出页面")
        data = table["data"]
        rows, columns = data["num_rows"], data["num_cols"]
        cells, occupied, header_rows = [], set(), set()
        for raw in data["table_cells"]:
            if "ref" in raw:
                raise UnsupportedTableOutput("Docling 富单元格引用尚未适配，不能静默丢弃内容")
            row, column = raw["start_row_offset_idx"], raw["start_col_offset_idx"]
            rs, cs = raw["row_span"], raw["col_span"]
            if row + rs != raw["end_row_offset_idx"] or column + cs != raw["end_col_offset_idx"]:
                raise UnsupportedTableOutput("Docling 跨度与行列区间不一致")
            cell_id = f"{identifier}_r{row}c{column}"
            box = _bbox(raw["bbox"], bounds.height) if raw.get("bbox") else None
            if box and (box[0] < -.5 or box[1] < -.5 or box[2] > bounds.width + .5 or box[3] > bounds.height + .5):
                raise UnsupportedTableOutput("Docling 单元格位置越出页面")
            text = raw["text"]
            cell = TableCell(cell_id, identifier, row, column, text,
                state="text" if text.strip() else "blank" if box else "unknown",
                row_span=rs, col_span=cs, text_origin=text_origin,
                spans=[source_span(page, box, cell_id, "cell_content")] if box else [])
            cells.append(cell)
            for r in range(row, row + rs):
                for c in range(column, column + cs):
                    if (r, c) in occupied:
                        raise UnsupportedTableOutput("Docling 表格单元格覆盖冲突")
                    occupied.add((r, c))
            if raw.get("column_header") is True:
                header_rows.update(range(row, row + rs))
        results.append(TableFragment(identifier, source_span(page, table_box, identifier), rows, columns,
            cells, header_rows=sorted(header_rows), backend="docling", backend_version=backend_version,
            structure_status="complete" if occupied == {(r, c) for r in range(rows) for c in range(columns)} else "partial",
            provenance_status="complete" if all(cell.spans for cell in cells) else "partial",
            raw_output={"adapter_version": DOCLING_ADAPTER_VERSION, "document_version": document["version"], "table": table}))
    return results


class DoclingBackend:
    def __init__(self, options: TableOptions, work_directory: str):
        self.options = options
        self.work_directory = Path(work_directory)

    def parse_page(self, page: pymupdf.Page, *, ocr: bool) -> TableBackendResult:
        if package_version() == "unavailable":
            raise TableBackendUnavailable("Docling 可选依赖未安装，请执行 uv sync --extra tables")
        models = Path(self.options.models_directory) if self.options.models_directory else None
        if models is None or not models.is_dir():
            raise TableBackendUnavailable("Docling 本地模型目录未配置或不存在")
        if not any(models.rglob("tm_config.json")):
            raise TableBackendUnavailable("Docling 目录缺少 TableFormer 配置或权重，请先下载完整模型")
        self.work_directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.work_directory, ignore_cleanup_errors=True) as directory:
            source, output = Path(directory) / "page.pdf", Path(directory) / "result.json"
            with pymupdf.open() as single:
                single.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
                single[0].set_rotation(0)
                source.write_bytes(single.tobytes())
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3]) + os.pathsep + env.get("PYTHONPATH", "")
            command = [sys.executable, "-m", "app.pdf.tables.docling_worker", str(source), str(output),
                       str(models.resolve()), self.options.mode, "1" if ocr else "0", self.options.ocr_language]
            result = subprocess.run(command, env=env, capture_output=True, timeout=120,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if result.returncode or not output.is_file():
                detail = result.stderr.decode("utf-8", errors="replace")[-1200:]
                raise TableBackendUnavailable(f"Docling 本地推理失败：{detail}")
            payload = json.loads(output.read_text(encoding="utf-8"))
        if payload["status"] not in {"success", "partial_success"}:
            raise TableBackendUnavailable("Docling 转换未成功")
        fragments = adapt_document(payload["document"], page, backend_version=package_version(),
                                   text_origin="unknown" if ocr else "native")
        warnings = ["Docling 部分转换未完成，表格需复核"] if payload["status"] == "partial_success" else []
        return TableBackendResult(fragments, warnings)
