"""Table geometry, Docling schema mapping, fallback and configuration identity."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pymupdf

from app.config import get_settings
from app.pdf.chunking import build_chunks
from app.pdf.inspection import inspect_page
from app.pdf.parser import parse_pdf
from app.pdf.serialization import restore_pdf
from app.pdf.service import prepare_pdf
from app.schemas import Settings
from app.pdf.tables.base import TableOptions, to_element
from app.pdf.tables.docling import DoclingBackend, UnsupportedTableOutput, adapt_document
from app.pdf.tables.service import backend_identity
from tests.pdf_regression.evaluation import evaluate, load_expectations
from tests.pdf_regression.fixtures import make_pdf


def docling_payload(expected, origin="TOPLEFT"):
    """Protocol specimen with explicit spans/flags, independent of backend output."""
    def box(values):
        left, top, right, bottom = (values[0] * 600, values[1] * 800, values[2] * 600, values[3] * 800)
        return {"l": left, "t": top if origin == "TOPLEFT" else 800 - top,
                "r": right, "b": bottom if origin == "TOPLEFT" else 800 - bottom, "coord_origin": origin}
    tables = []
    for fragment in expected["fragments"][:1]:
        tables.append({"prov": [{"page_no": 1, "bbox": box(fragment["bbox"]), "charspan": [0, 0]}],
            "data": {"num_rows": fragment["row_count"], "num_cols": fragment["column_count"], "table_cells": [
                {"text": cell["raw_text"], "bbox": box(cell["bbox"]), "row_span": cell["row_span"],
                 "col_span": cell["col_span"], "start_row_offset_idx": cell["row"],
                 "end_row_offset_idx": cell["row"] + cell["row_span"], "start_col_offset_idx": cell["column"],
                 "end_col_offset_idx": cell["column"] + cell["col_span"],
                 "column_header": cell["row"] in fragment["header_rows"]} for cell in fragment["cells"]]}})
    return {"schema_name": "DoclingDocument", "version": "1.10.0",
            "pages": {"1": {"size": {"width": 600, "height": 800}}}, "tables": tables}


class TableBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = {item["id"]: item for item in load_expectations()["cases"]}
        cls.root = Path(__file__).resolve().parents[1] / ".documents" / "tests"
        cls.root.mkdir(parents=True, exist_ok=True)

    def test_native_geometry_and_borderless_text_strategy_match_fixed_gold(self):
        for case in ("native_grid", "multi_header", "merged_cells", "borderless"):
            with self.subTest(case=case):
                parsed = parse_pdf(f"{case}.pdf", make_pdf(case), ocr_images=False)
                report = evaluate(parsed, self.gold[case])
                self.assertTrue(report["measured_checks_passed"], report)
                self.assertEqual(len(parsed.table_fragments), 1)
                self.assertEqual(len(parsed.logical_tables), 1)
                fragment = parsed.table_fragments[0]
                self.assertEqual(fragment.structure_status, "complete")
                if case == "multi_header":
                    cell = next(cell for cell in fragment.cells if cell.raw_text == "Properties")
                    self.assertEqual(cell.col_span, 2)
                if case == "merged_cells":
                    cell = next(cell for cell in fragment.cells if cell.raw_text == "Ex1")
                    self.assertEqual(cell.row_span, 2)
                if case == "borderless":
                    self.assertEqual(fragment.backend, "pymupdf-text")
                    self.assertTrue(any("Native paragraph" in item.text for item in parsed.elements if item.kind == "paragraph"))
                    self.assertEqual([cell.raw_text for cell in fragment.cells].count("0"), 1)

    def test_docling_multilevel_headers_and_coordinate_origins_map_to_same_source(self):
        with pymupdf.open(stream=make_pdf("multi_header"), filetype="pdf") as pdf:
            for origin in ("TOPLEFT", "BOTTOMLEFT"):
                with self.subTest(origin=origin):
                    fragments = adapt_document(docling_payload(self.gold["multi_header"], origin), pdf[0],
                                               backend_version="fixture", text_origin="native")
                    fragment = fragments[0]
                    self.assertEqual(fragment.header_rows, [0, 1])
                    self.assertEqual(fragment.span.bbox, (.1, .2, .9, .4))
                    cell = next(cell for cell in fragment.cells if cell.raw_text == "Properties")
                    self.assertEqual(cell.col_span, 2)
                    self.assertEqual(cell.spans[0].bbox, (.3, .2, .7, .25))
                    projected = to_element(fragment, "fixture")
                    self.assertEqual(projected.payload["rows"][0].count("Properties"), 1)

    def test_unknown_schema_bad_span_and_multiple_page_provenance_are_rejected(self):
        original = docling_payload(self.gold["multi_header"])
        broken = []
        future = deepcopy(original)
        future["version"] = "2.0"
        broken.append(future)
        wrong = deepcopy(original)
        wrong["tables"][0]["data"]["table_cells"][0]["row_span"] = 1
        broken.append(wrong)
        multipage = deepcopy(original)
        multipage["tables"][0]["prov"].append(deepcopy(multipage["tables"][0]["prov"][0]))
        broken.append(multipage)
        coordinates = deepcopy(original)
        coordinates["tables"][0]["prov"][0]["bbox"]["coord_origin"] = "UNKNOWN"
        broken.append(coordinates)
        with pymupdf.open(stream=make_pdf("multi_header"), filetype="pdf") as pdf:
            for payload in broken:
                with self.subTest(payload=payload["version"]), self.assertRaises(UnsupportedTableOutput):
                    adapt_document(payload, pdf[0], backend_version="fixture", text_origin="native")

    def test_docling_missing_dependency_falls_back_and_strict_mode_fails(self):
        with patch("app.pdf.tables.docling.package_version", return_value="unavailable"):
            parsed = parse_pdf("merged.pdf", make_pdf("merged_cells"), table_backend="auto", ocr_images=False)
            self.assertEqual(parsed.table_fragments[0].backend, "pymupdf-lines")
            attempts = parsed.page_inspections[0].metrics["table_backends"]
            self.assertEqual(attempts[-1]["status"], "failed")
            self.assertTrue(any("回退" in warning for warning in parsed.warnings))
            with self.assertRaisesRegex(ValueError, "关闭回退"):
                parse_pdf("merged.pdf", make_pdf("merged_cells"), table_backend="docling",
                          table_fallback=False, ocr_images=False)

    def test_docling_worker_receives_isolated_page_offline_flags_and_output_is_integrated(self):
        payload = docling_payload(self.gold["scanned_table"])
        with TemporaryDirectory(dir=self.root, ignore_cleanup_errors=True) as directory:
            models = Path(directory) / "models"
            models.mkdir()
            (models / "tm_config.json").write_text("{}", encoding="utf-8")
            def convert(command, **kwargs):
                self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
                self.assertEqual(kwargs["env"]["TRANSFORMERS_OFFLINE"], "1")
                self.assertEqual(kwargs["timeout"], 120)
                with pymupdf.open(command[3]) as single:
                    self.assertEqual(len(single), 1)
                    self.assertEqual(single[0].rotation, 0)
                Path(command[4]).write_text(json.dumps({"status": "success", "document": payload}), encoding="utf-8")
                return SimpleNamespace(returncode=0, stderr=b"")
            with patch("app.pdf.tables.docling.package_version", return_value="fixture"), \
                 patch("app.pdf.tables.docling.subprocess.run", side_effect=convert) as worker:
                parsed = parse_pdf("scan.pdf", make_pdf("scanned_table"), table_backend="docling",
                    table_models_directory=str(models), table_work_directory=directory, ocr_images=False)
            worker.assert_called_once()
            self.assertEqual(parsed.table_fragments[0].backend, "docling")
            table_route = next(route for route in parsed.page_inspections[0].routes if "docling_table" in route.planned_operations)
            self.assertIn("docling_table", table_route.executed_operations)
            self.assertTrue(evaluate(parsed, self.gold["scanned_table"])["measured_checks_passed"])
            chunks = build_chunks(parsed)
            self.assertTrue(chunks)
            restored = restore_pdf(json.loads(json.dumps(asdict(parsed))))
            self.assertEqual(restored.table_fragments, parsed.table_fragments)

    def test_backend_options_are_validated_and_identity_tracks_models(self):
        with TemporaryDirectory(dir=self.root, ignore_cleanup_errors=True) as directory:
            path = Path(directory) / ".env"
            settings = get_settings(path, {"PDF_TABLE_BACKEND": "auto", "PDF_TABLE_MODE": "fast",
                "PDF_TABLE_MODELS_DIRECTORY": "models", "PDF_TABLE_FALLBACK": "false"})
            self.assertEqual(settings.pdf_table_backend, "auto")
            self.assertEqual(settings.pdf_table_mode, "fast")
            self.assertEqual(settings.pdf_table_models_directory, str(Path(directory) / "models"))
            self.assertFalse(settings.pdf_table_fallback)
            options = TableOptions("auto", settings.pdf_table_models_directory)
            before = backend_identity(options)
            models = Path(options.models_directory)
            models.mkdir()
            (models / "config.json").write_text("{}", encoding="utf-8")
            self.assertNotEqual(backend_identity(options), before)
            self.assertNotEqual(backend_identity(options), backend_identity(replace(options, mode="fast")))
            for values in ({"PDF_TABLE_BACKEND": "invalid"}, {"PDF_TABLE_MODE": "invalid"}):
                with self.assertRaises(ValueError):
                    get_settings(path, values)

    def test_strict_backend_failure_saves_diagnosis_and_original(self):
        with TemporaryDirectory(dir=self.root, ignore_cleanup_errors=True) as directory:
            settings = Settings(api_key="", chat_model="", embedding_model="", pdf_directory=directory,
                                pdf_table_backend="docling", pdf_table_fallback=False, pdf_image_ocr=False)
            with patch("app.pdf.tables.docling.package_version", return_value="unavailable"), \
                 self.assertRaisesRegex(ValueError, "关闭回退"):
                prepare_pdf("strict.pdf", make_pdf("merged_cells"), settings)
            snapshot = json.loads(next((Path(directory) / "diagnostics").glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(snapshot["page_inspections"][0]["metrics"]["table_backends"][0]["status"], "failed")
            self.assertTrue((Path(directory) / "originals" / f"{snapshot['content_hash']}.pdf").exists())


if __name__ == "__main__":
    unittest.main()
