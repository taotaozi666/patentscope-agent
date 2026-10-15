"""页面诊断、区域路线、旋转来源和历史资产兼容的回归验证。"""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pymupdf
from app.pdf.ocr import OcrResult

from app.pdf.chunking import build_chunks
from app.pdf.inspection import InspectionPolicy, inspect_page, _union_area
from app.pdf.parser import PdfParseError, parse_pdf
from app.pdf.routing import route_regions
from app.pdf.serialization import restore_pdf
from app.pdf.service import prepare_pdf
from app.schemas import PageInspection, RegionInspection, RegionRoute, Settings
from app.storage.pdf_repository import PdfRepository
from tests.pdf_regression.evaluation import evaluate, load_expectations
from tests.pdf_regression.fixtures import make_pdf


class PdfInspectionTests(unittest.TestCase):
    def test_native_text_and_table_use_separate_routes_without_ocr(self):
        with patch("app.pdf.parser.ocr_region", side_effect=AssertionError("unexpected OCR")) as ocr:
            parsed = parse_pdf("native.pdf", make_pdf("native_grid"))
        ocr.assert_not_called()
        diagnosis = parsed.page_inspections[0]
        self.assertEqual(diagnosis.text_source, "native")
        self.assertEqual(diagnosis.metrics["table_candidate_count"], 1)
        self.assertEqual({region.role for region in diagnosis.regions}, {"text", "table"})
        for region, route in zip(diagnosis.regions, diagnosis.routes):
            self.assertEqual(region.id, route.region_id)
            expected = "native_table" if region.role == "table" else "native_text"
            self.assertEqual(route.planned_operations, [expected])
            self.assertEqual(route.executed_operations, [expected])

    def test_hybrid_page_preserves_native_route_and_disabled_image_ocr_plan(self):
        parsed = parse_pdf("hybrid.pdf", make_pdf("hybrid_page"), ocr_images=False)
        diagnosis = parsed.page_inspections[0]
        self.assertEqual(diagnosis.text_source, "hybrid")
        self.assertEqual(diagnosis.classification_status, "candidate")
        region = next(region for region in diagnosis.regions if region.role == "image")
        self.assertEqual(region.span.bbox, (.1, .2, .9, .35))
        route = next(route for route in diagnosis.routes if route.region_id == region.id)
        self.assertIn("region_ocr", route.planned_operations)
        self.assertIn("region_ocr", route.disabled_operations)
        self.assertNotIn("region_ocr", route.executed_operations)
        self.assertEqual(route.executed_operations, ["preserve_image"])
        self.assertIn("layout_analysis", route.deferred_operations)
        self.assertTrue(any(route.executed_operations == ["native_text"] for route in diagnosis.routes))

    def test_region_ocr_is_reported_as_executed_with_layout_deferred(self):
        with patch("app.pdf.parser.ocr_region", return_value=OcrResult([((228, 205, 370, 235), "35.2")])):
            parsed = parse_pdf("hybrid.pdf", make_pdf("hybrid_page"))
        route = next(route for route in parsed.page_inspections[0].routes
                     if "region_ocr" in route.planned_operations)
        self.assertNotIn("legacy_page_ocr", route.executed_operations)
        self.assertNotIn("region_ocr", route.deferred_operations)
        self.assertIn("region_ocr", route.executed_operations)

    def test_rejected_scan_retains_diagnosis_and_evaluation_checks_source(self):
        with self.assertRaises(PdfParseError) as failure:
            parse_pdf("scan.pdf", make_pdf("scanned_table"), ocr_images=False)
        diagnosis = failure.exception.page_inspections[0]
        self.assertEqual(diagnosis.text_source, "scanned")
        self.assertEqual(diagnosis.metrics["image_coverage_ratio"], 1.0)
        self.assertIn("page_ocr", diagnosis.routes[0].disabled_operations)
        expected = next(case for case in load_expectations()["cases"] if case["id"] == "scanned_table")
        report = evaluate(None, expected, error=str(failure.exception),
                          page_inspections=failure.exception.page_inspections)
        self.assertEqual(report["metrics"]["page_source_classification"], {"matched": 1, "expected": 1})
        self.assertFalse(report["measured_checks_passed"])

    def test_mocked_scan_records_whole_page_ocr_and_deferred_layout(self):
        with patch("app.pdf.parser.ocr_region", return_value=OcrResult([((68, 205, 210, 235), "Ex1")])):
            parsed = parse_pdf("scan.pdf", make_pdf("scanned_table"))
        route = parsed.page_inspections[0].routes[0]
        self.assertEqual(route.executed_operations, ["preserve_image", "page_ocr"])
        self.assertEqual(route.deferred_operations, ["layout_analysis"])

    def test_rotation_keeps_raw_crop_box_and_normalizes_display_position(self):
        with pymupdf.open(stream=make_pdf("hybrid_page"), filetype="pdf") as pdf:
            pdf[0].set_rotation(90)
            parsed = parse_pdf("rotated.pdf", pdf.tobytes(), ocr_images=False)
        diagnosis = parsed.page_inspections[0]
        self.assertEqual((diagnosis.width, diagnosis.height, diagnosis.rotation), (800, 600, 90))
        region = next(region for region in diagnosis.regions if region.role == "image")
        self.assertEqual(region.pdf_bbox, (60, 160, 540, 280))
        self.assertEqual(region.span.bbox, (.65, .1, .8, .9))
        image = next(element for element in parsed.elements if element.kind == "image")
        self.assertEqual(image.span.bbox, region.span.bbox)

    def test_sparse_text_blank_and_vector_figure_are_distinguished(self):
        with pymupdf.open() as pdf:
            sparse = pdf.new_page(width=600, height=800)
            sparse.insert_text((60, 60), "A")
            blank = pdf.new_page(width=600, height=800)
            vector = pdf.new_page(width=600, height=800)
            vector.draw_rect((60, 100, 300, 300))
            # 新增页面可能使旧 Page 对象失效，按索引重新取页。
            results = [inspect_page(pdf[index]).diagnosis for index in range(3)]
        self.assertEqual([result.text_source for result in results], ["native", "empty", "unknown"])
        self.assertEqual(results[1].regions, [])
        routes = route_regions(results[2], ocr_enabled=True)
        self.assertEqual(routes[0].planned_operations, ["preserve_region", "review_layout"])

    def test_corrupt_text_layer_routes_only_affected_region_to_review(self):
        good = {"type": 0, "bbox": (60, 60, 400, 80),
                "lines": [{"spans": [{"text": "Reliable paragraph", "size": 11}]}]}
        bad = {"type": 0, "bbox": (60, 100, 400, 120),
               "lines": [{"spans": [{"text": "bad\ufffd\ufffd", "size": 11}]}]}
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            with patch.object(pymupdf.Page, "get_text", return_value={"blocks": deepcopy([good, bad])}), \
                 patch.object(pymupdf.Page, "find_tables", return_value=None):
                diagnosis = inspect_page(page).diagnosis
        self.assertEqual(diagnosis.text_source, "unreliable_text_layer")
        routes = route_regions(diagnosis, ocr_enabled=True)
        self.assertEqual(routes[0].planned_operations, ["native_text"])
        self.assertEqual(routes[1].planned_operations, ["preserve_native_text", "region_ocr", "review_text_layer"])
        self.assertFalse(diagnosis.metrics["render_alignment_verified"])

    def test_table_detection_failure_preserves_native_text_and_reason(self):
        with patch.object(pymupdf.Page, "find_tables", side_effect=RuntimeError("detector failure")) as detector:
            parsed = parse_pdf("native.pdf", make_pdf("native_grid"), ocr_images=False)
        detector.assert_called_once()
        diagnosis = parsed.page_inspections[0]
        self.assertEqual(diagnosis.metrics["table_detection"], "failed")
        self.assertTrue(diagnosis.warnings)
        self.assertTrue(any("35.2" in element.text for element in parsed.elements))
        self.assertFalse(any(region.role == "table" for region in diagnosis.regions))

    def test_diagnostics_survive_repository_and_json_roundtrip(self):
        content = make_pdf("hybrid_page")
        parsed = parse_pdf("hybrid.pdf", content, ocr_images=False)
        chunks = build_chunks(parsed)
        root = Path(__file__).resolve().parents[1] / ".documents" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root, ignore_cleanup_errors=True) as directory:
            repository = PdfRepository(directory)
            repository.save(parsed, content, chunks)
            data = repository.get_run(chunks[0].metadata["pdf_parse_id"])
            assert data is not None
            restored = restore_pdf(data)
        self.assertEqual(restored.page_inspections, parsed.page_inspections)
        self.assertIsInstance(restored.page_inspections[0], PageInspection)
        self.assertIsInstance(restored.page_inspections[0].regions[0], RegionInspection)
        self.assertIsInstance(restored.page_inspections[0].routes[0], RegionRoute)
        self.assertEqual(restore_pdf(json.loads(json.dumps(asdict(restored)))).page_inspections,
                         restored.page_inspections)

    def test_legacy_assets_without_diagnostics_are_readable_without_guessing(self):
        data = asdict(parse_pdf("native.pdf", make_pdf("native_grid"), ocr_images=False))
        data.pop("page_inspections")
        self.assertEqual(restore_pdf(data).page_inspections, [])

    def test_failed_scan_saves_original_and_diagnosis_without_fake_parse_run(self):
        content = make_pdf("scanned_table")
        root = Path(__file__).resolve().parents[1] / ".documents" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root, ignore_cleanup_errors=True) as directory:
            settings = Settings(api_key="", chat_model="", embedding_model="",
                                pdf_directory=directory, pdf_image_ocr=False)
            with self.assertRaises(PdfParseError):
                prepare_pdf("scan.pdf", content, settings)
            paths = list((Path(directory) / "diagnostics").glob("*.json"))
            self.assertEqual(len(paths), 1)
            snapshot = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertEqual(snapshot["status"], "parse_failed")
            self.assertEqual(snapshot["inspected_pages"], [1])
            self.assertTrue(snapshot["inspection_complete"])
            self.assertEqual(snapshot["page_inspections"][0]["text_source"], "scanned")
            repository = PdfRepository(directory)
            self.assertEqual(repository.read_original(snapshot["content_hash"]), content)
            self.assertEqual(repository.list_runs(), [])
            self.assertFalse(list((Path(directory) / "diagnostics").glob("*.tmp")))

    def test_early_text_failure_marks_partial_document_inspection(self):
        content = make_pdf("continued_table")
        bad = {"type": 0, "bbox": (60, 100, 400, 120),
               "lines": [{"spans": [{"text": "bad\ufffd\ufffd", "size": 11}]}]}
        root = Path(__file__).resolve().parents[1] / ".documents" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root, ignore_cleanup_errors=True) as directory:
            settings = Settings(api_key="", chat_model="", embedding_model="",
                                pdf_directory=directory, pdf_image_ocr=False)
            with patch.object(pymupdf.Page, "get_text", return_value={"blocks": deepcopy([bad])}), \
                 patch.object(pymupdf.Page, "find_tables", return_value=None), \
                 self.assertRaises(PdfParseError):
                prepare_pdf("corrupt-text.pdf", content, settings)
            path = next((Path(directory) / "diagnostics").glob("*.json"))
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(snapshot["page_count"], 2)
            self.assertEqual(snapshot["inspected_pages"], [1])
            self.assertFalse(snapshot["inspection_complete"])

    def test_policy_and_overlapping_image_union(self):
        self.assertEqual(_union_area([(0, 0, 100, 100), (50, 0, 150, 100)]), 15000)
        for kwargs in ({"replacement_ratio": 0}, {"block_image_overlap": float("nan")},
                       {"vector_ocr_path_count": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                InspectionPolicy(**kwargs)


if __name__ == "__main__":
    unittest.main()
