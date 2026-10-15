"""Cropped raster -> OCR PDF -> source positions, spatial fusion and indexing."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pymupdf

from app.pdf.chunking import build_chunks
from app.pdf.ocr import OcrResult, ocr_region
from app.pdf.parser import parse_pdf
from app.pdf.serialization import restore_pdf
from app.pdf.text_fusion import TextEvidence, fuse_text
from app.quality import check_pdf
from app.schemas.pdf import SourceSpan
from app.storage.pdf_repository import PdfRepository
from tests.pdf_regression.fixtures import make_pdf


def evidence(identifier, text, bbox, origin="ocr", page=1):
    return TextEvidence(identifier, text, bbox, SourceSpan(page, "", (.1, .2, .3, .4), identifier, "line"), origin)


class RegionOcrTests(unittest.TestCase):
    def test_crop_pixel_rounding_and_all_page_rotations_map_back_to_source(self):
        with pymupdf.open() as source:
            page = source.new_page(width=600, height=800)
            box = (60.3, 160.7, 180.8, 240.1)
            for rotation in (0, 90, 180, 270):
                with self.subTest(rotation=rotation):
                    page.set_rotation(rotation)
                    truth = {}
                    device = (pymupdf.Rect(box) * page.rotation_matrix * pymupdf.Matrix(200 / 72, 200 / 72)).irect

                    def recognize(pix, language):
                        self.assertEqual(language, "eng")
                        # OCR PDF's page size comes from raster DPI, not original PDF points.
                        width, height = pix.width * 72 / pix.xres, pix.height * 72 / pix.yres
                        self.assertLess(pix.width * pix.height, 120000)
                        with pymupdf.open() as engine_output:
                            output_page = engine_output.new_page(width=width, height=height)
                            output_page.insert_text((10, 25), "35.2", fontsize=11)
                            line = output_page.get_text("dict")["blocks"][0]["lines"][0]
                            bbox = line["bbox"]
                            displayed = pymupdf.Rect(bbox[0] + device.x0 * 72 / pix.xres,
                                                     bbox[1] + device.y0 * 72 / pix.yres,
                                                     bbox[2] + device.x0 * 72 / pix.xres,
                                                     bbox[3] + device.y0 * 72 / pix.yres)
                            truth["bbox"] = tuple((displayed * page.derotation_matrix) & pymupdf.Rect(box))
                            return engine_output.tobytes()

                    with patch("app.pdf.ocr._recognize_pixmap", side_effect=recognize):
                        result = ocr_region(page, box, "eng", 200)
                    self.assertEqual(result.blocks[0][1], "35.2")
                    for actual, expected in zip(result.blocks[0][0], truth["bbox"]):
                        self.assertAlmostEqual(actual, expected, places=4)
                    for actual, expected in zip(result.transform["crop_pdf_bbox"], box):
                        self.assertAlmostEqual(actual, expected, places=4)
                    self.assertEqual(result.transform["dpi"], 200)

    def test_native_table_area_is_masked_before_recognition(self):
        with pymupdf.open() as source:
            page = source.new_page(width=600, height=800)
            page.draw_rect((60, 160, 180, 240), color=(0, 0, 0), fill=(0, 0, 0))
            def recognize(pix, language):
                self.assertEqual(pix.pixel(pix.width // 2, pix.height // 2), (255, 255, 255))
                with pymupdf.open() as output:
                    output.new_page(width=pix.width * 72 / 200, height=pix.height * 72 / 200)
                    return output.tobytes()
            with patch("app.pdf.ocr._recognize_pixmap", side_effect=recognize):
                result = ocr_region(page, (60, 160, 180, 240), "eng", 200,
                                    exclude_regions=[(60, 160, 180, 240)])
            self.assertEqual(result.blocks, [])
            self.assertEqual(len(result.transform["excluded_pdf_regions"]), 1)

    def test_invalid_crop_and_dpi_do_not_call_engine(self):
        with pymupdf.open() as source:
            page = source.new_page(width=600, height=800)
            with patch("app.pdf.ocr._recognize_pixmap") as engine:
                for box, dpi in (((900, 900, 1000, 1000), 200), ((60, 160, 180, 240), 10)):
                    with self.assertRaises(ValueError):
                        ocr_region(page, box, "eng", dpi)
                engine.assert_not_called()

    def test_equal_numbers_at_different_positions_and_pages_are_not_deleted(self):
        left = evidence("left", "35.2", (60, 160, 100, 180), "native")
        duplicate = evidence("duplicate", "35.2", (61, 160, 101, 180))
        right = evidence("right", "35.2", (220, 160, 260, 180))
        other_page = evidence("page2", "35.2", (60, 160, 100, 180), page=2)
        result = fuse_text([left], [duplicate, right, other_page])
        self.assertEqual({item.id for item in result.selected}, {"left", "right", "page2"})
        self.assertEqual(len(result.duplicates), 1)

    def test_numeric_conflict_preserves_zero_and_alternative_evidence(self):
        native = evidence("native", "0", (60, 160, 100, 180), "native")
        ocr = evidence("ocr", "8", (61, 160, 101, 180))
        result = fuse_text([native], [ocr])
        self.assertEqual([item.text for item in result.selected], ["0"])
        self.assertTrue(result.conflicts[0]["numeric_conflict"])
        self.assertEqual(result.conflicts[0]["ocr"]["text"], "8")

    def test_corrupt_native_text_can_use_ocr_without_discarding_original(self):
        native = evidence("native", "bad\ufffd\ufffd", (60, 160, 160, 180), "native")
        ocr = evidence("ocr", "recovered", (61, 160, 161, 180))
        result = fuse_text([native], [ocr])
        self.assertEqual([item.text for item in result.selected], ["recovered"])
        self.assertEqual(result.conflicts[0]["native_or_previous"]["text"], "bad\ufffd\ufffd")

    def test_partially_overlapping_native_paragraph_is_not_indexed_again_in_image(self):
        block = {"type": 0, "bbox": (60, 100, 400, 240), "lines": [
            {"bbox": (60, 100, 400, 120), "spans": [{"text": "Reliable native paragraph.", "size": 11}]},
            {"bbox": (228, 205, 260, 225), "spans": [{"text": "35.2", "size": 11}]},
        ]}
        with patch.object(pymupdf.Page, "get_text", return_value={"blocks": deepcopy([block])}), \
             patch.object(pymupdf.Page, "find_tables", return_value=None), \
             patch("app.pdf.parser.ocr_region", return_value=OcrResult([((228, 205, 260, 225), "35.2")])):
            parsed = parse_pdf("overlap.pdf", make_pdf("hybrid_page"))
        self.assertEqual(sum(item.text.count("35.2") for item in parsed.elements), 1)
        image = next(item for item in parsed.elements if item.kind == "image")
        self.assertEqual(image.text, "")
        self.assertEqual(image.payload["ocr_text"], "35.2")
        self.assertEqual(len(image.payload["text_duplicates"]), 1)

    def test_corrupt_paragraph_uses_region_ocr_and_retains_review_evidence(self):
        block = {"type": 0, "bbox": (60, 100, 400, 120), "lines": [
            {"bbox": (60, 100, 400, 120), "spans": [{"text": "bad\ufffd\ufffd", "size": 11}]},
        ]}
        with patch.object(pymupdf.Page, "get_text", return_value={"blocks": deepcopy([block])}), \
             patch.object(pymupdf.Page, "find_tables", return_value=None), \
             patch("app.pdf.parser.ocr_region", return_value=OcrResult([((60, 100, 400, 120), "Recovered paragraph")])):
            parsed = parse_pdf("repaired.pdf", make_pdf("native_grid"))
        paragraph = next(item for item in parsed.elements if item.kind == "paragraph")
        self.assertEqual(paragraph.text, "Recovered paragraph")
        self.assertEqual(paragraph.payload["native_text"], "bad\ufffd\ufffd")
        self.assertTrue(paragraph.payload["text_conflicts"])
        route = parsed.page_inspections[0].routes[0]
        self.assertIn("region_ocr", route.executed_operations)
        self.assertIn("review_text_layer", route.deferred_operations)
        self.assertTrue(any(issue.code == "ocr_conflict" for issue in check_pdf(parsed, build_chunks(parsed)).issues))

    def test_scan_paragraphs_have_independent_positions_and_no_background_duplicate(self):
        blocks = [((68, 205, 180, 225), "First scanned paragraph."),
                  ((228, 245, 380, 265), "Second scanned paragraph.")]
        with patch("app.pdf.parser.ocr_region", return_value=OcrResult(blocks)):
            parsed = parse_pdf("scan.pdf", make_pdf("scanned_table"))
        paragraphs = [item for item in parsed.elements if item.kind == "paragraph"]
        self.assertEqual(len(paragraphs), 2)
        self.assertEqual(paragraphs[0].span.bbox, (68 / 600, 205 / 800, 180 / 600, 225 / 800))
        self.assertEqual(next(item for item in parsed.elements if item.kind == "image").text, "")
        chunks = build_chunks(parsed)
        text = "\n".join(chunk.page_content for chunk in chunks)
        self.assertEqual(text.count("First scanned paragraph."), 1)
        self.assertEqual(text.count("Second scanned paragraph."), 1)
        mappings = [item for chunk in chunks for item in json.loads(chunk.metadata["pdf_serialized_map"])]
        self.assertTrue(all(item["role"] == "ocr" and item["span"]["granularity"] == "line" for item in mappings))

    def test_ocr_failure_is_isolated_to_one_image(self):
        with pymupdf.open(stream=make_pdf("hybrid_page"), filetype="pdf") as source:
            image = source[0].get_pixmap().tobytes("png")
        with pymupdf.open() as source:
            page = source.new_page(width=600, height=800)
            page.insert_text((60, 60), "Reliable native paragraph.")
            page.insert_image((60, 160, 240, 300), stream=image, keep_proportion=False)
            page.insert_image((300, 160, 480, 300), stream=image, keep_proportion=False)
            content = source.tobytes()
        with patch("app.pdf.parser.ocr_region", side_effect=[RuntimeError("No tessdata specified"),
                   OcrResult([((310, 180, 450, 200), "Recovered second image")])]) as engine:
            parsed = parse_pdf("two-images.pdf", content)
        self.assertEqual(engine.call_count, 2)
        images = [item for item in parsed.elements if item.kind == "image"]
        self.assertEqual([item.payload["ocr_status"] for item in images], ["unavailable", "recognized"])
        self.assertEqual(images[1].text, "Recovered second image")
        self.assertTrue(any("Reliable native paragraph." in item.text for item in parsed.elements))
        route = next(route for route in parsed.page_inspections[0].routes if route.region_id == images[0].id)
        self.assertEqual(route.failed_operations, ["region_ocr"])
        self.assertNotIn("region_ocr", route.deferred_operations)

    def test_image_line_sources_and_conflict_survive_chunks_quality_and_storage(self):
        with pymupdf.open(stream=make_pdf("hybrid_page"), filetype="pdf") as source:
            source[0].insert_text((228, 225), "35.2")
            content = source.tobytes()
        with patch("app.pdf.parser.ocr_region", return_value=OcrResult([((228, 212, 250, 228), "40.8"),
                                                                      ((388, 252, 410, 270), "0")])):
            parsed = parse_pdf("conflict.pdf", content)
        image = next(item for item in parsed.elements if item.kind == "image")
        self.assertIn("35.2", image.text)
        self.assertNotIn("40.8", image.text)
        self.assertIn("40.8", image.payload["ocr_text"])
        self.assertTrue(image.payload["text_conflicts"][0]["numeric_conflict"])
        chunks = build_chunks(parsed)
        report = check_pdf(parsed, chunks)
        self.assertTrue(any(issue.code == "ocr_conflict" for issue in report.issues))
        mapping = next(item for chunk in chunks for item in json.loads(chunk.metadata["pdf_serialized_map"])
                       if item.get("role") == "ocr")
        self.assertEqual(mapping["span"]["bbox"], [388 / 600, 252 / 800, 410 / 600, 270 / 800])
        root = Path(__file__).resolve().parents[1] / ".documents" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root, ignore_cleanup_errors=True) as directory:
            repository = PdfRepository(directory)
            repository.save(parsed, content, chunks)
            data = repository.get_run(chunks[0].metadata["pdf_parse_id"])
            assert data is not None
            restored = restore_pdf(data)
        restored_image = next(item for item in restored.elements if item.kind == "image")
        self.assertEqual(restored_image.payload["text_conflicts"][0]["ocr"]["text"], "40.8")


if __name__ == "__main__":
    unittest.main()
