"""Export reproducible PDF specimens and measure current parsing gaps offline."""

import argparse
import json
from pathlib import Path
import sys
from dataclasses import asdict

# Support direct execution from an uninstalled source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pdf.parser import parse_pdf
from app.pdf.chunking import build_chunks
from app.quality import check_pdf
from tests.pdf_regression.evaluation import evaluate, load_expectations
from tests.pdf_regression.fixtures import make_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".documents/pdf-regression"))
    parser.add_argument("--ocr", action="store_true", help="Use installed Tesseract; default disables OCR")
    parser.add_argument("--table-backend", choices=("native", "auto", "docling"), default="native")
    parser.add_argument("--table-models", default="", help="Prefetched local Docling model directory")
    parser.add_argument("--strict-tables", action="store_true", help="Reject failed Docling processing instead of native fallback")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for expected in load_expectations()["cases"]:
        case_id = expected["id"]
        content = make_pdf(case_id)
        (args.output / f"{case_id}.pdf").write_bytes(content)
        parsed, error, inspections = None, None, []
        try:
            parsed = parse_pdf(f"{case_id}.pdf", content, ocr_images=args.ocr,
                               table_backend=args.table_backend, table_models_directory=args.table_models,
                               table_fallback=not args.strict_tables,
                               table_work_directory=str(args.output / "work"))
        except ValueError as exc:
            error = str(exc)
            inspections = getattr(exc, "page_inspections", [])
        report = evaluate(parsed, expected, error=error, page_inspections=inspections)
        report["quality_report"] = asdict(check_pdf(parsed, build_chunks(parsed))) if parsed else None
        reports.append(report)
        cells = report["metrics"]["cell_text"]
        result = "measured checks passed" if report["measured_checks_passed"] else "gaps found"
        print(f"{case_id}: {result}; cells {cells['matched']}/{cells['expected']}")
    output = args.output / "report.json"
    output.write_text(json.dumps({
        "schema_version": 1, "specimen_source": "synthetic",
        "ocr_mode": "tesseract_requested" if args.ocr else "disabled",
        "table_backend_requested": args.table_backend,
        "note": "Gap report, not acceptance or real-document accuracy certification.",
        "cases": reports,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {output.resolve()}")


if __name__ == "__main__":
    main()
