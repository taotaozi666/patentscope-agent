"""Isolated local inference; model download is disabled in the process environment."""

import json
from pathlib import Path
import sys


def main():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TesseractCliOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    source, output, models, mode, ocr, language = sys.argv[1:]
    options = PdfPipelineOptions(artifacts_path=Path(models), do_table_structure=True,
                                do_ocr=ocr == "1", enable_remote_services=False,
                                document_timeout=90.0)
    options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=2)
    options.table_structure_options.mode = TableFormerMode.ACCURATE if mode == "accurate" else TableFormerMode.FAST
    options.table_structure_options.do_cell_matching = True
    options.ocr_options = TesseractCliOcrOptions(lang=language.split("+"))
    converter = DocumentConverter(allowed_formats=[InputFormat.PDF],
                                  format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    result = converter.convert(Path(source), raises_on_error=False)
    Path(output).write_text(json.dumps({"status": result.status.value,
                                       "document": result.document.model_dump(mode="json")}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
