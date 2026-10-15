"""Render only the requested region, OCR it, then map lines to PDF coordinates."""

from dataclasses import dataclass, field
import math

import pymupdf
from PIL import Image, ImageDraw


OCR_VERSION = "region-ocr-v1"
Box = tuple[float, float, float, float]


@dataclass
class OcrResult:
    blocks: list[tuple[Box, str]] = field(default_factory=list)
    transform: dict = field(default_factory=dict)


def _recognize_pixmap(pixmap: pymupdf.Pixmap, language: str) -> bytes:
    return pixmap.pdfocr_tobytes(language=language, tessdata=pymupdf.get_tessdata())


def ocr_region(page: pymupdf.Page, bbox: Box, language: str, dpi: int,
               *, exclude_regions: list[Box] | None = None) -> OcrResult:
    if type(dpi) is not int or not 72 <= dpi <= 600:
        raise ValueError("OCR DPI 必须介于 72 和 600 之间")
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError("OCR 区域坐标无效")
    rotation = page.rotation_matrix
    derotation = page.derotation_matrix
    bounds = page.rect * derotation
    raw_clip = pymupdf.Rect(bbox) & bounds
    clip = (raw_clip * rotation) & page.rect
    if clip.is_empty or clip.is_infinite:
        raise ValueError("OCR 区域不在页面内")
    pixmap = page.get_pixmap(dpi=dpi, clip=clip, colorspace=pymupdf.csRGB,
                             alpha=False, annots=False)
    device_rect = tuple(pixmap.irect)
    device_x, device_y = pixmap.x, pixmap.y
    # Keep raster coordinates local, retaining device origin for inverse mapping.
    pixmap.set_origin(0, 0)
    scale = dpi / 72
    exclusions, pixel_masks = [], []
    for raw in exclude_regions or []:
        excluded = pymupdf.Rect(raw) & raw_clip
        if excluded.is_empty:
            continue
        absolute = (excluded * rotation * pymupdf.Matrix(scale, scale)).irect
        pixel_box = pymupdf.IRect(absolute.x0 - device_x, absolute.y0 - device_y,
                                  absolute.x1 - device_x, absolute.y1 - device_y) & pixmap.irect
        if not pixel_box.is_empty:
            pixel_masks.append(pixel_box)
            exclusions.append(tuple(excluded))
    if pixel_masks:
        # Avoid native Pixmap.clear_with: some Windows builds corrupt the heap
        # when clearing rendered crops. Mask a detached byte buffer instead.
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        drawing = ImageDraw.Draw(image)
        for mask in pixel_masks:
            drawing.rectangle((mask.x0, mask.y0, mask.x1 - 1, mask.y1 - 1), fill="white")
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, image.width, image.height, image.tobytes(), False)
        pixmap.set_dpi(dpi, dpi)
    output = _recognize_pixmap(pixmap, language)
    with pymupdf.open(stream=output, filetype="pdf") as recognized:
        if len(recognized) != 1:
            raise ValueError("OCR 引擎必须返回单页 PDF")
        ocr_page = recognized[0]
        sx, sy = pixmap.width / ocr_page.rect.width, pixmap.height / ocr_page.rect.height
        transform = {
            "version": OCR_VERSION, "engine": "tesseract", "language": language, "dpi": dpi,
            "crop_pdf_bbox": tuple(raw_clip), "crop_display_bbox": tuple(clip),
            "pixel_rect": device_rect, "ocr_page_size": (ocr_page.rect.width, ocr_page.rect.height),
            "rotation_matrix": (rotation.a, rotation.b, rotation.c, rotation.d, rotation.e, rotation.f),
            "derotation_matrix": (derotation.a, derotation.b, derotation.c, derotation.d, derotation.e, derotation.f),
            "excluded_pdf_regions": exclusions,
        }
        blocks = []
        text_page = ocr_page.get_textpage(flags=pymupdf.TEXTFLAGS_DICT)
        text_data = text_page.extractDICT(cb=ocr_page.cropbox, sort=True)
        for block in text_data["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if not text:
                    continue
                left, top, right, bottom = line["bbox"]
                # Device origin accounts for integer raster rounding at crop edges.
                displayed = pymupdf.Rect((left * sx + device_x) / scale,
                                         (top * sy + device_y) / scale,
                                         (right * sx + device_x) / scale,
                                         (bottom * sy + device_y) / scale)
                raw = (displayed * derotation) & raw_clip
                if raw.is_empty or raw.is_infinite:
                    continue
                if any((raw & pymupdf.Rect(excluded)).get_area() >= raw.get_area() * .8
                       for excluded in exclusions):
                    continue
                blocks.append(((raw.x0, raw.y0, raw.x1, raw.y1), text))
    return OcrResult(blocks, transform)
