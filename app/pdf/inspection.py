"""一次读取原生页面信息，输出可保存的诊断和不依赖引擎对象的输入。

诊断不执行 OCR 或模型调用。图片中的内容角色和文字层与渲染图的
一致性尚未核验，扫描/混合标签只是有理由可复核的候选。
"""

from dataclasses import asdict, dataclass
import math
from typing import Literal, cast

import pymupdf

from app.schemas.inspection import PageInspection, PageSource, RegionInspection
from app.schemas.pdf import NativeImageInfo, NativeTextBlock, NativeTextPage
from .geometry import source_span


INSPECTION_VERSION = "page-inspection-v1"


@dataclass(frozen=True)
class InspectionPolicy:
    replacement_ratio: float = .02
    block_table_overlap: float = .8
    block_image_overlap: float = .65
    vector_ocr_path_count: int = 100

    def __post_init__(self):
        for name in ("replacement_ratio", "block_table_overlap", "block_image_overlap"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} 必须大于 0 且不超过 1")
        if type(self.vector_ocr_path_count) is not int or self.vector_ocr_path_count < 0:
            raise ValueError("vector_ocr_path_count 必须是非负整数")


@dataclass
class NativeTableInput:
    index: int
    bbox: tuple[float, float, float, float]
    rows: list[list[str | None]]
    column_count: int
    cell_boxes: list[list[tuple[float, float, float, float] | None]]
    strategy: str = "lines"


@dataclass
class InspectedPage:
    diagnosis: PageInspection
    text_blocks: list[NativeTextBlock]
    images: list[NativeImageInfo]
    tables: list[NativeTableInput]


def _block_text(block) -> str:
    return "\n".join("".join(span["text"] for span in line["spans"])
                     for line in block["lines"]).strip()


def _valid_box(box) -> bool:
    return (len(box) == 4 and all(math.isfinite(value) for value in box)
            and box[0] < box[2] and box[1] < box[3])


def _coverage(inner, outer) -> float:
    rect = pymupdf.Rect(inner)
    return (rect & pymupdf.Rect(outer)).get_area() / rect.get_area() if rect.get_area() else 0.0


def _union_area(boxes) -> float:
    """矩形并集面积，避免同一图片/遮罩重叠后覆盖率超过 1。"""
    xs = sorted({value for box in boxes for value in (box[0], box[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((box[1], box[3]) for box in boxes if box[0] < right and box[2] > left)
        end, length = -math.inf, 0.0
        for top, bottom in intervals:
            length += max(0.0, bottom - max(top, end))
            end = max(end, bottom)
        area += (right - left) * length
    return area


def inspect_page(page: pymupdf.Page, policy: InspectionPolicy | None = None) -> InspectedPage:
    policy = policy or InspectionPolicy()
    if page.number is None:
        raise ValueError("PDF 页面未关联文档，无法记录来源页码")
    page_no = page.number + 1
    data = cast(NativeTextPage, page.get_text("dict", sort=True,
                flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES))
    blocks = [block for block in data["blocks"] if block["type"] == 0]
    _remove_overlapping_text(blocks)
    raw_images = cast(list[NativeImageInfo], page.get_image_info())
    images = _deduplicate_image_infos(raw_images)
    warnings = []
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
        warnings.append(f"第 {page_no} 页矢量区域检查失败，不能确认文字轮廓和框线")
    texts = [_block_text(block) for block in blocks]
    char_count = sum(len(text) for text in texts)
    replacement_count = sum(text.count("\ufffd") for text in texts)
    invalid_boxes = sum(bool(text) and not _valid_box(block["bbox"])
                        for block, text in zip(blocks, texts))
    unreliable = bool(invalid_boxes or any(text and text.count("\ufffd") / len(text) > policy.replacement_ratio
                                         for text in texts))
    tables = []
    table_detection = "not_attempted"
    if char_count:
        try:
            finder = page.find_tables(paths=drawings)
            for index, table in enumerate(finder.tables if finder is not None else []):
                rows = table.extract()
                if len(rows) >= 2 and table.col_count >= 2:
                    tables.append(NativeTableInput(
                        index, tuple(table.bbox), rows, table.col_count,
                        [[tuple(box) if box else None for box in row.cells] for row in table.rows],
                    ))
            table_detection = "completed"
        except Exception:
            table_detection = "failed"
            warnings.append(f"第 {page_no} 页表格检测失败，已保留原生文字，表格需复核")
    # 坐标在转换前裁到未旋转页面，用并集计算真实图片覆盖率。
    bounds = page.rect * page.derotation_matrix
    image_boxes = [tuple(pymupdf.Rect(image["bbox"]) & bounds) for image in raw_images]
    image_boxes = [box for box in image_boxes if _valid_box(box)]
    image_coverage = _union_area(image_boxes) / bounds.get_area() if bounds.get_area() else 0.0
    source: PageSource
    status: Literal["observed", "candidate"] = "candidate"
    if unreliable:
        source, status, reasons = "unreliable_text_layer", "candidate", ["replacement_characters_or_invalid_text_geometry"]
    elif char_count and images:
        source, status, reasons = "hybrid", "candidate", ["native_text_with_unverified_raster_regions"]
    elif char_count:
        source, status, reasons = "native", "observed", ["extractable_native_text"]
    elif images:
        source, status, reasons = "scanned", "candidate", ["raster_regions_without_text_layer"]
    elif len(drawings) > policy.vector_ocr_path_count:
        source, status, reasons = "unknown", "candidate", ["possible_vector_text_without_text_layer"]
    elif drawings:
        source, status, reasons = "unknown", "candidate", ["vector_graphics_without_text_layer"]
    else:
        source, status, reasons = "empty", "observed", ["no_text_images_or_vector_paths"]
    regions = []
    for table in tables:
        identifier = f"p{page_no}_table{table.index}"
        regions.append(RegionInspection(identifier, source_span(page, table.bbox, identifier),
                       table.bbox, "table", "native", "observed",
                       {"row_count": len(table.rows), "column_count": table.column_count},
                       ["native_table_candidate"]))
    for index, (block, text) in enumerate(zip(blocks, texts)):
        if not text or not _valid_box(block["bbox"]):
            continue
        box = tuple(block["bbox"])
        if any(_coverage(box, table.bbox) >= policy.block_table_overlap for table in tables):
            continue  # 所属表格已经有独立路线，避免重复派发文字。
        if any(_coverage(box, image["bbox"]) >= policy.block_image_overlap for image in images):
            continue  # 图片的原生文字由同一区域计划负责融合。
        identifier = f"p{page_no}_text{index}"
        ratio = text.count("\ufffd") / len(text)
        suspect = ratio > policy.replacement_ratio
        regions.append(RegionInspection(identifier, source_span(page, box, identifier), box, "text",
                       "unreliable_text_layer" if suspect else "native", "candidate" if suspect else "observed",
                       {"native_char_count": len(text), "replacement_ratio": ratio},
                       ["replacement_characters"] if suspect else ["extractable_native_text"]))
    if not char_count and (images or len(drawings) > policy.vector_ocr_path_count):
        identifier = f"p{page_no}_page_image"
        box = tuple(bounds)
        regions.append(RegionInspection(identifier, source_span(page, box, identifier, "page"),
                       box, "unknown", source, "candidate", reasons=list(reasons)))
    else:
        for index, image in enumerate(images):
            box = tuple(image["bbox"])
            if not _valid_box(box):
                continue
            native_count = sum(len(text) for block, text in zip(blocks, texts)
                               if _coverage(block["bbox"], box) >= policy.block_image_overlap)
            identifier = f"p{page_no}_image{index}"
            regions.append(RegionInspection(identifier, source_span(page, box, identifier), box,
                           "image", "hybrid" if native_count else "scanned", "candidate",
                           {"native_char_count": native_count}, ["raster_content_not_yet_classified"]))
        if not char_count and drawings and not images:
            identifier = f"p{page_no}_vector_region"
            box = tuple(bounds)
            regions.append(RegionInspection(identifier, source_span(page, box, identifier),
                           box, "unknown", "unknown", "candidate", reasons=list(reasons)))
    # 交叠信息供后续区域 OCR 排除已提取区域；本步骤不剪裁或删除原内容。
    for index, region in enumerate(regions):
        for other in regions[index + 1:]:
            if (pymupdf.Rect(region.pdf_bbox) & pymupdf.Rect(other.pdf_bbox)).get_area() > 0:
                region.overlaps.append(other.id)
                other.overlaps.append(region.id)
    diagnosis = PageInspection(
        page_no, page.get_label() or "", page.rect.width, page.rect.height, page.rotation,
        source, status, metrics={
            "native_char_count": char_count, "replacement_ratio": replacement_count / char_count if char_count else 0.0,
            "invalid_text_boxes": invalid_boxes, "image_count": len(images),
            "image_placement_count": len(raw_images),
            "image_coverage_ratio": min(1.0, image_coverage), "drawing_count": len(drawings),
            "table_candidate_count": len(tables), "table_detection": table_detection,
            "render_alignment_verified": False,
        }, policy=asdict(policy), reasons=reasons, warnings=warnings, regions=regions,
    )
    return InspectedPage(diagnosis, blocks, images, tables)


def _deduplicate_image_infos(images: list[NativeImageInfo]) -> list[NativeImageInfo]:
    """Merge duplicate/partial raster placements whose smaller box is mostly covered."""
    groups: list[list[NativeImageInfo]] = []

    def bounds(group: list[NativeImageInfo]) -> tuple[float, float, float, float]:
        boxes = [tuple(float(value) for value in item["bbox"]) for item in group]
        return (min(box[0] for box in boxes), min(box[1] for box in boxes),
                max(box[2] for box in boxes), max(box[3] for box in boxes))

    for image in images:
        box = tuple(float(value) for value in image["bbox"])
        digest = image.get("digest")
        match = None
        for group in groups:
            area = max(0.0, box[2]-box[0]) * max(0.0, box[3]-box[1])
            for member in group:
                member_box = tuple(float(value) for value in member["bbox"])
                left, top = max(box[0], member_box[0]), max(box[1], member_box[1])
                right, bottom = min(box[2], member_box[2]), min(box[3], member_box[3])
                intersection = max(0.0, right-left) * max(0.0, bottom-top)
                member_area = max(0.0, member_box[2]-member_box[0]) * max(0.0, member_box[3]-member_box[1])
                coverage = intersection / min(area, member_area) if min(area, member_area) else 0.0
                same_image = bool(digest and member.get("digest") == digest)
                if coverage >= (0.5 if same_image else 0.65):
                    match = group
                    break
            if match is not None:
                break
        if match is None:
            groups.append([image])
        else:
            match.append(image)

    merged: list[NativeImageInfo] = []
    for group in groups:
        # Preserve useful image properties from the largest underlying placement,
        # while rendering the union so clipped/masked fragments are not lost.
        representative = max(group, key=lambda item: (item["bbox"][2]-item["bbox"][0]) *
                             (item["bbox"][3]-item["bbox"][1]))
        merged.append({**representative, "bbox": bounds(group)})
    return merged


def _remove_overlapping_text(blocks) -> None:
    """同页相同文字且区域几乎重合才去重；保留不同位置的重复数字。"""
    seen = {}
    for block in blocks:
        for line in block["lines"]:
            retained = []
            for span in line["spans"]:
                text = span["text"]
                box = span.get("bbox")
                if not text.strip() or box is None:
                    retained.append(span)
                    continue
                rect = pymupdf.Rect(box)
                duplicate = any(
                    min(rect.get_area(), old.get_area()) > 0
                    and (rect & old).get_area() / min(rect.get_area(), old.get_area()) >= .95
                    and all(abs(a-b) <= 1 for a, b in zip(rect, old))
                    for old in seen.get(text, [])
                )
                if not duplicate:
                    retained.append(span)
                    seen.setdefault(text, []).append(rect)
            line["spans"] = retained
