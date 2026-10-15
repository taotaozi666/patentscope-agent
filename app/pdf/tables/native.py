"""Recover native cell spans from actual geometry, retaining unknowns on mismatch."""

from dataclasses import asdict

import pymupdf

from app.schemas.table import TableCell, TableFragment
from ..geometry import source_span
from ..inspection import InspectedPage, NativeTableInput


NATIVE_TABLE_VERSION = "native-table-grid-v1"


def alignment_regions(page: pymupdf.Page) -> list[tuple[float, float, float, float]]:
    words = page.get_text("words")
    if not isinstance(words, list):
        return []
    lines = []
    for word in sorted(words, key=lambda value: (value[1], value[0])):
        line = next((line for line in lines if abs(line["y"] - word[1]) < 3), None)
        if line is None:
            line = {"y": word[1], "words": []}
            lines.append(line)
        line["words"].append(word)
    starts = []
    for line in lines:
        words = sorted(line["words"], key=lambda value: value[0])
        groups, end = [], -1.0
        for word in words:
            if not groups or word[0] - end > 20:
                groups.append(word[0])
            end = word[2]
        if len(groups) >= 2:
            starts.append({"starts": groups, "words": words, "y": line["y"]})
    regions, consumed = [], set()
    for index, row in enumerate(starts):
        if index in consumed:
            continue
        matched = [(number, other) for number, other in enumerate(starts) if number not in consumed
                   and abs(other["starts"][0] - row["starts"][0]) < 8
                   and abs(other["starts"][1] - row["starts"][1]) < 8]
        groups = []
        for number, other in matched:
            if not groups or other["y"] - groups[-1][-1][1]["y"] > 100:
                groups.append([])
            groups[-1].append((number, other))
        for group in groups:
            if len(group) < 3:
                continue
            selected = [word for _, other in group for word in other["words"]]
            regions.append((min(word[0] for word in selected) - 1, min(word[1] for word in selected) - 1,
                            max(word[2] for word in selected) + 1, max(word[3] for word in selected) + 1))
            consumed.update(number for number, _ in group)
    return regions


def has_column_alignment(page: pymupdf.Page) -> bool:
    return bool(alignment_regions(page))


def native_inputs(page: pymupdf.Page, inspected: InspectedPage) -> list[NativeTableInput]:
    if inspected.tables or inspected.diagnosis.metrics["table_detection"] == "failed" or not has_column_alignment(page):
        return inspected.tables
    results = []
    for clip in alignment_regions(page):
        finder = page.find_tables(clip=clip, strategy="text", min_words_vertical=2, min_words_horizontal=1)
        for table in finder.tables if finder is not None else []:
            rows = table.extract()
            kept = [row for row, values in enumerate(rows) if any(value and value.strip() for value in values)]
            if len(kept) < 3 or table.col_count < 2:
                continue
            results.append(NativeTableInput(len(results), tuple(table.bbox), [rows[row] for row in kept], table.col_count,
                [[tuple(box) if box else None for box in table.rows[row].cells] for row in kept], strategy="text"))
    return results


def _boundaries(values):
    result = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > .5:
            result.append(value)
    return result


def parse_native_table(page: pymupdf.Page, table: NativeTableInput) -> TableFragment:
    assert page.number is not None
    identifier = f"p{page.number + 1}_table{table.index}"
    boxes = [box for row in table.cell_boxes for box in row if box is not None]
    xs = _boundaries([value for box in boxes for value in (box[0], box[2])])
    ys = _boundaries([value for box in boxes for value in (box[1], box[3])])
    geometry_matches = len(xs) == table.column_count + 1 and len(ys) == len(table.rows) + 1
    cells, occupied = [], {}
    images = page.get_image_info()
    words = page.get_text("words") if table.strategy == "text" else []
    complete = True
    for row, values in enumerate(table.rows):
        for column, value in enumerate(values):
            box = table.cell_boxes[row][column]
            if box is None:
                continue
            row_span = col_span = None
            if table.strategy == "text":
                row_span, col_span = 1, 1
            elif geometry_matches:
                left = min(range(len(xs)), key=lambda index: abs(xs[index] - box[0]))
                right = min(range(len(xs)), key=lambda index: abs(xs[index] - box[2]))
                top = min(range(len(ys)), key=lambda index: abs(ys[index] - box[1]))
                bottom = min(range(len(ys)), key=lambda index: abs(ys[index] - box[3]))
                if (left, top) == (column, row) and right > left and bottom > top:
                    row_span, col_span = bottom - top, right - left
            if row_span is None or col_span is None:
                complete = False
            cell_id = f"{identifier}_r{row}c{column}"
            # Native empty text is ambiguous where a raster image covers the cell.
            raster_overlap = any((pymupdf.Rect(box) & pymupdf.Rect(image["bbox"])).get_area() > 0
                                 for image in images) if value is None else False
            state = "text" if value and value.strip() else "unknown" if raster_overlap else "blank"
            source_box = box
            if table.strategy == "text" and value:
                included = [word for word in words if box[0] <= (word[0] + word[2]) / 2 < box[2]
                            and box[1] <= (word[1] + word[3]) / 2 < box[3]]
                if included:
                    source_box = (min(word[0] for word in included), min(word[1] for word in included),
                                  max(word[2] for word in included), max(word[3] for word in included))
            cell = TableCell(cell_id, identifier, row, column, value, state=state,
                             row_span=row_span, col_span=col_span, text_origin="native",
                             spans=[source_span(page, source_box, cell_id, "cell_content" if table.strategy == "text" else "cell")])
            cells.append(cell)
            for covered_row in range(row, row + (row_span or 1)):
                for covered_column in range(column, column + (col_span or 1)):
                    if (covered_row, covered_column) in occupied:
                        raise ValueError("原生表格检测产生重叠的单元格")
                    occupied[(covered_row, covered_column)] = cell
    for row in range(len(table.rows)):
        for column in range(table.column_count):
            if (row, column) not in occupied:
                complete = False
                cell_id = f"{identifier}_r{row}c{column}"
                cells.append(TableCell(cell_id, identifier, row, column, table.rows[row][column]))
    return TableFragment(identifier, source_span(page, table.bbox, identifier), len(table.rows),
        table.column_count, cells, backend=f"pymupdf-{table.strategy}", backend_version=NATIVE_TABLE_VERSION,
        structure_status="complete" if complete else "partial",
        provenance_status="complete" if all(cell.spans for cell in cells) else "partial",
        raw_output=asdict(table))
