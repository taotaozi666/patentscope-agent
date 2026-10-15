"""Synthetic PDF specimens; no OCR engine, downloads or model calls required.

Do not read expected.json here: changing a parser or expectation must not change
the PDF input. Coordinates are in unrotated PDF points on 600 x 800 pages.
"""

import pymupdf


CASES = (
    "native_grid", "hybrid_page", "scanned_table", "multi_header",
    "merged_cells", "borderless", "continued_table", "distinct_tables",
)


def _text(page, x, y, value, size=11):
    font = "china-s" if any(ord(char) > 127 for char in value) else "helv"
    page.insert_text((x, y), value, fontsize=size, fontname=font)


def _grid(page, xs, ys, cells, *, borders=True):
    """Draw actual cell rectangles, including missing internal merged borders."""
    edges = set()
    for row, column, value, row_span, col_span in cells:
        left, right = xs[column], xs[column + col_span]
        top, bottom = ys[row], ys[row + row_span]
        for start, end in (
            ((left, top), (right, top)), ((left, bottom), (right, bottom)),
            ((left, top), (left, bottom)), ((right, top), (right, bottom)),
        ):
            edges.add((start, end))
        if value:
            _text(page, left + 8, top + 25, value)
    if borders:
        for start, end in sorted(edges):
            page.draw_line(start, end, width=.7)


def _rows(values):
    return [(row, col, value, 1, 1)
            for row, values_row in enumerate(values)
            for col, value in enumerate(values_row)]


def _simple(page, *, borders=True):
    _grid(page, [60, 220, 380, 540], [160, 200, 240, 280], _rows([
        ["实施例", "拉伸强度 MPa", "Count"],
        ["Ex1", "35.2", "0"], ["Ex2", "40.8", ""],
    ]), borders=borders)


def _raster_table(page, *, full_page):
    with pymupdf.open() as source:
        original = source.new_page(width=600, height=800)
        _simple(original)
        clip = original.rect if full_page else pymupdf.Rect(60, 160, 540, 280)
        image = original.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=clip).tobytes("png")
    page.insert_image(clip, stream=image, keep_proportion=False)


def make_pdf(case: str) -> bytes:
    """Build a fixture without consuming the gold annotations."""
    if case not in CASES:
        raise ValueError(f"Unknown PDF regression case: {case}")
    with pymupdf.open() as document:
        page = document.new_page(width=600, height=800)
        if case != "scanned_table":
            _text(page, 60, 60, "专利实验结果", size=20)
            _text(page, 60, 100, "Native paragraph: compare the experimental samples.")
        if case in {"native_grid", "borderless"}:
            _simple(page, borders=case != "borderless")
        elif case in {"hybrid_page", "scanned_table"}:
            _raster_table(page, full_page=case == "scanned_table")
        elif case == "multi_header":
            _grid(page, [60, 180, 300, 420, 540], [160, 200, 240, 280, 320], [
                (0, 0, "Sample", 2, 1), (0, 1, "Properties", 1, 2),
                (0, 3, "Note", 2, 1), (1, 1, "Strength MPa", 1, 1),
                (1, 2, "Elongation %", 1, 1),
                (2, 0, "Ex1", 1, 1), (2, 1, "35.2", 1, 1),
                (2, 2, "120", 1, 1), (2, 3, "pass", 1, 1),
                (3, 0, "Ex2", 1, 1), (3, 1, "40.8", 1, 1),
                (3, 2, "140", 1, 1), (3, 3, "-", 1, 1),
            ])
        elif case == "merged_cells":
            _grid(page, [60, 220, 380, 540], [160, 200, 240, 280], [
                (0, 0, "Sample", 1, 1), (0, 1, "Strength MPa", 1, 1),
                (0, 2, "Elongation %", 1, 1), (1, 0, "Ex1", 2, 1),
                (1, 1, "35.2", 1, 1), (1, 2, "120", 1, 1),
                (2, 1, "40.8", 1, 1), (2, 2, "140", 1, 1),
            ])
        else:
            for index in range(2):
                if index:
                    page = document.new_page(width=600, height=800)
                same_table = case == "continued_table"
                caption = "Table 4 Results" if index == 0 else (
                    "Table 4 continued" if same_table else "Table 5 Other results")
                _text(page, 60, 140, caption)
                unit = "MPa" if same_table or index == 0 else "kPa"
                start = index * 2 + 1
                _grid(page, [60, 220, 380, 540], [160, 200, 240, 280], _rows([
                    ["Sample", f"Strength {unit}", "Elongation %"],
                    [f"Ex{start}", "35.2", "120"],
                    [f"Ex{start + 1}", "40.8", "140"],
                ]))
        return document.tobytes()
