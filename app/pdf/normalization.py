"""PDF display/index text cleanup while keeping source offsets traceable."""

from bisect import bisect_left
import re
from typing import Any

PDF_TEXT_NORMALIZATION_VERSION = "pdf-soft-wrap-v4"

PATENT_PARAGRAPH_START = re.compile(r"^\[\d{3,}\]")

_STRUCTURE_START = re.compile(
    r"^(?:\[\d{3,}\]|\(?\d+[.)、]|[（(][一二三四五六七八九十百\d]+[）)]|"
    r"[一二三四五六七八九十]+、|[-*•▪■]|权利要求\s*\d+|第\s*\d+\s*[条项])"
)
_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]$")
_CJK_START = re.compile(r"^[\u3400-\u9fff\uf900-\ufaff]")
_CJK_SPACE_CHARS = {" ", "\t", "\u00a0", "\u3000"}


def normalize_pdf_text(text: str, mappings: list[dict[str, Any]] | None = None
                       ) -> tuple[str, list[dict[str, Any]]]:
    """Join visual soft wraps, retain paragraph/list breaks, and remap chunk offsets.

    Mapping ``start``/``end`` fields refer to the chunk text and are translated
    to the returned normalized text. Source character offsets remain untouched.
    """
    if not text:
        return text, [dict(item) for item in mappings or []]

    lines: list[tuple[str, int, list[int]]] = []
    cursor = 0
    for part in text.splitlines(keepends=True):
        body = part.rstrip("\r\n")
        lines.append((body, cursor,
                      list(range(cursor + len(body), cursor + len(part)))))
        cursor += len(part)
    if cursor < len(text):
        lines.append((text[cursor:], cursor, []))

    output: list[str] = []
    source_positions: list[int] = []
    previous = ""
    pending_breaks: list[int] = []
    had_blank_line = False
    numbered_paragraph = False

    def append(value: str, positions: list[int]) -> None:
        output.append(value)
        source_positions.extend(positions)

    for body, start, line_breaks in lines:
        left_trimmed = body.lstrip(" \t")
        line = left_trimmed.rstrip(" \t")
        line_start = start + len(body) - len(left_trimmed)
        if not line:
            had_blank_line = bool(previous)
            pending_breaks.extend(line_breaks)
            continue

        if previous:
            break_pos = pending_breaks[0] if pending_breaks else max(0, start - 1)
            if PATENT_PARAGRAPH_START.match(line):
                # 专利编号是明确段界，不继承 PDF 文本块引入的额外空行。
                append("\n", [break_pos])
            elif had_blank_line and not numbered_paragraph:
                positions = (pending_breaks[:2] if len(pending_breaks) >= 2
                             else [break_pos, break_pos])
                append("\n\n", positions)
            elif _keep_line_break(previous, line):
                append("\n", [break_pos])
            else:
                separator = _soft_wrap_separator(previous, line)
                if separator:
                    append(separator, [break_pos] * len(separator))

        append(line, list(range(line_start, line_start + len(line))))
        if PATENT_PARAGRAPH_START.match(line):
            numbered_paragraph = True
        previous = line
        pending_breaks = line_breaks
        had_blank_line = False

    normalized = "".join(output)
    # PDF text layers sometimes insert spaces between adjacent Chinese glyphs.
    # Remove only those horizontal gaps; retain Latin, numeric, and line spacing.
    chars = list(normalized)
    filtered_chars: list[str] = []
    filtered_positions: list[int] = []
    index = 0
    while index < len(chars):
        if chars[index] in _CJK_SPACE_CHARS and filtered_chars and _is_cjk(filtered_chars[-1]):
            next_index = index + 1
            while next_index < len(chars) and chars[next_index] in _CJK_SPACE_CHARS:
                next_index += 1
            if next_index < len(chars) and _is_cjk(chars[next_index]):
                index = next_index
                continue
        filtered_chars.append(chars[index])
        filtered_positions.append(source_positions[index])
        index += 1
    normalized = "".join(filtered_chars).strip()
    source_positions = filtered_positions
    # Trimming only affects boundaries; project positions against the retained text.
    leading = len("".join(output)) - len("".join(output).lstrip())
    trailing_end = leading + len(normalized)
    source_positions = source_positions[leading:trailing_end]

    translated: list[dict[str, Any]] = []
    for original in mappings or []:
        item = dict(original)
        start = item.get("start")
        end = item.get("end")
        if isinstance(start, int) and isinstance(end, int):
            item["start"] = bisect_left(source_positions, start)
            item["end"] = bisect_left(source_positions, end)
        translated.append(item)
    return normalized, translated


def _keep_line_break(previous: str, current: str) -> bool:
    # A line beginning with a claim/list marker starts a new semantic unit.
    # A preceding line may itself begin with a marker (for example "4.") and
    # continue after a visual PDF wrap, so its prefix must not preserve the break.
    if _STRUCTURE_START.match(current.strip()):
        return True
    return False


def _is_cjk(char: str) -> bool:
    return bool(re.fullmatch(r"[\u3400-\u9fff\uf900-\ufaff]", char))


def _soft_wrap_separator(previous: str, current: str) -> str:
    left = previous.rstrip()
    right = current.lstrip()
    if not left or not right or left.endswith(("-", "\u00ad")):
        return ""
    if _CJK.search(left) or _CJK_START.match(right):
        return ""
    if left[-1].isalnum() and right[0].isalnum():
        return " "
    return ""


def is_page_number_region(text: str, page_no: int, bbox) -> bool:
    """小型页底数字区域，数字必须与物理页码一致；不处理公式或整页图片。"""
    tokens = text.split()
    if not tokens or not all(token.isascii() and token.isdecimal() and int(token) == page_no for token in tokens):
        return False
    if len(bbox) != 4:
        return False
    x0, y0, x1, y1 = bbox
    return (0 <= x0 < x1 <= 1 and .90 <= y0 < y1 <= 1
            and y1 - y0 <= .05 and x1 - x0 <= .3)
