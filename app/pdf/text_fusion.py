"""Fuse evidence by position; never deduplicate equal numbers at different places."""

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import re
import unicodedata

from app.schemas.pdf import SourceSpan


FUSION_VERSION = "position-text-fusion-v1"


@dataclass
class TextEvidence:
    id: str
    text: str
    bbox: tuple[float, float, float, float]
    span: SourceSpan
    origin: str


@dataclass
class FusionResult:
    selected: list[TextEvidence]
    duplicates: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)


def _normalized(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def _overlap(left, right) -> float:
    area = min((left[2] - left[0]) * (left[3] - left[1]),
               (right[2] - right[0]) * (right[3] - right[1]))
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * \
                   max(0, min(left[3], right[3]) - max(left[1], right[1]))
    return intersection / area if area > 0 else 0.0


def _bad(text: str) -> bool:
    return text.count("\ufffd") / max(1, len(text)) > .02


def fuse_text(native: list[TextEvidence], ocr: list[TextEvidence],
              *, existing: list[TextEvidence] | None = None) -> FusionResult:
    result = FusionResult(list(native))
    # Existing evidence belongs to another region and cannot be replaced here.
    for candidate in ocr:
        matched = False
        for previous in [*(existing or []), *result.selected]:
            if previous.span.page_no != candidate.span.page_no or _overlap(previous.bbox, candidate.bbox) < .65:
                continue
            old, new = _normalized(previous.text), _normalized(candidate.text)
            if old == new:
                result.duplicates.append({"retained_id": previous.id, "suppressed": asdict(candidate)})
                matched = True
                break
            similarity = SequenceMatcher(None, old, new).ratio()
            pattern = r"[+-]?\d+(?:[.,]\d+)*"
            old_numbers, new_numbers = re.findall(pattern, old), re.findall(pattern, new)
            numeric_conflict = old_numbers != new_numbers
            same_numeric_context = bool(old_numbers and new_numbers and
                                        re.sub(pattern, "", old) == re.sub(pattern, "", new))
            if similarity < .5 and not _bad(previous.text) and not (numeric_conflict and same_numeric_context):
                continue
            choose_ocr = previous in result.selected and previous.origin == "native" and _bad(previous.text) and not _bad(candidate.text)
            result.conflicts.append({"native_or_previous": asdict(previous), "ocr": asdict(candidate),
                                     "selected_id": candidate.id if choose_ocr else previous.id,
                                     "numeric_conflict": numeric_conflict, "requires_review": True})
            if choose_ocr:
                result.selected[result.selected.index(previous)] = candidate
            matched = True
            break
        if not matched:
            result.selected.append(candidate)
    result.selected.sort(key=lambda value: (value.bbox[1], value.bbox[0], value.id))
    return result


def serialize_fusion(result: FusionResult) -> tuple[str, list[dict]]:
    """Index text plus exact output ranges pointing back to each evidence block."""
    text, fragments = "", []
    for evidence in result.selected:
        if text:
            text += "\n"
        start = len(text)
        text += evidence.text
        fragments.append({"start": start, "end": len(text), "span": asdict(evidence.span),
                          "evidence_id": evidence.id, "origin": evidence.origin})
    return text, fragments
