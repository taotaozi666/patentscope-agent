"""共享的中国专利公开号识别与规范化。"""

import re

PUBLICATION_NUMBER = re.compile(r"(?<![A-Z0-9])CN\s*\d{6,12}\s*[A-CUSY](?:\d)?(?![A-Z0-9])", re.IGNORECASE)


def publication_numbers(text: str) -> list[str]:
    """按出现顺序去重，空白和大小写不影响公开号身份。"""
    return list(dict.fromkeys(re.sub(r"\s+", "", match.group()).upper()
                              for match in PUBLICATION_NUMBER.finditer(text)))


def normalize_publication_number(value: str) -> str:
    match = PUBLICATION_NUMBER.fullmatch(value.strip())
    return re.sub(r"\s+", "", match.group()).upper() if match else ""
