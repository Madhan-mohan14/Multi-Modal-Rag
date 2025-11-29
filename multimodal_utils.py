# multimodal_utils.py
"""
Small human-friendly utilities for markdown/table/image normalization.
"""

import re
import unicodedata
from typing import Optional


def normalize_markdown(md: str) -> str:
    if not md:
        return ""
    md = unicodedata.normalize("NFC", md).replace("\x00", "")
    # ensure a space after markdown headings: "##Heading" -> "## Heading"
    md = re.sub(r"^(#{1,6})(?!\s)", r"\1 ", md, flags=re.MULTILINE)
    # collapse excessive blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = "\n".join(line.rstrip() for line in md.splitlines())
    return md.strip()


def sanitize_table_markdown(md: str) -> str:
    if not md:
        return ""
    lines = md.splitlines()
    if len(lines) < 2:
        return md
    # detect table-like spacing
    if not any(re.search(r"\s{2,}", ln) for ln in lines[:10]):
        return md
    out = []
    for ln in lines:
        if re.search(r"\s{2,}", ln):
            row = re.sub(r"[\t ]{2,}", " | ", ln).strip()
            row = row.strip("| ")
            out.append("| " + row + " |")
        else:
            out.append(ln)
    # ensure header separator
    if out and out[0].startswith("|") and (len(out) == 1 or not re.match(r"^\|\s*[-:]+\s*(\|\s*[-:]+\s*)+\|?$", out[1])):
        cols = [c.strip() for c in out[0].strip("| ").split("|")]
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        out.insert(1, sep)
    return "\n".join(out)


def format_image_block(image_id: Optional[str], caption: Optional[str], ocr_text: Optional[str]) -> str:
    parts = []
    parts.append(f"![image]({image_id})" if image_id else "![image](image)")
    if caption:
        c = caption.strip()
        if c:
            parts.append(f"**Caption:** {c}")
    if ocr_text:
        txt = re.sub(r"\n{2,}", "\n", ocr_text.strip())
        parts.append(f"**OCR:** {txt[:300]}")
    return "\n\n".join(parts).strip()


def safe_filename(name: str) -> str:
    if not name:
        return "file"
    n = unicodedata.normalize("NFKD", name)
    n = n.encode("ascii", "ignore").decode("ascii")
    n = n.strip().replace(" ", "_")
    n = re.sub(r"[^A-Za-z0-9_\-\.]", "", n).lower()
    return n or "file"


def shorten_text_preview(text: str, limit: int = 200) -> str:
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text).strip()
    return s if len(s) <= limit else s[:limit].rstrip() + "…"
