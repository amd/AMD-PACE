# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""General-purpose extraction toolkit made available inside the sandbox.

The planner writes this module's source into every execution workspace as
``gaia_toolkit.py`` so generated code can ``import gaia_toolkit`` and rely on
real parsing tools instead of the model re-deriving format-specific recipes.

Design goals:
- ONE entry point, ``read(path)``, that turns *any* local file into text by
  sniffing its real type (magic bytes first, extension second) and dispatching
  to the right reader. This is what "rely on the tools, not model reasoning"
  looks like: the agent asks for the content, not for a recipe.
- Robust HTML handling: ``html_to_text`` and ``extract_code`` always unescape
  entities and can target a specific language section on a multi-language page
  (wiki / Rosetta-style) rather than grabbing the first ``<pre>``.
- Graceful degradation: optional libraries are imported lazily; if one is
  missing the function raises a clear ``ToolkitError`` telling the caller which
  package to ``provision``.

Everything here uses only the standard library plus optional third-party
readers, so the same module works in the planner process (for tests) and in the
sandbox.
"""

from __future__ import annotations

import html as _html
import re
import zipfile
from pathlib import Path


class ToolkitError(RuntimeError):
    """Raised when a file cannot be read or a needed reader is unavailable."""


_MAGIC = (
    (b"%PDF", "pdf"),
    (b"PK\x03\x04", "zip"),  # docx/pptx/xlsx/odf/plain zip - refined below
    (b"\x89PNG\r\n\x1a\n", "image"),
    (b"\xff\xd8\xff", "image"),  # jpeg
    (b"GIF87a", "image"),
    (b"GIF89a", "image"),
    (b"RIFF", "image"),  # webp (RIFF....WEBP) - good enough for OCR routing
    (b"II*\x00", "image"),  # tiff
    (b"MM\x00*", "image"),
    (b"<!DOCTYPE html", "html"),
    (b"<html", "html"),
    (b"<?xml", "xml"),
)

_EXT_TYPE = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".pptx": "pptx",
    ".ppt": "pptx",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".xls": "xlsx",
    ".csv": "csv",
    ".tsv": "csv",
    ".htm": "html",
    ".html": "html",
    ".xml": "xml",
    ".json": "text",
    ".zip": "zip",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".bmp": "image",
    ".tif": "image",
    ".tiff": "image",
    ".webp": "image",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".flac": "audio",
}

_ZIP_OFFICE = {
    "word/document.xml": "docx",
    "ppt/presentation.xml": "pptx",
    "xl/workbook.xml": "xlsx",
}


def sniff_type(path: str | Path) -> str:
    """Return a coarse type for *path*: pdf/docx/pptx/xlsx/csv/html/xml/image/
    audio/zip/text. Uses magic bytes first, then the extension, so a file whose
    name lies (or has no extension) is still classified correctly."""
    p = Path(path)
    head = b""
    try:
        with open(p, "rb") as fh:
            head = fh.read(512)
    except OSError as exc:  # pragma: no cover - passthrough
        raise ToolkitError(f"cannot open {p}: {exc}") from exc

    if head[:4] == b"PK\x03\x04":
        # A ZIP container - distinguish office formats by their member names.
        try:
            with zipfile.ZipFile(p) as zf:
                names = set(zf.namelist())
        except Exception:
            names = set()
        for member, kind in _ZIP_OFFICE.items():
            if member in names:
                return kind
        return "zip"

    for magic, kind in _MAGIC:
        if head.startswith(magic):
            return kind
    # HTML often starts with whitespace/BOM or a comment; sniff a bit deeper.
    lowered = head[:256].lstrip().lower()
    if lowered.startswith(b"<!doctype html") or lowered.startswith(b"<html"):
        return "html"

    ext = p.suffix.lower()
    if ext in _EXT_TYPE:
        return _EXT_TYPE[ext]
    # Fall back on printable-text heuristic.
    if not head or _looks_textual(head):
        return "text"
    return "binary"


def _looks_textual(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    text_chars = bytes(range(32, 127)) + b"\n\r\t\f\b"
    nontext = sum(byte not in text_chars for byte in data)
    return nontext / max(len(data), 1) < 0.30


def read(path: str | Path, *, kind: str | None = None) -> str:
    """Read *any* supported local file and return its text content.

    ``kind`` forces a reader; otherwise the type is sniffed. Raises
    ToolkitError (naming the package to provision) when a reader is missing.
    """
    p = Path(path)
    if not p.is_file():
        raise ToolkitError(f"no such file: {p}")
    resolved = kind or sniff_type(p)
    reader = {
        "pdf": _read_pdf,
        "docx": _read_docx,
        "pptx": _read_pptx,
        "xlsx": _read_xlsx,
        "csv": _read_text,
        "html": _read_html_file,
        "xml": _read_text,
        "image": _read_image_ocr,
        "zip": _read_zip_listing,
        "text": _read_text,
        "binary": _read_text,
        "audio": _reject_audio,
    }.get(resolved, _read_text)
    return reader(p)


def _require(module: str, package: str | None = None):
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        pkg = package or module
        raise ToolkitError(
            f"reader needs the '{pkg}' package - provision it "
            f"(pip: {pkg}) and retry"
        ) from exc


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        pypdf = _require("pypdf")
        reader = pypdf.PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except ToolkitError:
        pdfminer = _require("pdfminer.high_level", "pdfminer.six")
        return pdfminer.extract_text(str(path))


def _read_docx(path: Path) -> str:
    docx = _require("docx", "python-docx")
    document = docx.Document(str(path))
    parts = [para.text for para in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _read_pptx(path: Path) -> str:
    pptx = _require("pptx", "python-pptx")
    presentation = pptx.Presentation(str(path))
    parts: list[str] = []
    for index, slide in enumerate(presentation.slides, 1):
        parts.append(f"--- slide {index} ---")
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if shape.has_table:
                for row in shape.table.rows:
                    parts.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _read_xlsx(path: Path) -> str:
    openpyxl = _require("openpyxl")
    workbook = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    parts: list[str] = []
    for sheet in workbook.worksheets:
        parts.append(f"--- sheet {sheet.title} ---")
        for row in sheet.iter_rows(values_only=True):
            parts.append("\t".join("" if v is None else str(v) for v in row))
    return "\n".join(parts)


def _read_image_ocr(path: Path) -> str:
    pytesseract = _require("pytesseract")
    # __import__("PIL.Image") returns the top-level PIL package (not the submodule), so grab
    # Image explicitly — PIL.open() does not exist and would raise AttributeError.
    _require("PIL.Image", "Pillow")
    from PIL import Image

    image = Image.open(str(path))
    return pytesseract.image_to_string(image)


def _read_zip_listing(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        return "\n".join(zf.namelist())


def _reject_audio(path: Path) -> str:
    raise ToolkitError(
        f"{path.name} is audio; the sandbox has no speech-to-text. Use the "
        "planner's analyze_audio tool instead of reading it here."
    )


def _read_html_file(path: Path) -> str:
    return html_to_text(path.read_text(encoding="utf-8", errors="replace"))


def html_to_text(source: str) -> str:
    """Return readable text from an HTML string, preferring a real parser.

    Uses BeautifulSoup when available (dropping script/style), otherwise a
    tag-stripping fallback. Entities are always unescaped.
    """
    try:
        bs4 = __import__("bs4")
        soup = bs4.BeautifulSoup(source, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
    except ImportError:
        no_tags = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", source)
        no_tags = re.sub(r"(?s)<[^>]+>", "", no_tags)
        text = _html.unescape(no_tags)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_code(source: str | Path, language: str | None = None) -> str:
    """Extract a code block from an HTML page (string or file path).

    When *language* is given (e.g. "cpp", "python", "C++"), the block under the
    matching language heading/anchor is preferred over the first code block on
    the page - important for multi-language pages where the first ``<pre>`` is
    usually pseudocode. HTML entities are always unescaped.
    """
    text = (
        source.read_text(encoding="utf-8", errors="replace")
        if isinstance(source, Path)
        else source
    )
    if isinstance(source, str) and "\n" not in source and "<" not in source:
        # Looks like a path given as str.
        maybe = Path(source)
        if maybe.is_file():
            text = maybe.read_text(encoding="utf-8", errors="replace")

    candidates = _code_blocks(text)
    if not candidates:
        raise ToolkitError("no <pre>/<code> block found in the HTML")

    if language:
        aliases = _language_aliases(language)
        # 1) A code container explicitly tagged for the language (e.g. class
        #    "mw-highlight-lang-cpp" or "language-python").
        for block, context, _offset in candidates:
            ctx = context.lower()
            if any(
                f"lang-{a}" in ctx or f"language-{a}" in ctx or f"highlight-{a}" in ctx
                for a in aliases
            ):
                return block
        # 2) The first code block appearing AFTER the language's section heading.
        heading = _heading_position(text, aliases)
        if heading is not None:
            after = [b for b in candidates if b[2] > heading]
            if after:
                return after[0][0]
    return candidates[0][0]


def _code_blocks(text: str) -> list[tuple[str, str, int]]:
    """Return (clean_code, raw_container_html, start_offset) for each code block."""
    blocks: list[tuple[str, str, int]] = []
    # Prefer <pre> containers (optionally wrapped in a highlight <div>); capture
    # a little preceding context so class-based language tags are visible.
    for match in re.finditer(r"(?is)(<div[^>]*>\s*)?<pre[^>]*>(.*?)</pre>", text):
        raw_inner = match.group(2)
        context = text[max(0, match.start() - 200) : match.start() + 60]
        code = _html.unescape(re.sub(r"(?s)<[^>]+>", "", raw_inner)).strip("\n")
        if code.strip():
            blocks.append((code, context, match.start()))
    if not blocks:
        for match in re.finditer(r"(?is)<code[^>]*>(.*?)</code>", text):
            context = text[max(0, match.start() - 200) : match.start() + 60]
            code = _html.unescape(re.sub(r"(?s)<[^>]+>", "", match.group(1))).strip(
                "\n"
            )
            if code.strip():
                blocks.append((code, context, match.start()))
    return blocks


def _language_aliases(language: str) -> list[str]:
    lang = language.strip().lower()
    table = {
        "c++": ["cpp", "c++", "cxx"],
        "cpp": ["cpp", "c++", "cxx"],
        "c#": ["csharp", "cs", "c#"],
        "python": ["python", "py"],
        "javascript": ["javascript", "js"],
    }
    return table.get(lang, [re.sub(r"[^a-z0-9+#]", "", lang)])


def _heading_position(text: str, aliases: list[str]) -> int | None:
    """Offset of the content heading (h1-h6 / anchor id) for the language."""
    for alias in aliases:
        esc = re.escape(alias)
        patterns = [
            rf'(?is)<span[^>]*class="mw-headline"[^>]*id="{esc}"',
            rf'(?is)<h[1-6][^>]*id="{esc}"',
            rf'(?is)id="{esc}"[^>]*>\s*{esc}\s*<',
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.start()
    return None


def list_zip(path: str | Path) -> list[str]:
    with zipfile.ZipFile(Path(path)) as zf:
        return zf.namelist()


def extract_zip(path: str | Path, dest: str | Path = ".") -> list[str]:
    import shutil

    destination = Path(dest).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []

    with zipfile.ZipFile(Path(path)) as zf:
        for member in zf.infolist():
            name = member.filename
            if not name or name.endswith("/"):
                continue
            target = (destination / name).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise ToolkitError(f"zip member escapes destination: {name}") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(name)

    return extracted


__all__ = [
    "ToolkitError",
    "sniff_type",
    "read",
    "html_to_text",
    "extract_code",
    "list_zip",
    "extract_zip",
]
