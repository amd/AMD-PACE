# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from gaia_planner import sandbox_toolkit as tk

try:
    import openpyxl

    HAS_OPENPYXL = True
except ImportError:
    openpyxl = None  # type: ignore[assignment,misc]
    HAS_OPENPYXL = False


def _make_xlsx(path: Path) -> None:
    if openpyxl is None:
        raise unittest.SkipTest("openpyxl is not installed")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Name", "Value"])
    ws.append(["alpha", 3])
    wb.save(str(path))


class TestSandboxToolkit(unittest.TestCase):
    @unittest.skipUnless(HAS_OPENPYXL, "openpyxl is not installed")
    def test_sniff_type_uses_magic_bytes_over_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # An xlsx (zip container) saved WITHOUT an .xlsx extension must still be
            # recognized as xlsx, not misread as an image or unknown blob.
            blob = tmp_path / "attachment_no_ext"
            _make_xlsx(blob)
            self.assertEqual(tk.sniff_type(blob), "xlsx")

            pdf = tmp_path / "doc.bin"
            pdf.write_bytes(b"%PDF-1.4\n...trailer...")
            self.assertEqual(tk.sniff_type(pdf), "pdf")

            html = tmp_path / "page.txt"
            html.write_text("<!DOCTYPE html><html><body>hi</body></html>")
            self.assertEqual(tk.sniff_type(html), "html")

    @unittest.skipUnless(HAS_OPENPYXL, "openpyxl is not installed")
    def test_read_dispatches_by_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            xlsx = tmp_path / "data.xlsx"
            _make_xlsx(xlsx)
            text = tk.read(xlsx)
            self.assertIn("alpha", text)
            self.assertIn("3", text)

            plain = tmp_path / "notes.txt"
            plain.write_text("just text")
            self.assertEqual(tk.read(plain), "just text")

    def test_html_to_text_strips_tags_and_unescapes(self):
        html = "<html><body><p>a &lt; b &amp; c</p><script>x=1</script></body></html>"
        out = tk.html_to_text(html)
        self.assertIn("a < b & c", out)
        self.assertNotIn("x=1", out)  # script content dropped

    def test_extract_code_targets_language_section_over_first_block(self):
        # Page with pseudocode FIRST, then a C++ block tagged for the language.
        html = (
            "<h2><span class='mw-headline' id='Pseudo'>Pseudocode</span></h2>"
            "<pre><code>function quicksort(a)</code></pre>"
            "<h2><span class='mw-headline' id='C++'>C++</span></h2>"
            "<div class='mw-highlight mw-highlight-lang-cpp'><pre>"
            "#include &lt;vector&gt;\nint main(){return 0;}"
            "</pre></div>"
        )
        code = tk.extract_code(html, "C++")
        self.assertIn("#include <vector>", code)  # entities unescaped
        self.assertNotIn("function quicksort", code)  # skipped the pseudocode block

    def test_extract_code_defaults_to_first_block_without_language(self):
        html = "<pre>print('hello')</pre>"
        self.assertEqual(tk.extract_code(html), "print('hello')")

    def test_list_and_extract_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "bundle.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("a.txt", "A")
                zf.writestr("sub/b.txt", "B")
            self.assertEqual(set(tk.list_zip(archive)), {"a.txt", "sub/b.txt"})
            names = tk.extract_zip(archive, tmp_path / "out")
            self.assertEqual((tmp_path / "out" / "sub" / "b.txt").read_text(), "B")
            self.assertIn("a.txt", names)

    def test_missing_reader_raises_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            docx = tmp_path / "d.docx"
            # Minimal zip that sniffs as docx via member name.
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("word/document.xml", "<xml/>")
            docx.write_bytes(buf.getvalue())
            self.assertEqual(tk.sniff_type(docx), "docx")

            def fake_require(module, package=None):
                if module == "docx":
                    raise tk.ToolkitError(
                        "reader needs the 'python-docx' package - provision it "
                        "(pip: python-docx) and retry"
                    )
                return tk._require(module, package)

            with patch.object(tk, "_require", fake_require):
                with self.assertRaises(tk.ToolkitError) as exc:
                    tk.read(docx)
            self.assertIn("python-docx", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
