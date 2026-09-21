"""
stamp_page_numbers.py — add a footer with "page X of Y" to a PDF.

Used by scripts/build_pdf.jl after headless Chrome has printed the notebook
printout. Chrome's command line cannot emit page numbers (only the DevTools
protocol can), so the footer is stamped here instead.

    python scripts/stamp_page_numbers.py input.pdf output.pdf

Requires: pypdf, reportlab  (`python -m pip install pypdf reportlab`)
Exits with status 1 (and prints a note) when those are missing, so the build
continues with the un-numbered PDF.
"""
import io
import sys

try:
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except ImportError as exc:  # pragma: no cover - optional dependency
    print(f"page numbers skipped: {exc}")
    sys.exit(1)

FOOTER_LEFT = "Carbon Accounting in Julia — Pluto notebook printout"
MARGIN = 34
BASELINE = 22


def stamp(src: str, dst: str) -> int:
    reader = PdfReader(src)
    total = len(reader.pages)
    writer = PdfWriter()
    for number, page in enumerate(reader.pages, start=1):
        buffer = io.BytesIO()
        canvas_obj = canvas.Canvas(buffer, pagesize=A4)
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillGray(0.45)
        canvas_obj.drawString(MARGIN, BASELINE, FOOTER_LEFT)
        canvas_obj.drawRightString(A4[0] - MARGIN, BASELINE, f"page {number} of {total}")
        canvas_obj.save()
        buffer.seek(0)
        page.merge_page(PdfReader(buffer).pages[0])
        writer.add_page(page)
    with open(dst, "wb") as handle:
        writer.write(handle)
    print(f"wrote {dst} ({total} pages, numbered)")
    return total


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    stamp(sys.argv[1], sys.argv[2])
