"""Build the print PDF: inline fonts, force the light palette, add metadata."""
import io
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

FONT_LINK = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
             'family=Spectral:ital,wght@0,300;0,400;0,600;1,400'
             '&family=Archivo:wght@400;500;600'
             '&family=JetBrains+Mono:wght@400;500&display=swap">')

PRINT_CSS = """
<style>
/* Print build: a PDF has no viewer theme, so the light palette is forced. */
:root, :root[data-theme="dark"] {
  --paper:#FFFFFF; --surface:#FAFAFC; --ink:#14181F; --muted:#4E5866;
  --faint:#7C8695; --rule:#CFD5DE; --rule-soft:#E4E8EE;
  --confound:#8F5606; --method:#08514E; --chance:#98A1AE;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:#FFFFFF; --surface:#FAFAFC; --ink:#14181F; --muted:#4E5866;
    --faint:#7C8695; --rule:#CFD5DE; --rule-soft:#E4E8EE;
    --confound:#8F5606; --method:#08514E; --chance:#98A1AE;
  }
}
@page { size: A4; margin: 20mm 18mm 22mm; }
html, body { background:#FFFFFF !important; }
body { font-size: 10.3pt; line-height: 1.5; }
.page { max-width: none; padding: 0; gap: 1.4rem; }
:root { --measure: 100%; }
h1 { font-size: 21pt; line-height: 1.12; }
.standfirst { font-size: 11pt; }
.byline { font-size: 8.2pt; gap: 0.22rem 1.3rem; }
.eyebrow { font-size: 7.2pt; }
h2.sec { font-size: 13pt; margin-top: 0.4rem; }
h3.sub { font-size: 10.3pt; }
figcaption { font-size: 8.3pt; }
table { font-size: 8pt; }
th, td { padding: 0.3rem 0.45rem; }
.abstract { padding: 0.95rem 1.1rem; }
.claim { padding: 0.85rem 0.95rem; }
.claim .stmt { font-size: 10.2pt; }
.claim .ev { font-size: 7.8pt; }
.refs li { font-size: 8.5pt; }
footer { font-size: 8pt; }
figure, .abstract, .claim, .tablewrap, .figbody { break-inside: avoid; }
h2.sec, h3.sub { break-after: avoid; }
p, li { orphans: 3; widows: 3; }
a { color: var(--method) !important; }
</style>
"""


def build_print_html() -> Path:
    src = (HERE / "cough-geography-paper.html").read_text(encoding="utf-8")
    fonts = (HERE / "fonts-inline.css").read_text(encoding="utf-8")
    if FONT_LINK not in src:
        raise SystemExit("font link not found; the header changed")
    out = src.replace(FONT_LINK, f"<style>\n{fonts}\n</style>") + PRINT_CSS
    path = HERE / "paper-print.html"
    path.write_text(out, encoding="utf-8")
    return path


def render(html: Path, pdf: Path) -> None:
    subprocess.run([
        CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--virtual-time-budget=40000",
        "--run-all-compositor-stages-before-draw",
        # Chrome renamed this flag; passing both covers either build.
        # Without it the PDF carries a header with the local file path.
        "--no-pdf-header-footer",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf}",
        html.resolve().as_uri(),
    ], check=True, capture_output=True, timeout=300)


def finish(raw: Path, final: Path) -> None:
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    reader = PdfReader(str(raw))
    n = len(reader.pages)

    # Running footer: the page number, and the short title from page two on.
    overlay = io.BytesIO()
    c = canvas.Canvas(overlay, pagesize=A4)
    for i in range(n):
        c.setFont("Helvetica", 7.5)
        c.setFillGray(0.45)
        if i:
            c.drawString(51, 30, "Site-invariant representations for cough-audio screening")
        c.drawRightString(A4[0] - 51, 30, f"{i + 1} of {n}")
        c.showPage()
    c.save()
    overlay.seek(0)

    marks = PdfReader(overlay)
    writer = PdfWriter()
    for i, page in enumerate(reader.pages):
        page.merge_page(marks.pages[i])
        writer.add_page(page)

    writer.add_metadata({
        "/Title": ("Site-invariant representations for cough-audio screening: "
                   "removing the confound does not recover disease signal"),
        "/Author": "Abdoulie Balisa",
        "/Subject": ("Leave-one-country-out evaluation of adversarial domain-invariance "
                     "on 2,739 COUGHVID cough recordings across nine countries."),
        "/Keywords": ("cough audio, domain adaptation, gradient reversal, "
                      "shortcut learning, tuberculosis screening, COUGHVID, "
                      "negative result, leave-one-country-out"),
        "/Creator": "Abdoulie Balisa",
    })
    with final.open("wb") as fh:
        writer.write(fh)


def main() -> int:
    html = build_print_html()
    raw = HERE / "paper-raw.pdf"
    final = HERE / "Balisa-2026-site-invariant-cough-screening.pdf"
    render(html, raw)
    finish(raw, final)

    from pypdf import PdfReader
    r = PdfReader(str(final))
    fonts = set()
    for p in r.pages:
        try:
            res = p["/Resources"]["/Font"]
            for k in res:
                fonts.add(str(res[k].get("/BaseFont", "?")))
        except Exception:
            pass
    print(f"pages: {len(r.pages)}")
    print(f"author: {r.metadata.get('/Author')}")
    print(f"size: {final.stat().st_size / 1024:.0f} KB")
    print("fonts:")
    for f in sorted(fonts):
        print("  ", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
