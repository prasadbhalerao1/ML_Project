# dev — working material

Everything used to *produce* the project that is not part of the running
system: the original brief, the presentation, the written report, and the
scripts that generate them.

Nothing here is imported by `backend/` or `frontend/`, and nothing here is
needed to run or deploy the application. Reference documentation lives in
[`../documentation/`](../documentation).

## Contents

| Path | What it is |
|---|---|
| [`task_claude.md`](task_claude.md) | The original project brief — the requirements this was built against |
| `deck/` | Seminar presentation and the printable handout |
| `report/` | The written project report |
| `build-scripts/` | Generators for the deck, report and technical sheet |
| `archive/` | Superseded drafts, kept until they are confirmed unwanted |

## deck/

| File | |
|---|---|
| `print.pdf` | **The one to print.** 2 pages, A4 landscape: 4 slides on page 1, technical sheet on page 2. Print double-sided. |
| `Loan_Default_Prediction_Seminar.pptx` | The 4-slide deck, for presenting |
| `Loan_Default_Prediction_Seminar.pdf` | Same deck as PDF |
| `Technical_Sheet_onepage.pdf` | The technical sheet alone, A4 portrait |

Slide 1 is the college's own title slide, taken from the department template
and edited in place; slides 2–4 are generated.

## report/

| File | |
|---|---|
| `Seminar_Report_Loan_Default_Prediction.docx` / `.pdf` | **The seminar report.** 23 pages in the department's 2026 format |
| `Loan_Default_Prediction_Report.docx` / `.pdf` | The earlier project report — 19 pages, Times New Roman 14 / 12 / 10 pt |
| `assets/` | The images both reports embed: live screenshots of the deployed app, and the documentation diagrams rendered to PNG |

The seminar report follows `seminar report format 2026.docx`: A4, Times New
Roman, 12 pt body, 16 pt chapter and 14 pt section headings, 1.5 line spacing,
justified, left margin 1.5 in and 1 in elsewhere, figure captions below and
table captions above, Roman numerals for the preliminary pages and Arabic from
Chapter 1. It runs title page → certificate → acknowledgement → abstract →
index → Chapters 1–7 → IEEE references, with 5 figures and 10 tables.

6 of the 23 pages are the front matter the template mandates; the content is
17 pages. Written deliberately tight — short paragraphs, tables and bullets
in preference to prose.

The page numbers in the index, list of figures and list of tables are written
literally, so if the text changes enough to repaginate, re-render the PDF and
correct them.

## build-scripts/

These read the trained model's saved files and regenerate the documents, so
the deck and report cannot contradict the model.

| Script | Builds | Run with |
|---|---|---|
| `build_deck.py` | The 4-slide `.pptx` | `python build_deck.py <project_root> <source.pptx> <out.pptx>` |
| `build_technical_sheet.py` | The A4 portrait sheet (HTML) | `python build_technical_sheet.py <project_root> <out.html>` |
| `build_print_handout.py` | The 2-page duplex handout (HTML) | `python build_print_handout.py <project_root> <slides_png_dir> <out.html>` |
| `build_report.js` | The earlier report `.docx` | `node build_report.js <project_root> <out.docx>` |
| `build_seminar_report.py` | The seminar report `.docx` | `python build_seminar_report.py <project_root> <assets_dir> <out.docx>` |
| `svg2png.py` | Documentation diagrams → PNG, for embedding | `python svg2png.py <svg_dir> <out_dir> 2.4 <chrome.exe>` |
| `export_slides.ps1` | Slide PNGs, via PowerPoint | `./export_slides.ps1 -In deck.pptx -OutDir png -Slides 1,2,3,4 -Width 2600` |
| `pptx_to_pdf.ps1`, `docx_to_pdf.ps1` | PDF conversion, via Office | `./pptx_to_pdf.ps1 -In a.pptx -Out a.pdf` |

The two HTML generators are rendered to PDF with headless Chrome:

```bash
chrome --headless=new --disable-gpu --no-pdf-header-footer \
       --print-to-pdf=out.pdf file:///abs/path/to/out.html
```

### Dependencies

`python-docx` and `python-pptx` for the Office generators, `pptxgenjs`-free
(the report uses the `docx` npm package) for `build_report.js`, plus Microsoft
Office for the PDF conversions and Chrome for the HTML ones. These are
authoring tools only — they are deliberately **not** in
`backend/requirements.txt`.

### If you retrain

Regenerate in this order, since each reads the artifact:

```bash
cd backend && python -m ml.train && python -m ml.evaluate --cross-validate
# then rebuild the deck, sheet and report from dev/build-scripts/
```

## archive/

Earlier attempts at the handout and one-pager, plus their render previews.
Kept because they were not all mine to delete — remove the folder once you are
satisfied with what is in `deck/`.
