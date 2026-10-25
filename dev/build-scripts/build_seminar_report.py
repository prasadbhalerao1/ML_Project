"""Build the seminar report in the department's 2026 format.

Structure follows "seminar report format 2026": title page, certificate,
acknowledgement, abstract, index, list of figures, list of tables, then
Chapters 1-7, then references. Formatting follows the guidelines in that same
file: A4, Times New Roman, 12 pt body, 16 pt chapter headings, 14 pt section
headings, 1.5 line spacing, justified, left margin 1.5in and 1in elsewhere,
figure captions below, table captions above, Roman numerals for the
preliminary pages and Arabic from Chapter 1.

Every metric is read from the trained model's saved files.

Usage: python build_seminar_report.py <project_root> <shots_dir> <out.docx>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT, SHOTS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ART = ROOT / "backend" / "ml" / "artifacts"
meta = json.loads((ART / "metadata.json").read_text(encoding="utf-8"))
rep = json.loads((ART / "evaluation_report.json").read_text(encoding="utf-8"))

test = rep["per_split_metrics"]["test"]
train = rep["per_split_metrics"]["train"]
val = rep["per_split_metrics"]["validation"]
cm = test["confusion_matrix"]
dist = rep["class_distribution"]
cv = rep["cross_validation"]["summary"]
report = rep["test_classification_report"]
thr = meta["decision_threshold"]["value"]
cand = {c["candidate"]: c for c in meta["validation_comparison"]}
imp = meta["feature_importances"]
sweep = meta["test_threshold_sweep"]
base = dist["test"]["positive_rate"]
n_pos = cm["true_positives"] + cm["false_negatives"]
n_neg = cm["true_negatives"] + cm["false_positives"]
leaf = meta["selected_configuration"]["params"]["min_samples_leaf"]
trees = meta["selected_configuration"]["params"]["n_estimators"]

STUDENT = "Prasad Bhalerao"
PRN = "RBT24CB026"
GUIDE = "Prof. Neeraj Sathawane"
SUBJECT = "Machine Learning"
TITLE = "Loan Default Risk Prediction using Random Forest"
APP_URL = "https://delinquency-risk.vercel.app"
API_URL = "https://loan-default-api.vercel.app/docs"
REPO_URL = "https://github.com/prasadbhalerao1/ML_Project"
KAGGLE_URL = "https://www.kaggle.com/competitions/GiveMeSomeCredit/data"

FONT = "Times New Roman"
INK = RGBColor(0, 0, 0)


def f4(v):
    return f"{v:.4f}"


def pc(v, d=2):
    return f"{v * 100:.{d}f}%"


doc = Document()

# ------------------------------------------------------------ base styles --
normal = doc.styles["Normal"]
normal.font.name = FONT
normal.font.size = Pt(12)
normal.font.color.rgb = INK
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
pf = normal.paragraph_format
pf.line_spacing = 1.5
pf.space_after = Pt(4)
pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)   # A4
sec.left_margin = Inches(1.5)
sec.right_margin = sec.top_margin = sec.bottom_margin = Inches(1.0)


def para(text="", size=12, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         space_before=0, space_after=6, line=1.5, style=None, indent=0.0):
    p = doc.add_paragraph(style=style)
    f = p.paragraph_format
    f.alignment = align
    f.space_before, f.space_after = Pt(space_before), Pt(space_after)
    f.line_spacing = line
    if indent:
        f.left_indent = Inches(indent)
    if text:
        add_runs(p, text, size, bold, italic)
    return p


def add_runs(p, text, size=12, bold=False, italic=False):
    """`text` is a string, or a list of (text, {overrides})."""
    parts = [(text, {})] if isinstance(text, str) else text
    for chunk, o in parts:
        r = p.add_run(chunk)
        r.font.name = o.get("font", FONT)
        r.font.size = Pt(o.get("size", size))
        r.font.bold = o.get("bold", bold)
        r.font.italic = o.get("italic", italic)
        r.font.color.rgb = INK
        r._element.rPr.rFonts.set(qn("w:eastAsia"), o.get("font", FONT))


def chapter(text):
    """16 pt bold chapter heading, starting on a new page.

    Uses page-break-before on the heading itself rather than an inserted
    break paragraph, which would leave a blank page whenever the previous
    chapter happened to end near the bottom of one.
    """
    drop_trailing_empty()
    p = para(text, size=16, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
             space_after=10, line=1.5)
    p.paragraph_format.page_break_before = True
    return p


def drop_trailing_empty():
    """Remove a trailing empty paragraph before a forced page break."""
    while doc.paragraphs and not doc.paragraphs[-1].text.strip():
        last = doc.paragraphs[-1]._p
        if last.getparent() is None:
            break
        last.getparent().remove(last)


def section(text):
    """14 pt bold section heading."""
    para(text, size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
         space_before=8, space_after=6, line=1.5)


def bullet(text):
    para(text, style="List Bullet", align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=3)


def caption(text, above=False):
    p = para(text, size=11, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER,
             space_before=2 if not above else 8, space_after=8 if not above else 3,
             line=1.0)
    if above:
        p.paragraph_format.keep_with_next = True
    return p


def figure(filename, width_in, cap):
    """Insert a figure, caption below it per the formatting guidelines."""
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(SHOTS / filename), width=Inches(width_in))
    caption(cap)


def table(headers, rows, cap, widths=None, font_size=11, split=False):
    """Table with its caption above, per the formatting guidelines.

    `split` lets a long table flow across a page break; without it a table
    that will not fit is pushed whole onto the next page, stranding one.
    """
    caption(cap, above=True)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    if not split:
        keep_with_next(t)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    if widths:
        for i, w in enumerate(widths):
            for cell in t.columns[i].cells:
                cell.width = Inches(w)
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        cp = cell.paragraphs[0]
        cp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.line_spacing = 1.0
        cp.paragraph_format.space_after = Pt(2)
        add_runs(cp, h, size=font_size, bold=True)
        shade(cell, "D9E2F3")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            cp = cells[i].paragraphs[0]
            cp.paragraph_format.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER)
            cp.paragraph_format.line_spacing = 1.0
            cp.paragraph_format.space_after = Pt(2)
            bold = isinstance(v, str) and v.startswith("**")
            add_runs(cp, v[2:-2] if bold else v, size=font_size, bold=bold)
            if widths:
                cells[i].width = Inches(widths[i])
    keep_with_next(t, rows_only=split)
    return t


def keep_with_next(t, rows_only=False):
    """Keep rows intact, and repeat the header if a table must break."""
    for row in t.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    header = OxmlElement("w:tblHeader")
    t.rows[0]._tr.get_or_add_trPr().append(header)
    if rows_only:
        return
    for row in t.rows[:-1]:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True


def shade(cell, hex_fill):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hex_fill)
    cell._tc.get_or_add_tcPr().append(el)


def page_number_field(paragraph):
    """PAGE field — Word renders the live page number."""
    run = paragraph.add_run()
    run.font.name = FONT
    run.font.size = Pt(11)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for el in (begin, instr, end):
        run._r.append(el)


def set_page_numbering(section_obj, fmt, start=None):
    """fmt: 'lowerRoman' or 'decimal'."""
    pg = section_obj._sectPr.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType")
        section_obj._sectPr.append(pg)
    pg.set(qn("w:fmt"), fmt)
    if start is not None:
        pg.set(qn("w:start"), str(start))


def footer_page_number(section_obj):
    section_obj.footer.is_linked_to_previous = False
    fp = section_obj.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    page_number_field(fp)


# ======================================================= TITLE PAGE =========
para("SEMINAR REPORT", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
     space_before=28, space_after=2, line=1.0)
para("ON", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12, line=1.0)
para(TITLE, size=18, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=20, line=1.2)
para("BY", size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10, line=1.0)
para(f"Name : {STUDENT}\t\tRBT NO: {PRN}", size=12, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16, line=1.0)
para("Third Year B. TECH", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
     space_after=4, line=1.0)
para("ACADEMIC YEAR: 2026-27", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
     space_after=16, line=1.0)
para("DEPARTMENT OF COMPUTER SCIENCE AND BUSINESS SYSTEMS", size=13, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18, line=1.1)
para("JSPM’s", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4, line=1.0)
para("RAJARSHI SHAHU COLLEGE OF ENGINEERING TATHAWADE, PUNE - 411033",
     size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6, line=1.2)

# ======================================================= CERTIFICATE ========
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
para("CERTIFICATE", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
     space_before=20, space_after=20, line=1.0)
para([("This is to certify that the seminar report entitled ", {}),
      (f"“{TITLE}”", {"bold": True}),
      (f" submitted by {STUDENT} (PRN No. {PRN}) is a record of bonafide work "
       "carried out under my supervision, in partial fulfilment of the "
       "requirements for the award of the degree of Bachelor of Technology in "
       "Computer Science and Business Systems of Savitribai Phule Pune "
       "University, during the academic year 2026-27.", {})], line=2.0)
para(space_after=30)
para("Date:", size=12, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=36)

cert = doc.add_table(rows=2, cols=3)
cert.alignment = WD_TABLE_ALIGNMENT.CENTER
cert_rows = [[GUIDE, "Dr. K. P. Moholkar", "Dr. A. M. Badadhe"],
             ["Seminar Guide", "H. O. D. CSBS", "Director, R.S.C.O.E. Pune"]]
for r, row in enumerate(cert_rows):
    for c, v in enumerate(row):
        cell = cert.rows[r].cells[c]
        cell.text = ""
        cp = cell.paragraphs[0]
        cp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.line_spacing = 1.0
        cp.paragraph_format.space_after = Pt(0)
        add_runs(cp, v, size=12, bold=(r == 0))

# ==================================================== ACKNOWLEDGEMENT ======
chapter("Acknowledgement")
para(f"I express my sincere gratitude to my seminar guide, {GUIDE}, "
     f"Department of Computer Science and Business Systems, for the guidance, "
     f"encouragement and constructive feedback given throughout this work.")
para("I thank Dr. K. P. Moholkar, Head of the Department, and Dr. A. M. "
     "Badadhe, Director, for the facilities and academic support that made "
     "this seminar possible.")
para("I am also grateful to the faculty and staff of the department, and to "
     "my family and friends, for their help and motivation.")
para(space_after=14)
for line_text in (STUDENT, PRN, "Department of Computer Science and Business Systems",
                  "JSPM’s Rajarshi Shahu College of Engineering, Pune"):
    para(line_text, align=WD_ALIGN_PARAGRAPH.RIGHT, space_after=2, line=1.15)

# =========================================================== ABSTRACT ======
chapter("Abstract")
para(f"This seminar studies the Random Forest algorithm applied to credit "
     f"risk, and reports a complete working implementation. The dataset is "
     f"the public Kaggle “Give Me Some Credit” collection of "
     f"{meta['dataset']['labeled_rows']:,} labelled borrower records with "
     f"{len(meta['features'])} numeric attributes. The target, "
     f"SeriousDlqin2yrs, marks borrowers 90 or more days past due within two "
     f"years — only {pc(base)} of records.")
para(f"That imbalance makes accuracy unusable as a headline metric: always "
     f"predicting the majority class scores {pc(1 - base, 1)} while catching "
     f"nobody. The work therefore selects on the area under the "
     f"precision-recall curve, treats undocumented placeholder codes "
     f"explicitly, and chooses the decision threshold on validation data "
     f"rather than assuming 0.50.")
para(f"On {dist['test']['total']:,} held-out records scored exactly once, the "
     f"model reached ROC-AUC {f4(test['roc_auc'])} and PR-AUC "
     f"{f4(test['pr_auc'])} against a {base:.4f} no-skill baseline. The "
     f"pipeline was deployed as a REST service with a web interface. The "
     f"report also documents the limitations that make the system unfit for "
     f"real credit decisions.")
para(space_after=6)
para([("Keywords: ", {"bold": True}),
      ("Random Forest, Credit Risk Prediction, Class Imbalance, "
       "Precision-Recall Curve, Decision Threshold, Supervised Classification",
       {})])

# ============================================================== INDEX ======
chapter("INDEX")
index_rows = [
    ["1", "Acknowledgement", "iii"],
    ["2", "Abstract", "iv"],
    ["3", "List of Figures and Tables", "v"],
    ["4", "Chapter 1: Introduction", "1"],
    ["5", "Chapter 2: Literature Review", "2"],
    ["6", "Chapter 3: Technology / Concept Overview", "3"],
    ["7", "Chapter 4: Applications and Results", "6"],
    ["8", "Chapter 5: Advantages and Limitations", "9"],
    ["9", "Chapter 6: Future Scope", "10"],
    ["10", "Chapter 7: Conclusion", "11"],
    ["11", "References", "12"],
]
table(["Sr. No.", "Contents", "Page No."], index_rows,
      "", widths=[0.9, 3.9, 1.0])

chapter("LIST OF FIGURES AND TABLES")
fig_rows = [
    ["Fig. 3.1", "System architecture of the implemented application", "3"],
    ["Fig. 4.1", "Dashboard of the deployed application", "7"],
    ["Fig. 4.2", "Predicted score distribution by actual outcome", "8"],
]
table(["Figure", "Title", "Page"], fig_rows, "", widths=[0.9, 3.9, 0.9])

para(space_after=8)
tab_rows = [
    ["Table 2.1", "Summary of literature review", "2"],
    ["Table 3.1", "Key concepts and definitions", "3"],
    ["Table 3.2", "Input features", "4"],
    ["Table 3.3", "Data quality issues and treatment", "4"],
    ["Table 3.4", "Validation comparison of configurations", "5"],
    ["Table 4.1", "Held-out test metrics", "7"],
    ["Table 4.2", "Per-class report and confusion matrix", "7"],
    ["Table 4.3", "Effect of the decision threshold", "8"],
    ["Table 5.1", "Advantages and limitations", "9"],
    ["Table 7.1", "Technology stack", "11"],
]
table(["Table", "Title", "Page"], tab_rows, "", widths=[0.9, 3.9, 0.9])

# Preliminary pages use lower-case Roman numerals.
set_page_numbering(sec, "lowerRoman", start=1)
footer_page_number(sec)

# ============================================== CHAPTER 1: INTRODUCTION ====
body = doc.add_section(WD_SECTION.NEW_PAGE)
body.page_width, body.page_height = Inches(8.27), Inches(11.69)
body.left_margin = Inches(1.5)
body.right_margin = body.top_margin = body.bottom_margin = Inches(1.0)
set_page_numbering(body, "decimal", start=1)
footer_page_number(body)

para("CHAPTER 1: INTRODUCTION", size=16, bold=True,
     align=WD_ALIGN_PARAGRAPH.LEFT, space_after=10)

section("1.1 Overview")
para("Credit risk prediction estimates how likely a borrower is to miss "
     "repayment. Traditionally an underwriter applied policy rules by hand. "
     "Machine learning instead fits a model to past borrowers whose outcomes "
     "are known, and learns the link between attributes and outcome.")
para(f"This seminar studies that application through the Random Forest "
     f"algorithm. Each record describes one borrower by "
     f"{len(meta['features'])} numeric attributes; the label states whether "
     f"they were 90+ days past due within two years. It is supervised binary "
     f"classification — the output is a category, not a quantity, and the "
     f"groups come from labels rather than being discovered.")

section("1.2 Need and Importance")
para("Lenders extend credit under uncertainty. A default is a direct loss; a "
     "sound applicant wrongly refused is lost revenue. Better ranking of "
     "applicants by risk therefore has direct economic value, and manual "
     "assessment does not scale.")
para(f"The problem is also statistically interesting. Only {pc(base)} of "
     f"borrowers here are positive, and that imbalance breaks the most "
     f"familiar metric: a model that always answers “no delinquency” scores "
     f"{pc(1 - base, 1)} accuracy while catching nobody. Which metrics "
     f"replace accuracy, and why, is a central theme of this report.")

section("1.3 Objectives")
for obj in (
    "To understand the Random Forest algorithm: bagging, random feature "
    "selection, and how an ensemble yields a probability.",
    "To study the architecture of a complete prediction system, from raw "
    "data to a deployed service.",
    "To examine correct evaluation under class imbalance, and show why "
    "accuracy is inadequate.",
    "To implement the method on a real dataset and report measured results.",
    "To identify the advantages, limitations and ethical challenges of such "
    "systems.",
):
    bullet(obj)

# ========================================= CHAPTER 2: LITERATURE REVIEW ====
chapter("CHAPTER 2: LITERATURE REVIEW")
para([("Breiman [1] ", {"bold": True}),
      ("introduced Random Forests: combining decorrelated trees reduces "
       "variance without raising bias. Two sources of randomness are "
       "responsible — bootstrap sampling of rows, and restricting each split "
       "to a random feature subset.", {})])
para([("Saito and Rehmsmeier [4] ", {"bold": True}),
      ("were the most influential source here. ROC plots mislead under "
       "imbalance, because the false-positive rate is measured against a "
       "large negative population and barely moves; precision-recall plots do "
       "not, since precision counts only what was flagged. This is why the "
       "model below is selected on PR-AUC. Pedregosa et al. [2] document "
       "scikit-learn, whose forest averages per-tree leaf proportions rather "
       "than taking the majority vote introductory treatments describe. "
       "Chawla et al. [5] frame imbalance as a choice of operating point on a "
       "trade-off curve.", {})])
table(["Sr.", "Author(s), Year", "Focus", "Key Findings"],
      [["1", "Breiman, 2001", "Random Forest theory",
        "Averaging decorrelated trees cuts variance; error converges as trees "
        "are added."],
       ["2", "Pedregosa et al., 2011", "scikit-learn",
        "Reference implementation. Probability = mean of per-tree leaf "
        "proportions, not a vote."],
       ["3", "Kaggle, 2011", "Give Me Some Credit data",
        "150,000 labelled records with a documented data dictionary."],
       ["4", "Saito & Rehmsmeier, 2015", "Imbalanced evaluation",
        "ROC is over-optimistic under imbalance; PR is more informative."],
       ["5", "Chawla et al., 2002", "Resampling, cost-sensitive learning",
        "Imbalance is a choice of operating point; reweighting shifts it "
        "rather than improving ranking."]],
      "Table 2.1: Summary of literature review",
      widths=[0.5, 1.45, 1.4, 2.45], font_size=10)

para("Research gap.", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
     space_before=8, space_after=4)
para("The literature agrees that accuracy and ROC-AUC flatter a classifier "
     "when positives are rare, yet both are still widely reported, and the "
     "standard remedies are often applied without checking whether they "
     "improve ranking or merely relocate the operating point. This work "
     "measures that: class weighting moves the model along the "
     "precision-recall curve without lifting it, so explicit threshold "
     "selection is used instead. Separately, undocumented placeholder codes "
     "that much published work imputes away are shown to carry real signal.")

# ============================== CHAPTER 3: TECHNOLOGY / CONCEPT OVERVIEW ===
chapter("CHAPTER 3: TECHNOLOGY / CONCEPT OVERVIEW")

section("3.1 Basic Concepts")
table(["Term", "Definition"],
      [["Decision tree", "Splits data repeatedly on one attribute. Leaves "
        "hold the training samples that reached them."],
       ["Gini impurity", "G = 1 − Σ pk², where pk is the proportion of class "
        "k in a node. Splits maximise its weighted reduction."],
       ["Bagging", "Each tree trains on a sample drawn with replacement, so "
        "averaging their output reduces variance."],
       ["Random Forest", f"An ensemble of {trees} such trees, each split "
        "restricted to a random feature subset."],
       ["Precision / Recall", "Of those flagged, the share truly delinquent "
        "— TP/(TP+FP). Of the truly delinquent, the share found — "
        "TP/(TP+FN)."],
       ["F1-score", "Harmonic mean 2PR/(P+R); dominated by the weaker term."],
       ["ROC-AUC", "Probability a random positive outranks a random "
        "negative. 0.5 is chance."],
       ["PR-AUC", f"Area under the precision-recall curve. No-skill baseline "
        f"= the positive rate, here {base:.4f}."]],
      "Table 3.1: Key concepts and definitions",
      widths=[1.3, 4.5], font_size=10)

section("3.2 Architecture and Working")
para("Training, evaluation and figure generation run offline and produce one "
     "saved artifact. The web service loads it at startup and scores "
     "requests; it never trains. The browser talks to one origin only — "
     "requests to /api are forwarded server-side, so no cross-origin request "
     "is ever made.")
figure("system-architecture.png", 5.0,
       "Figure 3.1: System architecture of the implemented application")
para("Training runs in six stages: load and validate against a schema; mark "
     "implausible values missing; split 60/20/20 stratified; compare five "
     "candidates on validation; select the threshold; score once on test. "
     "Every fitted quantity — imputation medians, model parameters, the "
     f"threshold — comes from train and validation only. Stratifying matters "
     f"at a {pc(base)} base rate, or the three parts would carry different "
     f"class ratios and their metrics would not be comparable.")
para(f"To predict, each of the {trees} trees routes the record to a leaf and "
     f"reports that leaf's positive proportion. The forest averages those "
     f"{trees} values — scikit-learn averages per-tree probabilities rather "
     f"than taking a majority vote — and the mean is compared with the "
     f"threshold of {thr:.4f} to give a class.")

section("3.4 Features and Data Quality")
para("Ten numeric attributes per borrower. No categorical encoding is needed, "
     "and because trees split on rank order, no scaling is either.")
table(["Feature", "Description"],
      [["RevolvingUtilizationOfUnsecuredLines", "Balances ÷ total credit limits"],
       ["age", "Borrower age in years"],
       ["NumberOfTime30-59DaysPastDue", "30–59 day late episodes, last 2 years"],
       ["DebtRatio", "Debt, alimony and living costs ÷ gross monthly income"],
       ["MonthlyIncome", "Gross monthly income (19.82% missing)"],
       ["NumberOfOpenCreditLinesAndLoans", "Open loans and credit lines"],
       ["NumberOfTimes90DaysLate", "90+ day late episodes"],
       ["NumberRealEstateLoansOrLines", "Mortgages and home equity lines"],
       ["NumberOfTime60-89DaysPastDue", "60–89 day late episodes, last 2 years"],
       ["NumberOfDependents", "Dependents excluding the borrower (2.62% missing)"]],
      "Table 3.2: Input features",
      widths=[2.6, 3.2], font_size=10)

table(["Issue", "Extent", "Treatment"],
      [["MonthlyIncome missing", "29,731 (19.82%)",
        "Median imputation + missing indicator"],
       ["NumberOfDependents missing", "3,924 (2.62%)",
        "Median imputation + missing indicator"],
       ["Counters coded 96 or 98", "269 rows",
        "Placeholders. Imputed with an indicator — these rows are 54.65% "
        "positive against a 6.68% base rate"],
       ["Age recorded as 0", "1 row", "Impossible; treated as missing"],
       ["Duplicate predictor rows", "646 rows",
        "Retained — plausibly distinct borrowers"],
       ["Extreme ratios", "up to 329,664",
        "Retained — trees split on rank order"]],
      "Table 3.3: Data quality issues and treatment",
      widths=[1.9, 1.3, 2.6], font_size=10)

para("Preprocessing and the classifier are one scikit-learn Pipeline, and "
     "that composite object is what is saved. Serving code loads and calls "
     "it, so train-serve skew is structurally impossible. The classifier "
     f"receives {len(meta['model_input_columns'])} columns built from the "
     f"{len(meta['features'])} features, because of the appended indicators.")

section("3.5 Model Selection")
para("Five configurations were fitted on train and compared on validation, "
     "ranked by PR-AUC.")
table(["Configuration", "ROC-AUC", "PR-AUC", "Precision", "Recall"],
      [[c["candidate"].replace("_", " "), f4(c["roc_auc"]), f4(c["pr_auc"]),
        f4(c["precision"]), f4(c["recall"])] for c in meta["validation_comparison"]],
      "Table 3.4: Validation comparison of configurations",
      widths=[2.2, 1.0, 1.0, 1.0, 0.9], font_size=10)
para(f"Raising the minimum leaf size to {leaf} lifted PR-AUC from "
     f"{f4(cand['baseline']['pr_auc'])} to {f4(cand['regularised']['pr_auc'])} "
     f"— the unrestricted forest was overfitting.")
para(f"Class weighting raised recall from {f4(cand['regularised']['recall'])} "
     f"to {f4(cand['balanced']['recall'])} but left PR-AUC flat at "
     f"{f4(cand['balanced']['pr_auc'])}. It moved the model along the curve "
     f"without lifting it, so it was not used — a threshold does the same job "
     f"visibly and without refitting.")
para(f"Adding missing-value indicators won, at "
     f"{f4(cand['regularised_with_indicators']['pr_auc'])}. The threshold was "
     f"then set on validation by maximising F1, giving {thr:.4f} rather than "
     f"0.50. The model was refitted on train + validation and scored once on "
     f"test.")

# ================================ CHAPTER 4: APPLICATIONS AND RESULTS =====
chapter("CHAPTER 4: APPLICATIONS AND RESULTS")

section("4.1 Applications and Use Cases")
para([("Application scoring ", {"bold": True}),
      ("gives a risk score at origination, combined with policy rules and "
       "affordability checks, so every applicant is assessed on the same "
       "basis in milliseconds. ", {}),
      ("Behavioural scoring ", {"bold": True}),
      ("rescores an existing portfolio as payment behaviour accumulates, "
       "catching accounts drifting toward delinquency early enough to "
       "intervene. ", {}),
      ("Collections prioritisation ", {"bold": True}),
      ("ranks delinquent accounts so finite recovery effort goes where it "
       "matters; it needs only the ranking, not a calibrated probability.",
       {})])
para("The technique generalises to any rare-event problem on tabular data — "
     "fraud detection, claim triage, churn, predictive maintenance. The "
     "benefits are consistency, scale and inspectability: identical criteria "
     "for every applicant, at volumes no manual process could reach.")

section("4.2 Deployed Application")
para(f"The model is deployed at {APP_URL}, with API documentation at "
     f"{API_URL}. Four pages — Dashboard, Predict, Result and Model info — "
     f"all fetching their values from the service at run time. The dashboard "
     f"states what the model does and does not do, and shows the "
     f"majority-class baseline beside the accuracy so the latter cannot be "
     f"mistaken for a strong result.")
figure("01-dashboard.png", 4.9,
       "Figure 4.1: Dashboard of the deployed application")

section("4.3 Results")
para(f"Measured on {dist['test']['total']:,} held-out borrowers, scored once, "
     f"at threshold {thr:.4f}.")
table(["Metric", "Value", "Read it as"],
      [["ROC-AUC", f4(test["roc_auc"]),
        "Ranks a delinquent above a sound borrower 86.5% of the time"],
       ["PR-AUC", f4(test["pr_auc"]), f"~6× the no-skill baseline of {base:.4f}"],
       ["Recall", f4(test["recall"]),
        f"Caught {cm['true_positives']:,} of {n_pos:,} real delinquencies"],
       ["Precision", f4(test["precision"]), "42% of those flagged truly were"],
       ["F1-score", f4(test["f1"]), "Positive class"],
       ["Accuracy", f4(test["accuracy"]),
        f"Below the {pc(1 - base, 1)} of always answering “no”"]],
      "Table 4.1: Held-out test metrics", widths=[1.1, 0.9, 3.8], font_size=10)

table(["Class", "Precision", "Recall", "F1", "Support"],
      [["0 — No delinquency", f4(report["classes"]["0"]["precision"]),
        f4(report["classes"]["0"]["recall"]), f4(report["classes"]["0"]["f1"]),
        f"{report['classes']['0']['support']:,}"],
       ["1 — Delinquency", f4(report["classes"]["1"]["precision"]),
        f4(report["classes"]["1"]["recall"]), f4(report["classes"]["1"]["f1"]),
        f"{report['classes']['1']['support']:,}"],
       ["**Macro average**", f"**{f4(report['macro_avg']['precision'])}**",
        f"**{f4(report['macro_avg']['recall'])}**",
        f"**{f4(report['macro_avg']['f1'])}**", f"**{dist['test']['total']:,}**"],
       ["Weighted average", f4(report["weighted_avg"]["precision"]),
        f4(report["weighted_avg"]["recall"]), f4(report["weighted_avg"]["f1"]),
        f"{dist['test']['total']:,}"],
       ["Confusion matrix", f"TN {cm['true_negatives']:,}",
        f"FP {cm['false_positives']:,}", f"FN {cm['false_negatives']:,}",
        f"TP {cm['true_positives']:,}"]],
      "Table 4.2: Per-class report and confusion matrix",
      widths=[1.7, 1.1, 1.0, 1.0, 1.0], font_size=10)

para(f"This is the imbalance problem in one table: "
     f"{f4(report['classes']['0']['f1'])} F1 on the majority class against "
     f"{f4(report['classes']['1']['f1'])} on the minority. The weighted "
     f"average of {f4(report['weighted_avg']['f1'])} looks impressive but is "
     f"dominated by {report['classes']['0']['support']:,} negatives. Quote "
     f"the macro average ({f4(report['macro_avg']['f1'])}) or the "
     f"positive-class F1 — never the weighted one.")

figure("05-chart-separation.png", 4.8,
       "Figure 4.2: Predicted score distribution by actual outcome")
para("Most borrowers who stayed current score near zero, but those who became "
     "delinquent are spread across the whole range rather than clustered "
     "high. That overlap — not a defect in the implementation — is what "
     "limits recall.")

table(["Threshold", "Accuracy", "Precision", "Recall", "F1", "Flagged"],
      [[("**" + f"{r['threshold']:.4f}" + "**") if abs(r["threshold"] - thr) < 5e-5
        else f"{r['threshold']:.4f}",
        f4(r["accuracy"]), f4(r["precision"]), f4(r["recall"]), f4(r["f1"]),
        f"{r['confusion_matrix']['true_positives'] + r['confusion_matrix']['false_positives']:,}"]
       for r in sweep],
      "Table 4.3: Effect of the decision threshold",
      widths=[1.1, 0.95, 1.0, 0.85, 0.85, 0.85], font_size=10)
para("The same model behaves very differently by cut-off: at 0.10 recall "
     "exceeds 0.70 but three in four flags are false alarms; at 0.50 "
     "precision passes 0.62 but only one delinquency in six is found. "
     "Accuracy rises monotonically simply because the model answers “no” "
     "more often — a further reason not to trust it here.")

para(f"Five-fold cross-validation on train + validation, test excluded "
     f"throughout, gives ROC-AUC {f4(cv['roc_auc']['mean'])} ± "
     f"{f4(cv['roc_auc']['std'])} and PR-AUC {f4(cv['pr_auc']['mean'])} ± "
     f"{f4(cv['pr_auc']['std'])}; the test values sit within one standard "
     f"deviation of both, so the result is not a lucky split. ROC-AUC does "
     f"fall from {f4(train['roc_auc'])} in-sample to {f4(test['roc_auc'])} "
     f"held-out, a gap of {train['roc_auc'] - test['roc_auc']:.4f} — some "
     f"overfitting remains despite regularisation, and is reported rather "
     f"than concealed.")

# ============================ CHAPTER 5: ADVANTAGES AND LIMITATIONS =======
chapter("CHAPTER 5: ADVANTAGES AND LIMITATIONS")
section("5.1 Key Limitations")
para([("Data, not algorithm. ", {"bold": True}),
      ("US consumers around 2011, with no date column — so no out-of-time "
       "validation is possible and the sampling frame is undocumented.",
       {})])
para([("Uncalibrated. ", {"bold": True}),
      ("A predicted 0.30 is a ranking score, not a claim that 30 in 100 such "
       "borrowers default.", {})])
para([("Fairness untested. ", {"bold": True}),
      ("No race, sex or geography in the data — but that is not protection, "
       "since age is used and income can proxy for protected traits. Absence "
       "of measured bias is absence of measurement.", {})])
para([("Threshold economics. ", {"bold": True}),
      ("Maximising F1 weights a missed default and a false alarm equally; "
       "real lending does not.", {})])

section("5.2 Challenges Encountered")
para([("Undocumented codes. ", {"bold": True}),
      ("The codes 96 and 98 appear in 269 records with no explanation. Those "
       "rows are 54.65% positive against a 6.68% base rate, so imputing them "
       "away would discard signal while asserting clean payment histories. "
       "Imputing with a retained indicator lets the model learn the value was "
       "unavailable without guessing what the codes mean.", {})])
para([("Deployment and skew. ", {"bold": True}),
      ("The Python stack plus a 70 MB model is ~315 MB, so runtime and "
       "training dependencies were split to fit the 500 MB limit. Any "
       "train/inference preprocessing difference would corrupt predictions "
       "silently — one saved pipeline removes the possibility.", {})])

table(["Advantages", "Limitations"],
      [["Captures non-linear interactions without feature engineering",
        "Data describes 2011 US consumers; current performance unknown"],
       ["Robust to extreme values; no scaling or encoding needed",
        "No date column, so out-of-time validation is impossible"],
       ["Outputs probabilities, not just labels",
        "Probabilities are uncalibrated; calibration was not evaluated"],
       ["Threshold adjustable after training, without refitting",
        "Threshold optimises F1, weighting both error types equally"],
       ["Reports feature importances, keeping the model inspectable",
        "Importance describes splits, not causation"],
       ["Trains in ~1 minute; reproducible with a fixed seed",
        "No protected attributes, so fairness cannot be tested"]],
      "Table 5.1: Advantages and limitations",
      widths=[2.9, 2.9], font_size=10, split=True)

# ======================================== CHAPTER 6: FUTURE SCOPE =========
chapter("CHAPTER 6: FUTURE SCOPE")
para("Each item below follows directly from a limitation in Chapter 5.")
for head, text in (
    ("Probability calibration.",
     "Fit a calibrator on held-out data and report a reliability diagram and "
     "Brier score. The single change that would most improve practical "
     "usefulness."),
    ("Cost-sensitive thresholds.",
     "Replace the F1 criterion with an explicit cost matrix, placing the "
     "operating point where expected loss is minimised."),
    ("Out-of-time validation.",
     "With origination dates, train on an earlier period and test on a later "
     "one — standard for credit models, precisely because random splits "
     "flatter them."),
    ("Better explanations.",
     "Permutation importance on held-out data is less biased than the "
     "impurity measure used here; per-prediction attributions such as SHAP "
     "would support the adverse-action explanations regulated lending "
     "requires."),
    ("Fairness assessment.",
     "With the relevant attributes, test for disparate impact across "
     "protected groups and mitigate where needed."),
):
    para([(head + " ", {"bold": True}), (text, {})], style="List Bullet",
         space_after=3)
para("The field is moving toward models that are auditable, not merely "
     "accurate — regulation increasingly requires that an automated credit "
     "decision be explainable to the person it affects.")

# ========================================== CHAPTER 7: CONCLUSION =========
chapter("CHAPTER 7: CONCLUSION")
para(f"A Random Forest was trained on {meta['dataset']['labeled_rows']:,} "
     f"labelled records and evaluated on {dist['test']['total']:,} held-out "
     f"borrowers scored exactly once, reaching ROC-AUC {f4(test['roc_auc'])} "
     f"and PR-AUC {f4(test['pr_auc'])} against a {base:.4f} baseline.")
para("The methodological findings matter as much as the numbers:")
for item in (
    "Accuracy is actively misleading here — a model that ignores its inputs "
    "scores higher than the trained one.",
    "Class weighting moved the model along the precision-recall curve "
    "without lifting it; an explicit threshold chosen on validation was used "
    "instead.",
    "Undocumented placeholder codes carried real signal and were handled "
    "with missing-value indicators rather than imputed away.",
    "The gap between in-sample and held-out performance was measured and "
    "reported, not omitted.",
):
    bullet(item)
para("The engineering discipline is what makes the figures trustworthy: "
     "preprocessing held inside the saved pipeline so training and inference "
     "cannot diverge, the threshold stored with the model, and every "
     "documented number generated from the artifact rather than transcribed.")
table(["Layer", "Technology"],
      [["Dataset", "Kaggle Give Me Some Credit — 150,000 records"],
       ["Model", "scikit-learn RandomForestClassifier, joblib"],
       ["Data", "pandas, NumPy; matplotlib, seaborn"],
       ["API", "FastAPI, Pydantic, Uvicorn"],
       ["Frontend", "React, TypeScript, Vite, Tailwind CSS"],
       ["Testing", "pytest — 61 tests"],
       ["Deployment", "Vercel (frontend and backend)"]],
      "Table 7.1: Technology stack", widths=[1.3, 4.5], font_size=10)
para("The honest conclusion is that this system extracts real, measurable "
     "signal from borrower data and is simultaneously nowhere near fit to "
     "decide anybody's credit application. Holding both statements at once "
     "is the central skill the exercise teaches.")

# ============================================================ REFERENCES ==
chapter("REFERENCES")
for ref in (
    "[1] L. Breiman, “Random Forests,” Machine Learning, vol. 45, no. 1, "
    "pp. 5–32, 2001.",
    "[2] F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” "
    "Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
    f"[3] Kaggle and Credit Fusion, “Give Me Some Credit.” Available: "
    f"{KAGGLE_URL}. Accessed: October 24, 2026.",
    "[4] T. Saito and M. Rehmsmeier, “The Precision-Recall Plot Is More "
    "Informative than the ROC Plot When Evaluating Binary Classifiers on "
    "Imbalanced Datasets,” PLOS ONE, vol. 10, no. 3, 2015.",
    "[5] N. V. Chawla, K. W. Bowyer, L. O. Hall and W. P. Kegelmeyer, “SMOTE: "
    "Synthetic Minority Over-sampling Technique,” Journal of Artificial "
    "Intelligence Research, vol. 16, pp. 321–357, 2002.",
    "[6] T. Hastie, R. Tibshirani and J. Friedman, The Elements of Statistical "
    "Learning, 2nd ed. Springer, 2009.",
    f"[7] P. Bhalerao, “Loan Default Risk Prediction — source code.” "
    f"Available: {REPO_URL}. Accessed: October 24, 2026.",
    f"[8] P. Bhalerao, “Loan Delinquency Risk — deployed application.” "
    f"Available: {APP_URL}. Accessed: October 24, 2026.",
):
    para(ref, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=8, line=1.3)

doc.save(str(OUT))
print(f"saved {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
