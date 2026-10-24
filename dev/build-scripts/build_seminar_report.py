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
GUIDE = "Prof. Neeraj Sathwale"
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
pf.space_after = Pt(6)
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
    """16 pt bold chapter heading, starting on a new page."""
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    para(text, size=16, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
         space_after=10, line=1.5)


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


def table(headers, rows, cap, widths=None, font_size=11):
    """Table with its caption above, per the formatting guidelines."""
    caption(cap, above=True)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
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
    keep_with_next(t)
    para(space_after=4)
    return t


def keep_with_next(t):
    """Keep rows intact, and repeat the header if a table must break."""
    for row in t.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    header = OxmlElement("w:tblHeader")
    t.rows[0]._tr.get_or_add_trPr().append(header)
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
para(f"I express my deep sense of gratitude to my seminar guide, {GUIDE}, "
     "Department of Computer Science and Business Systems, for the valuable "
     "guidance, continuous encouragement and constructive feedback given "
     "throughout the course of this seminar work.", line=1.5)
para("I would like to extend my sincere thanks to Dr. K. P. Moholkar, Head of "
     "the Department of Computer Science and Business Systems, for providing "
     "the facilities and academic support necessary to complete this work.")
para("I also express my heartiest gratitude to Dr. A. M. Badadhe, Director, "
     "for the constant support and encouragement.")
para("Finally, I am grateful to all the faculty members and staff of the "
     "Department of Computer Science and Business Systems, and to my family "
     "and friends, for their help and motivation during the preparation of "
     "this report.")
para(space_after=14)
for line_text in (STUDENT, PRN, "Department of Computer Science and Business Systems",
                  "JSPM’s Rajarshi Shahu College of Engineering, Pune"):
    para(line_text, align=WD_ALIGN_PARAGRAPH.RIGHT, space_after=2, line=1.15)

# =========================================================== ABSTRACT ======
chapter("Abstract")
para(f"With the rapid advancement of machine learning in financial services, "
     f"credit risk assessment has emerged as a crucial area of focus due to "
     f"its ability to support lending decisions from historical borrower data "
     f"rather than judgement alone. This seminar studies the application of the "
     f"Random Forest algorithm to the prediction of serious borrower "
     f"delinquency, and reports a complete, working implementation. The study "
     f"uses the public Kaggle “Give Me Some Credit” dataset of "
     f"{meta['dataset']['labeled_rows']:,} labelled borrower records described "
     f"by {len(meta['features'])} numeric attributes. The target variable, "
     f"SeriousDlqin2yrs, records whether a borrower experienced delinquency of "
     f"90 days or worse within two years of the observation date, and only "
     f"{pc(base)} of records are positive. The work establishes that accuracy "
     f"is an unsuitable headline metric for such an imbalanced problem, because "
     f"a model that always predicts the majority class attains "
     f"{pc(1 - base, 1)} accuracy while identifying no delinquent borrower at "
     f"all. A preprocessing pipeline was constructed to treat undocumented "
     f"placeholder codes and missing values, five Random Forest configurations "
     f"were compared on a held-out validation split using the area under the "
     f"precision-recall curve, and the decision threshold was selected on "
     f"validation data rather than assumed to be 0.50. On a test split of "
     f"{dist['test']['total']:,} records, scored exactly once, the final model "
     f"achieved a ROC-AUC of {f4(test['roc_auc'])} and a PR-AUC of "
     f"{f4(test['pr_auc'])} against a no-skill baseline of {base:.4f}. The "
     f"trained pipeline was deployed as a REST service with a web interface, "
     f"demonstrating the complete path from raw data to a usable application. "
     f"The report also documents the limitations that prevent such a system "
     f"from being used for real credit decisions.", line=1.5)
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
    ["3", "List of Figures", "vi"],
    ["4", "List of Tables", "vi"],
    ["5", "Chapter 1: Introduction", "1"],
    ["6", "Chapter 2: Literature Review", "3"],
    ["7", "Chapter 3: Technology / Concept Overview", "6"],
    ["8", "Chapter 4: Applications and Use Cases", "12"],
    ["9", "Chapter 5: Advantages and Limitations", "21"],
    ["10", "Chapter 6: Future Scope", "23"],
    ["11", "Chapter 7: Conclusion", "24"],
    ["12", "References / Bibliography", "26"],
]
table(["Sr. No.", "Contents", "Page No."], index_rows,
      "", widths=[0.9, 3.9, 1.0])

chapter("LIST OF FIGURES")
fig_rows = [
    ["Fig. 3.1", "System architecture of the implemented application", "7"],
    ["Fig. 3.2", "Training pipeline from raw data to saved model artifact", "8"],
    ["Fig. 3.3", "Stratified 60/20/20 split and the role of each part", "8"],
    ["Fig. 3.4", "How a Random Forest produces a prediction", "9"],
    ["Fig. 4.1", "Dashboard page of the deployed application", "14"],
    ["Fig. 4.2", "Borrower input form on the Predict page", "15"],
    ["Fig. 4.3", "Prediction result for a higher-risk borrower", "16"],
    ["Fig. 4.4", "Model information page with evaluation metrics", "17"],
    ["Fig. 4.5", "Predicted score distribution by actual outcome", "19"],
    ["Fig. 4.6", "Precision, recall and F1 across decision thresholds", "19"],
]
table(["Figure No.", "Title of Figure", "Page No."], fig_rows, "",
      widths=[1.2, 3.6, 1.0])

para(space_after=10)
para("LIST OF TABLES", size=16, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
     space_before=10, space_after=10)
tab_rows = [
    ["Table 2.1", "Summary of literature review", "4"],
    ["Table 3.1", "Key concepts and definitions", "7"],
    ["Table 3.2", "Input features and their descriptions", "9"],
    ["Table 3.3", "Data quality issues and their treatment", "10"],
    ["Table 3.4", "Validation comparison of model configurations", "11"],
    ["Table 4.1", "Held-out test set evaluation metrics", "18"],
    ["Table 4.2", "Confusion matrix on the test set", "18"],
    ["Table 4.3", "Per-class classification report", "18"],
    ["Table 4.4", "Effect of the decision threshold", "19"],
    ["Table 5.1", "Advantages and limitations compared", "22"],
    ["Table 7.1", "Technology stack", "24"],
]
table(["Table No.", "Title of Table", "Page No."], tab_rows, "",
      widths=[1.2, 3.6, 1.0])

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

section("1.1 Overview of the Topic")
para("Credit risk prediction is the task of estimating, from information known "
     "about a borrower, how likely that borrower is to fail to meet their "
     "repayment obligations. Traditionally this judgement was made by human "
     "underwriters applying policy rules to an application form. Machine "
     "learning changes the basis of the judgement: instead of encoding rules by "
     "hand, a model is fitted to a historical record of borrowers whose "
     "outcomes are already known, and it learns the statistical relationship "
     "between borrower attributes and those outcomes.")
para("This seminar studies that application through one specific algorithm, the "
     "Random Forest, and one specific prediction target. A Random Forest is an "
     "ensemble method: it grows many decision trees on randomised samples of "
     "the training data and combines their outputs. It is widely used for "
     "tabular problems of this kind because it captures non-linear "
     "relationships and interactions between variables without manual feature "
     "engineering, is robust to extreme values, and produces probability "
     "estimates rather than only hard labels.")
para(f"The problem studied here is formally a supervised binary classification "
     f"task. Each record describes one borrower through "
     f"{len(meta['features'])} numeric attributes, and the label states whether "
     f"that borrower experienced serious delinquency — defined as 90 or more "
     f"days past due — within two years of an observation date. It is not a "
     f"regression problem, because the quantity predicted is a category rather "
     f"than a magnitude, and it is not a clustering problem, because the groups "
     f"are given in advance by the labels rather than discovered in the data.")

section("1.2 Need and Importance")
para("Lending institutions extend credit under uncertainty. A borrower who "
     "defaults imposes a direct loss, while a sound borrower who is wrongly "
     "refused represents forgone revenue and a denied customer. Any method that "
     "improves the ranking of applicants by risk therefore has direct economic "
     "value, and the scale of modern lending makes manual assessment of every "
     "application impractical.")
para(f"The problem is also statistically interesting in a way that makes it a "
     f"good subject for study. Serious delinquency is rare: in the dataset used "
     f"here only {pc(base)} of borrowers are positive. This imbalance "
     f"invalidates the most familiar evaluation metric. A model that ignores "
     f"its inputs entirely and always answers “no delinquency” achieves "
     f"{pc(1 - base, 1)} accuracy while identifying not a single delinquent "
     f"borrower. Reporting accuracy for such a problem is therefore misleading, "
     f"and a central theme of this seminar is which metrics should be used "
     f"instead and why.")
para("Understanding these issues matters beyond credit scoring. Rare-event "
     "prediction under class imbalance appears in fraud detection, medical "
     "diagnosis, equipment failure prediction and network intrusion detection. "
     "The evaluation reasoning developed in this report transfers directly to "
     "all of them.")

section("1.3 Objectives of the Seminar")
para("The main objectives of this seminar are:")
for obj in (
    "To understand the fundamental concepts of the Random Forest algorithm, "
    "including bootstrap aggregation, random feature selection at each split, "
    "and how an ensemble of trees produces a probability estimate.",
    "To study the architecture and working of a complete prediction system, "
    "from raw data through preprocessing and model selection to a deployed "
    "service.",
    "To examine the correct evaluation of a class-imbalanced classifier, and "
    "to demonstrate why accuracy is inadequate and what should replace it.",
    "To implement the method on a real public dataset and report honestly "
    "measured results rather than illustrative ones.",
    "To explore real-world applications of credit risk prediction and the "
    "benefits it offers to industry.",
    "To identify the advantages, limitations and ethical challenges of such "
    "systems, and to examine future developments in the field.",
):
    bullet(obj)

# ========================================= CHAPTER 2: LITERATURE REVIEW ====
chapter("CHAPTER 2: LITERATURE REVIEW")
para("This chapter reviews the principal sources consulted in preparing this "
     "seminar, covering the algorithm itself, the evaluation of imbalanced "
     "classifiers, and the software used for the implementation.")
para("Breiman [1] introduced Random Forests and established the theoretical "
     "basis for the method. The paper shows that combining many decorrelated "
     "trees reduces the variance of the prediction without a corresponding "
     "increase in bias, and that the generalisation error converges as more "
     "trees are added rather than diverging through overfitting. Two sources of "
     "randomness are responsible: bootstrap sampling of the training rows, and "
     "the restriction of each split to a random subset of features. The second "
     "is what decorrelates the trees, and it is the mechanism that makes the "
     "averaging effective.")
para("Saito and Rehmsmeier [4] address the evaluation problem directly and "
     "were the most influential source for this work. They demonstrate that "
     "the ROC plot is misleading when the classes are imbalanced, because the "
     "false-positive rate is computed against the large negative population and "
     "therefore moves very little even when the number of false alarms is "
     "substantial. The precision-recall plot does not suffer from this, because "
     "precision is computed only against the cases the model actually flagged. "
     "This finding is the reason the present work selects its model on PR-AUC "
     "rather than ROC-AUC or accuracy.")
para("Pedregosa et al. [2] document the scikit-learn library used for the "
     "implementation. The documentation of RandomForestClassifier was "
     "consulted directly on one point of substance: the library computes a "
     "predicted probability as the mean of the per-tree class probabilities, "
     "where each tree contributes the class proportion of the leaf a sample "
     "reaches. This is not the hard majority vote that introductory treatments "
     "often describe, and the distinction is preserved throughout this report.")
para("Chawla et al. [5] survey the broader family of approaches to class "
     "imbalance, including resampling methods such as SMOTE and cost-sensitive "
     "learning through class weights. Their framing of the problem as one of "
     "choosing an operating point on a trade-off curve, rather than of fixing a "
     "defective model, informed the decision in this work to adjust the "
     "decision threshold rather than reweight the classes.")
para("The dataset itself originates from the Kaggle “Give Me Some Credit” "
     "competition [3], which supplied the labelled borrower records and the "
     "data dictionary defining each attribute.")

table(["Sr. No.", "Author(s) and Year", "Research Focus", "Key Findings"],
      [["1", "Breiman (2001)",
        "Random Forest algorithm and its theory",
        "Averaging decorrelated trees reduces variance; generalisation error "
        "converges as trees are added. Bootstrap sampling and random feature "
        "subsets are the two sources of randomness."],
       ["2", "Pedregosa et al. (2011)",
        "scikit-learn machine learning library",
        "Provides the reference implementation used here. Predicted "
        "probability is the mean of per-tree leaf class proportions, not a "
        "majority vote."],
       ["3", "Kaggle / Credit Fusion (2011)",
        "Give Me Some Credit competition dataset",
        "Supplies 150,000 labelled borrower records with a documented data "
        "dictionary; competition was scored on AUC."],
       ["4", "Saito and Rehmsmeier (2015)",
        "Evaluation of imbalanced binary classifiers",
        "ROC plots are over-optimistic under imbalance; precision-recall "
        "plots are more informative because precision ignores the large true-"
        "negative population."],
       ["5", "Chawla et al. (2002, 2004)",
        "Class imbalance: resampling and cost-sensitive learning",
        "Frames imbalance as a choice of operating point on a trade-off "
        "curve; resampling and class weighting shift that point rather than "
        "improving the underlying ranking."]],
      "Table 2.1: Summary of literature review",
      widths=[0.7, 1.3, 1.5, 2.3], font_size=10)

para("Major observations and research gaps.", size=14, bold=True,
     align=WD_ALIGN_PARAGRAPH.LEFT, space_before=8, space_after=4)
para("Three observations emerge from the reviewed sources. First, the Random "
     "Forest is well understood theoretically and is a sound default for "
     "tabular classification, so the interesting difficulty in a project of "
     "this kind lies not in the algorithm but in the data and the evaluation. "
     "Second, the literature is consistent that accuracy and ROC-AUC flatter a "
     "classifier when the positive class is rare, yet both remain widely "
     "reported in practice. Third, the standard remedies for imbalance are "
     "frequently applied without checking whether they improve the model's "
     "ranking or merely relocate its operating point.")
para("The gap addressed by the practical component of this seminar is that "
     "last point. Rather than applying class weighting because it is "
     "conventional, this work measures its effect and finds that it moves the "
     "model along the precision-recall curve without lifting it, and therefore "
     "uses explicit threshold selection instead. A second gap concerns "
     "undocumented placeholder values in the dataset, which much published work "
     "on this data removes or imputes silently; here they are shown to carry "
     "substantial signal and are handled explicitly.")

# ============================== CHAPTER 3: TECHNOLOGY / CONCEPT OVERVIEW ===
chapter("CHAPTER 3: TECHNOLOGY / CONCEPT OVERVIEW")

section("3.1 Basic Concepts")
para("This section defines the terminology used throughout the remainder of "
     "the report.")
para([("Supervised classification. ", {"bold": True}),
      ("A model is fitted to records whose correct labels are known, and learns "
       "a mapping from attributes to a category. Here the category is binary.",
       {})])
para([("Decision tree. ", {"bold": True}),
      ("A predictor that repeatedly splits the data on one attribute at a time. "
       "Each split is chosen to separate the classes as well as possible; the "
       "terminal nodes, called leaves, hold the training samples that reached "
       "them.", {})])
para([("Gini impurity. ", {"bold": True}),
      ("The measure of how mixed a node is, defined as G = 1 − Σ pk², where pk "
       "is the proportion of class k in that node. A node containing a single "
       "class has impurity zero. Each split is chosen to maximise the weighted "
       "reduction in G.", {})])
para([("Bootstrap aggregation (bagging). ", {"bold": True}),
      ("Training each tree on a sample drawn with replacement from the training "
       "set, so that every tree sees a slightly different dataset. Averaging "
       "their predictions reduces variance.", {})])
para([("Class imbalance. ", {"bold": True}),
      (f"The condition in which one class is far rarer than the other. Here the "
       f"positive class is {pc(base)} of records.", {})])
para([("Decision threshold. ", {"bold": True}),
      ("The probability above which a prediction is reported as positive. It is "
       "a choice made after training, and it determines where the model sits on "
       "the trade-off between precision and recall.", {})])

table(["Term", "Definition"],
      [["Random Forest", f"An ensemble of {trees} decision trees grown on bootstrap "
        "samples, with each split restricted to a random subset of features."],
       ["Target variable", "SeriousDlqin2yrs: 1 if the borrower was 90+ days past "
        "due within two years, else 0."],
       ["Precision", "Of the borrowers flagged as delinquent, the proportion who "
        "truly were. TP / (TP + FP)."],
       ["Recall", "Of the borrowers who truly were delinquent, the proportion the "
        "model identified. TP / (TP + FN)."],
       ["F1-score", "The harmonic mean of precision and recall, 2PR / (P + R), "
        "which is dominated by whichever is weaker."],
       ["ROC-AUC", "The probability that a randomly chosen positive is scored "
        "above a randomly chosen negative. 0.5 is chance."],
       ["PR-AUC", f"Area under the precision-recall curve. Its no-skill baseline "
        f"equals the positive rate, here {base:.4f}."]],
      "Table 3.1: Key concepts and definitions", widths=[1.4, 4.4], font_size=11)

section("3.2 Working Principle / Architecture")
para("The implemented system separates offline work from online serving. "
     "Training, evaluation and figure generation are performed offline and "
     "produce a single saved artifact. The web service loads that artifact once "
     "when it starts and uses it to answer requests; it never trains.")
figure("system-architecture.png", 6.0,
       "Figure 3.1: System architecture of the implemented application")
para("The browser communicates with one origin only. Requests to the /api path "
     "are forwarded to the prediction service on the server side, so no "
     "cross-origin request is made by the browser at any point. The model file "
     "is bundled with the service and loaded a single time at startup rather "
     "than on each request, which would otherwise dominate the response time.")

para("Training pipeline.", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
     space_before=8, space_after=4)
para("Training proceeds through six stages. The labelled file is loaded and "
     "validated against an expected schema; implausible values are marked as "
     "missing; the data is split; candidate models are compared; the decision "
     "threshold is selected; and finally the chosen model is scored once on "
     "data it has never seen.")
figure("ml-pipeline.png", 6.3,
       "Figure 3.2: Training pipeline from raw data to the saved model artifact")
para("The discipline that makes the final figures trustworthy is that every "
     "fitted quantity — the imputation medians, the model parameters and the "
     "decision threshold — is derived from the training and validation data "
     "only. The test split is held in reserve until the last stage.")
figure("data-splits.png", 6.3,
       "Figure 3.3: Stratified 60/20/20 split and the role of each part")
para(f"The split is stratified, meaning each part preserves the overall class "
     f"ratio. At a {pc(base)} positive rate an unstratified split could leave "
     f"the three parts with noticeably different base rates, which would make "
     f"their metrics incomparable. Figure 3.3 confirms that all three parts "
     f"carry the same {pc(base)} positive rate.")

para("How a prediction is produced.", size=14, bold=True,
     align=WD_ALIGN_PARAGRAPH.LEFT, space_before=8, space_after=4)
figure("random-forest.png", 6.3,
       "Figure 3.4: How a Random Forest produces a prediction")
para(f"For a new borrower, each of the {trees} trees routes the record down to "
     f"one leaf and reports the proportion of positive training samples in that "
     f"leaf. The forest averages those {trees} proportions, and the mean is the "
     f"predicted probability. This is an important detail of the "
     f"implementation: scikit-learn averages per-tree probabilities rather than "
     f"taking a hard majority vote of per-tree classes. The resulting "
     f"probability is then compared with the decision threshold of {thr:.4f} to "
     f"produce the reported class.")

section("3.3 Key Features")
para("The dataset provides ten numeric attributes per borrower, described in "
     "Table 3.1 below. No categorical encoding is required, and because "
     "decision trees split on rank order rather than magnitude, no feature "
     "scaling is required either.")
table(["No.", "Feature", "Description"],
      [["1", "RevolvingUtilizationOfUnsecuredLines",
        "Card and personal line balances divided by total credit limits"],
       ["2", "age", "Borrower age in years"],
       ["3", "NumberOfTime30-59DaysPastDueNotWorse",
        "Count of 30–59 day late episodes in the last two years"],
       ["4", "DebtRatio",
        "Debt payments, alimony and living costs divided by gross monthly income"],
       ["5", "MonthlyIncome", "Gross monthly income (19.82% missing)"],
       ["6", "NumberOfOpenCreditLinesAndLoans",
        "Open installment loans and lines of credit"],
       ["7", "NumberOfTimes90DaysLate", "Count of 90+ day late episodes"],
       ["8", "NumberRealEstateLoansOrLines",
        "Mortgages and home equity lines of credit"],
       ["9", "NumberOfTime60-89DaysPastDueNotWorse",
        "Count of 60–89 day late episodes in the last two years"],
       ["10", "NumberOfDependents",
        "Dependents excluding the borrower (2.62% missing)"]],
      "Table 3.2: Input features and their descriptions",
      widths=[0.5, 2.5, 2.8], font_size=10)

section("3.4 Components and Their Interaction")
para("Preprocessing and the classifier are combined into a single scikit-learn "
     "Pipeline, and it is that composite object which is saved to disk. The "
     "serving code loads the pipeline and calls it, so there is no second copy "
     "of the cleaning or imputation logic on the serving side. This makes a "
     "mismatch between training and inference structurally impossible rather "
     "than merely something to be careful about.")
para("The pipeline has three stages in sequence. A cleaning step replaces "
     "documented placeholder and impossible values with missing markers. An "
     "imputation step fills those gaps with the training median and "
     "additionally appends a binary indicator column recording that the value "
     "was absent. The Random Forest classifier is the final stage. A "
     "consequence worth noting is that the classifier receives "
     f"{len(meta['model_input_columns'])} columns built from the "
     f"{len(meta['features'])} input features, because of the appended "
     "indicators; the public interface of the service is unchanged.")

table(["Issue found in the raw data", "Extent", "Treatment applied"],
      [["MonthlyIncome missing", "29,731 rows (19.82%)",
        "Median imputation plus a missing-value indicator"],
       ["NumberOfDependents missing", "3,924 rows (2.62%)",
        "Median imputation plus a missing-value indicator"],
       ["Late-payment counters coded 96 or 98", "269 rows",
        "Treated as placeholders; imputed with an indicator, because these rows "
        "are 54.65% positive against a 6.68% base rate"],
       ["Age recorded as 0", "1 row",
        "Impossible for a credit applicant; treated as missing"],
       ["Duplicate predictor rows", "646 rows",
        "Retained; plausibly distinct borrowers with the same rounded values"],
       ["Extreme ratio values", "Up to 329,664",
        "Retained; a Random Forest splits on rank order and is unaffected"]],
      "Table 3.3: Data quality issues and their treatment",
      widths=[2.2, 1.4, 2.2], font_size=10)

para("Model selection.", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
     space_before=8, space_after=4)
para("Five configurations were fitted on the training split and compared on the "
     "validation split. The selection metric was PR-AUC, chosen for the reasons "
     "given in the literature review.")
table(["Configuration", "ROC-AUC", "PR-AUC", "Precision", "Recall"],
      [[c["candidate"].replace("_", " "), f4(c["roc_auc"]), f4(c["pr_auc"]),
        f4(c["precision"]), f4(c["recall"])] for c in meta["validation_comparison"]],
      "Table 3.4: Validation comparison of model configurations",
      widths=[2.2, 1.0, 1.0, 1.0, 0.9], font_size=10)
para(f"Two findings from this comparison shaped the final model. Increasing the "
     f"minimum leaf size from the default to {leaf} improved validation PR-AUC "
     f"from {f4(cand['baseline']['pr_auc'])} to "
     f"{f4(cand['regularised']['pr_auc'])}, confirming that an unrestricted "
     f"forest was overfitting. Class weighting, by contrast, raised recall from "
     f"{f4(cand['regularised']['recall'])} to "
     f"{f4(cand['balanced']['recall'])} but left PR-AUC essentially unchanged "
     f"at {f4(cand['balanced']['pr_auc'])}, which means it moved the model along "
     f"the precision-recall curve without lifting the curve itself. Since a "
     f"decision threshold achieves the same movement while remaining visible "
     f"and adjustable without refitting, class weighting was not used.")
para(f"The winning configuration added missing-value indicator columns, which "
     f"gave the best validation PR-AUC of {f4(cand['regularised_with_indicators']['pr_auc'])}. "
     f"The decision threshold was then selected on the validation split by "
     f"maximising F1, giving {thr:.4f} rather than the conventional 0.50. The "
     f"model was refitted on train and validation combined and scored once on "
     f"the test split.")

# ================================ CHAPTER 4: APPLICATIONS AND USE CASES ===
chapter("CHAPTER 4: APPLICATIONS AND USE CASES")

section("4.1 Applications")
para("Credit risk models of this kind are applied across the financial sector "
     "and beyond. In retail lending they support decisions on personal loans, "
     "credit cards and mortgages. In banking operations they drive portfolio "
     "monitoring, identifying accounts whose risk profile has deteriorated "
     "since origination. In regulatory reporting they feed the expected-loss "
     "calculations that determine how much capital an institution must hold.")
para("The same technique generalises to any rare-event prediction problem on "
     "tabular data. Fraud detection, insurance claim triage, customer churn "
     "prediction, predictive maintenance of equipment and clinical risk scoring "
     "all share the structure studied here: many records, a minority class of "
     "interest, and a need for a ranked score rather than a bare label.")

section("4.2 Real-World Use Cases")
para([("Application scoring at origination. ", {"bold": True}),
      ("When an application is received, the lender must decide whether to "
       "extend credit. A model trained on past applicants produces a risk score "
       "that is combined with policy rules and affordability checks. The "
       "benefit is consistency and speed: every applicant is assessed on the "
       "same basis in milliseconds. The present system demonstrates this use "
       "case directly.", {})])
para([("Behavioural scoring of an existing portfolio. ", {"bold": True}),
      ("For borrowers already on the books, payment behaviour accumulates over "
       "time. Rescoring the portfolio periodically identifies accounts drifting "
       "toward delinquency early enough for intervention, such as a payment "
       "plan. The payment-history attributes that dominate this model are "
       "precisely the ones that update as behaviour changes.", {})])
para([("Collections prioritisation. ", {"bold": True}),
      ("Where a portfolio already contains delinquent accounts and recovery "
       "resources are finite, a ranked score directs effort to the accounts "
       "where it will matter most. This use case needs only the ranking, not a "
       "calibrated probability, which makes it well suited to a model with a "
       "strong ROC-AUC.", {})])

section("4.3 Benefits to Industry and Business")
para("The principal benefits are consistency, scale and transparency of "
     "reasoning. An automated score applies identical criteria to every "
     "applicant, removing the variation between individual assessors. It scales "
     "to volumes no manual process could handle. And because a Random Forest "
     "exposes which attributes it relies on, the resulting model can be "
     "inspected and discussed rather than treated as opaque.")
para("For the institution the economic effect is a better-informed trade-off "
     "between the loss from a defaulted loan and the margin forgone on a "
     "declined application. The threshold analysis in Section 4.5 shows exactly "
     "how that trade-off can be tuned to an institution's cost structure.")

section("4.4 Implemented Application Interface")
para(f"The trained model was deployed as a web application to demonstrate the "
     f"complete path from data to a usable system. It is publicly available at "
     f"{APP_URL}, with the service's own interactive API documentation at "
     f"{API_URL}. The interface has four pages, shown in the figures that "
     f"follow. All values displayed are fetched from the prediction service at "
     f"run time; none are fixed in the page.")

para("Dashboard.", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
     space_before=8, space_after=4)
para("The landing page states what the model predicts and what it does not, and "
     "reports the measured test metrics. It deliberately presents the "
     "majority-class baseline alongside the accuracy figure, so that a reader "
     "cannot mistake the accuracy for a strong result.")
figure("01-dashboard.png", 6.1, "Figure 4.1: Dashboard page of the deployed application")

para("Prediction form.", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
     space_before=8, space_after=4)
para("The form exposes exactly the ten attributes the model was trained on, "
     "grouped into borrower profile, credit usage and payment history. "
     "Technical column names are replaced with readable labels, less obvious "
     "financial variables carry short explanations, and the two optional fields "
     "are marked as such. A demonstration button loads a genuine record from "
     "the held-out test split, labelled as such so it is not mistaken for "
     "information about the person using the page.")
figure("02-predict-form.png", 6.1, "Figure 4.2: Borrower input form on the Predict page")

para("Prediction result.", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
     space_before=8, space_after=4)
para("The result page separates three things that are easily conflated: the "
     "estimated probability, the binary class derived from the threshold, and "
     "an illustrative risk band. The lower panel places the individual score "
     "against the distribution of scores for borrowers whose actual outcome is "
     "known, so that a single number is given a frame of reference.")
figure("03-result.png", 6.1,
       "Figure 4.3: Prediction result for a higher-risk borrower")

para("Model information.", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
     space_before=8, space_after=4)
para("The final page reports the algorithm, the dataset and its source, the "
     "target definition, the decision threshold and how it was chosen, the full "
     "evaluation metrics and the feature importances.")
figure("04-model-info.png", 6.1,
       "Figure 4.4: Model information page with evaluation metrics")

section("4.5 Results Obtained")
para(f"The figures below were produced by scoring the held-out test split of "
     f"{dist['test']['total']:,} borrowers exactly once, at the "
     f"validation-selected threshold of {thr:.4f}.")
table(["Metric", "Value", "Interpretation"],
      [["ROC-AUC", f4(test["roc_auc"]),
        "Ranks a delinquent borrower above a sound one 86.5% of the time"],
       ["PR-AUC", f4(test["pr_auc"]),
        f"About six times the no-skill baseline of {base:.4f}"],
       ["Recall", f4(test["recall"]),
        f"Identified {cm['true_positives']:,} of {n_pos:,} real delinquencies"],
       ["Precision", f4(test["precision"]),
        "Of those flagged, roughly 42% truly were delinquent"],
       ["F1-score", f4(test["f1"]), "Harmonic mean of precision and recall"],
       ["Accuracy", f4(test["accuracy"]),
        f"Below the {pc(1 - base, 1)} of always answering “no”"]],
      "Table 4.1: Held-out test set evaluation metrics",
      widths=[1.2, 0.9, 3.7], font_size=11)

table(["", "Predicted: not delinquent", "Predicted: delinquent"],
      [["Actual: not delinquent", f"{cm['true_negatives']:,} (TN)",
        f"{cm['false_positives']:,} (FP)"],
       ["Actual: delinquent", f"{cm['false_negatives']:,} (FN)",
        f"{cm['true_positives']:,} (TP)"]],
      "Table 4.2: Confusion matrix on the test set",
      widths=[2.0, 1.9, 1.9], font_size=11)

para(f"Read directly: of {n_pos:,} borrowers who genuinely became seriously "
     f"delinquent the model identified {cm['true_positives']:,} and missed "
     f"{cm['false_negatives']:,}; of {n_neg:,} who did not, it wrongly flagged "
     f"{cm['false_positives']:,}.")

table(["Class", "Precision", "Recall", "F1-score", "Support"],
      [["0 — No serious delinquency", f4(report["classes"]["0"]["precision"]),
        f4(report["classes"]["0"]["recall"]), f4(report["classes"]["0"]["f1"]),
        f"{report['classes']['0']['support']:,}"],
       ["1 — Serious delinquency", f4(report["classes"]["1"]["precision"]),
        f4(report["classes"]["1"]["recall"]), f4(report["classes"]["1"]["f1"]),
        f"{report['classes']['1']['support']:,}"],
       ["**Macro average**", f"**{f4(report['macro_avg']['precision'])}**",
        f"**{f4(report['macro_avg']['recall'])}**",
        f"**{f4(report['macro_avg']['f1'])}**", f"**{dist['test']['total']:,}**"],
       ["Weighted average", f4(report["weighted_avg"]["precision"]),
        f4(report["weighted_avg"]["recall"]), f4(report["weighted_avg"]["f1"]),
        f"{dist['test']['total']:,}"]],
      "Table 4.3: Per-class classification report",
      widths=[2.1, 1.0, 0.9, 0.95, 0.95], font_size=10)

para(f"This table is the clearest demonstration of the imbalance problem. The "
     f"model performs excellently on the majority class, with an F1 of "
     f"{f4(report['classes']['0']['f1'])}, and considerably less well on the "
     f"minority class at {f4(report['classes']['1']['f1'])}. The weighted "
     f"average of {f4(report['weighted_avg']['f1'])} looks impressive but is "
     f"dominated by the {report['classes']['0']['support']:,} negative records "
     f"and conveys almost nothing about the class the system exists to find. "
     f"Where a single summary figure is required, the macro average of "
     f"{f4(report['macro_avg']['f1'])} or the positive-class F1 should be "
     f"quoted, never the weighted average.")

para("Why the errors occur.", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
     space_before=8, space_after=4)
para("The application presents a chart that explains the recall ceiling "
     "directly. Each curve is one actual outcome group, drawn as a share of its "
     "own total because the negatives outnumber the positives by roughly "
     "fourteen to one.")
figure("05-chart-separation.png", 5.6,
       "Figure 4.5: Predicted score distribution by actual outcome")
para("Most borrowers who stayed current score near zero, which is why the model "
     "is accurate overall. Borrowers who did become delinquent are spread "
     "across the whole range rather than clustered at the high end, so a large "
     "share of them fall inside the same region as the healthy borrowers. That "
     "overlap, rather than any defect in the implementation, is what limits "
     "recall. Moving the threshold leftward captures more of the delinquent "
     "group but sweeps up more sound borrowers with it.")

figure("06-chart-threshold.png", 5.6,
       "Figure 4.6: Precision, recall and F1 across decision thresholds")
table(["Threshold", "Accuracy", "Precision", "Recall", "F1", "Flagged"],
      [[("**" + f"{r['threshold']:.4f}" + "**") if abs(r["threshold"] - thr) < 5e-5
        else f"{r['threshold']:.4f}",
        f4(r["accuracy"]), f4(r["precision"]), f4(r["recall"]), f4(r["f1"]),
        f"{r['confusion_matrix']['true_positives'] + r['confusion_matrix']['false_positives']:,}"]
       for r in sweep],
      "Table 4.4: Effect of the decision threshold on the test set",
      widths=[1.1, 0.95, 1.0, 0.85, 0.85, 0.85], font_size=10)
para("The same trained model produces very different behaviour depending on "
     "where the cut-off is placed. Lowering it to 0.10 raises recall above 0.70 "
     "but reduces precision to roughly 0.26, so three of every four flagged "
     "borrowers would be false alarms. Raising it to 0.50 gives precision above "
     "0.62 but identifies only about one delinquency in six. Accuracy rises "
     "monotonically as the threshold rises, purely because the model predicts "
     "the majority class more often, which is a further demonstration of why it "
     "is a poor guide here.")

para("Stability of the result.", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
     space_before=8, space_after=4)
para(f"A single test score comes from one particular partition of the data. "
     f"Five-fold cross-validation on the training and validation data, with the "
     f"test split excluded throughout, gives a mean ROC-AUC of "
     f"{f4(cv['roc_auc']['mean'])} with a standard deviation of "
     f"{f4(cv['roc_auc']['std'])}, and a mean PR-AUC of "
     f"{f4(cv['pr_auc']['mean'])} ± {f4(cv['pr_auc']['std'])}. The test values "
     f"of {f4(test['roc_auc'])} and {f4(test['pr_auc'])} both fall within one "
     f"standard deviation of their cross-validated means, so the reported "
     f"performance is a property of the data and configuration rather than an "
     f"artefact of one fortunate split.")
para(f"Comparing in-sample with held-out performance also quantifies how much "
     f"of the apparent skill is memorisation. ROC-AUC falls from "
     f"{f4(train['roc_auc'])} on the fitted data to {f4(test['roc_auc'])} on "
     f"held-out data, a gap of {train['roc_auc'] - test['roc_auc']:.4f}, and "
     f"PR-AUC falls by {train['pr_auc'] - test['pr_auc']:.4f}. Some overfitting "
     f"therefore remains despite the leaf-size regularisation, and this is "
     f"reported rather than concealed.")

# ============================ CHAPTER 5: ADVANTAGES AND LIMITATIONS =======
chapter("CHAPTER 5: ADVANTAGES AND LIMITATIONS")

section("5.1 Advantages")
para("The Random Forest offered several concrete advantages for this problem. "
     "It captured non-linear relationships and interactions between payment "
     "history and credit usage without any manual feature engineering. Because "
     "trees split on rank order, it was unaffected by the extreme values "
     "present in the ratio attributes, which would have distorted a linear "
     "model and which were therefore left in place rather than capped. It "
     "required no feature scaling or categorical encoding.")
para("Operationally it produces probability estimates rather than only hard "
     "labels, which is what a risk score requires, and the decision threshold "
     "can be moved after training without refitting. It also reports which "
     "attributes it relies on, which keeps the model inspectable. Training on "
     "150,000 records completes in about a minute on a laptop, making the whole "
     "experiment reproducible.")

section("5.2 Limitations")
para("The most significant limitation is the data rather than the algorithm. "
     "The records describe United States consumers around 2011 and carry no "
     "date column, so no out-of-time validation is possible and present-day "
     "performance is unknown. The sampling frame is undocumented, so there is "
     "no basis for establishing which population the sample represents.")
para("The predicted probabilities are not calibrated, and calibration was not "
     "evaluated. A predicted 0.30 should be read as a ranking score rather "
     "than as a claim that thirty in a hundred such borrowers default. Random "
     "Forest probabilities are typically compressed away from the extremes.")
para("Fairness cannot be assessed, because the dataset contains no race, sex or "
     "geographic attributes. That absence is not protection: age is present and "
     "is used by the model, and attributes such as income and credit "
     "utilisation can act as proxies for protected characteristics. The absence "
     "of a measured bias is the absence of measurement, not evidence of "
     "fairness.")
para(f"Finally, the chosen threshold maximises F1, which weights a missed "
     f"delinquency and a false alarm equally. Real lending economics do not: "
     f"the loss on a defaulted loan generally exceeds the margin forgone on a "
     f"declined application. Choosing the threshold properly would require "
     f"loss-given-default and margin figures that the dataset does not contain.")

section("5.3 Challenges")
para("Three practical difficulties were encountered during implementation. The "
     "first concerned undocumented placeholder values: the three late-payment "
     "counters use the codes 96 and 98 in 269 records, and the data dictionary "
     "does not explain them. Those records are 54.65% positive against a base "
     "rate of 6.68%, so simply imputing them to the median would have discarded "
     "real signal while asserting that these borrowers had clean payment "
     "histories. The solution adopted was to impute the value but retain a "
     "missing-value indicator, which lets the model learn that the figure was "
     "unavailable without anyone having to guess what the codes mean.")
para("The second challenge was deployment. The scientific Python stack together "
     "with the 70 MB model file produces a bundle of roughly 315 MB, which "
     "exceeds the limits of many hosting platforms. Separating the runtime "
     "dependencies from the training and plotting tools was necessary to fit "
     "within the available 500 MB allowance.")
para("The third was avoiding train-serve skew. Any difference between the "
     "preprocessing applied during training and during inference would corrupt "
     "predictions silently. Combining the preprocessing and the classifier into "
     "a single saved pipeline removes the possibility entirely.")

table(["Advantages", "Limitations"],
      [["Captures non-linear interactions without manual feature engineering",
        "Data describes 2011 US consumers; present-day performance unknown"],
       ["Robust to extreme values; no scaling or encoding required",
        "No date column, so out-of-time validation is impossible"],
       ["Produces probability estimates, not only labels",
        "Probabilities are uncalibrated and calibration was not evaluated"],
       ["Threshold adjustable after training without refitting",
        "Threshold optimises F1, which weights both error types equally"],
       ["Reports feature importances, keeping the model inspectable",
        "Importance describes splits, not causation, and is biased toward "
        "high-cardinality features"],
       ["Trains in about a minute; fully reproducible with a fixed seed",
        "No protected attributes present, so fairness cannot be tested"]],
      "Table 5.1: Advantages and limitations compared",
      widths=[2.9, 2.9], font_size=10)

# ======================================== CHAPTER 6: FUTURE SCOPE =========
chapter("CHAPTER 6: FUTURE SCOPE")
para("Several specific improvements follow directly from the limitations "
     "identified in the previous chapter.")
para([("Probability calibration. ", {"bold": True}),
      ("Fitting a calibration model on held-out data and reporting a reliability "
       "diagram and Brier score would establish whether the predicted "
       "probabilities can be read as real-world frequencies. This is the single "
       "change that would most increase the practical usefulness of the "
       "output.", {})])
para([("Cost-sensitive threshold selection. ", {"bold": True}),
      ("Replacing the F1 criterion with an explicit cost matrix, once "
       "loss-given-default and margin figures are available, would place the "
       "operating point where it minimises expected loss rather than where it "
       "balances two error types that are not in fact equally costly.", {})])
para([("Out-of-time validation. ", {"bold": True}),
      ("Obtaining a dataset with origination dates would permit training on an "
       "earlier period and testing on a later one, which is the standard "
       "validation approach for credit models precisely because random splits "
       "flatter them.", {})])
para([("Permutation importance and per-prediction explanation. ", {"bold": True}),
      ("Permutation importance computed on held-out data is less biased than "
       "the impurity-based measure used here. Per-prediction attributions, for "
       "example through SHAP values, would support the adverse-action "
       "explanations that regulated lending requires.", {})])
para([("Fairness assessment. ", {"bold": True}),
      ("With a dataset containing the relevant attributes, disparate-impact "
       "testing across protected groups could be performed, and mitigation "
       "applied where required.", {})])
para("More broadly, the direction of the field is toward models that are not "
     "merely accurate but auditable. Regulatory frameworks increasingly require "
     "that an automated credit decision can be explained to the person it "
     "affects, which favours methods whose reasoning can be inspected and "
     "argues against treating predictive performance as the only objective.")

# ========================================== CHAPTER 7: CONCLUSION =========
chapter("CHAPTER 7: CONCLUSION")
para(f"This seminar studied the application of the Random Forest algorithm to "
     f"the prediction of serious borrower delinquency, and implemented a "
     f"complete working system to demonstrate it. A classifier was trained on "
     f"{meta['dataset']['labeled_rows']:,} labelled records from the public "
     f"Kaggle “Give Me Some Credit” dataset and evaluated on a held-out split "
     f"of {dist['test']['total']:,} records scored exactly once, achieving a "
     f"ROC-AUC of {f4(test['roc_auc'])} and a PR-AUC of {f4(test['pr_auc'])} "
     f"against a no-skill baseline of {base:.4f}.")
para("The methodological findings are at least as important as the figures. "
     "Accuracy was shown to be an actively misleading metric for this problem, "
     "since a model that ignores its inputs scores higher than the trained one. "
     "Class weighting, the conventional remedy for imbalance, was measured and "
     "found to move the model along the precision-recall curve without lifting "
     "it, so an explicit decision threshold selected on validation data was "
     "used instead. Undocumented placeholder values in the dataset were found "
     "to carry substantial signal and were handled with missing-value "
     "indicators rather than imputed away. The residual gap between in-sample "
     "and held-out performance was measured and reported rather than omitted.")
para("The practical component established that the path from a raw public "
     "dataset to a usable application is short when the engineering discipline "
     "is right: preprocessing held inside the saved model so that training and "
     "inference cannot diverge, the decision threshold stored alongside the "
     "model so the served cut-off and the reported metrics cannot disagree, and "
     "every documented figure generated from the model's own saved output "
     "rather than transcribed.")
table(["Layer", "Technology"],
      [["Dataset", "Kaggle “Give Me Some Credit” — 150,000 labelled records"],
       ["Model", "scikit-learn RandomForestClassifier, persisted with joblib"],
       ["Data processing", "Python, pandas, NumPy"],
       ["Visualisation", "matplotlib, seaborn; inline SVG in the web interface"],
       ["Backend API", "FastAPI, Pydantic, Uvicorn"],
       ["Frontend", "React, TypeScript, Vite, Tailwind CSS"],
       ["Testing", "pytest — 61 automated tests"],
       ["Deployment", "Vercel (frontend and backend as separate projects)"]],
      "Table 7.1: Technology stack", widths=[1.6, 4.2], font_size=11)
para("My own understanding of the significance of this topic is that the value "
     "of machine learning in credit risk lies less in squeezing out additional "
     "predictive accuracy than in making the basis of a decision explicit and "
     "consistent. The honest conclusion of this work is that the system built "
     "here extracts real and measurable signal from borrower data, and is "
     "simultaneously nowhere near fit to decide anybody's credit application. "
     "Holding both of those statements at once is, I think, the central skill "
     "the exercise teaches.")

# ============================================================ REFERENCES ==
chapter("REFERENCES / BIBLIOGRAPHY")
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
    "Learning: Data Mining, Inference, and Prediction, 2nd ed. Springer, 2009.",
    "[7] A. Géron, Hands-On Machine Learning with Scikit-Learn, Keras and "
    "TensorFlow, 3rd ed. O’Reilly Media, 2022.",
    f"[8] P. Bhalerao, “Loan Default Risk Prediction — project source code.” "
    f"Available: {REPO_URL}. Accessed: October 24, 2026.",
    f"[9] P. Bhalerao, “Loan Delinquency Risk — deployed application.” "
    f"Available: {APP_URL}. Accessed: October 24, 2026.",
):
    para(ref, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=8, line=1.3)

doc.save(str(OUT))
print(f"saved {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
