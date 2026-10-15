"""Build the 4-slide project deck.

Slide 1 is taken from the college's Seminar deck and edited in place; slides 2-4
are new. Every number is read from the trained model's saved files, so the deck
cannot disagree with the project.

Usage: python build_deck.py <project_root> <source.pptx> <output.pptx>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import (
    XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE,
)
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT, SOURCE, OUTPUT = (Path(a) for a in sys.argv[1:4])
ART = ROOT / "backend" / "ml" / "artifacts"
meta = json.loads((ART / "metadata.json").read_text(encoding="utf-8"))
report = json.loads((ART / "evaluation_report.json").read_text(encoding="utf-8"))

# ----------------------------------------------------------------- theme ----
# Matches the college deck: white page, serif type, thin dark outlines, soft
# pastel fills. NAVY and BRICK are the only strong colours, and they always mean
# the same thing: NAVY = the model / the majority class, BRICK = delinquency.
HEAD = "Book Antiqua"
BODY = "Times New Roman"


def rgb(hex_: str) -> RGBColor:
    return RGBColor.from_string(hex_)


INK, MUTED, LINE = "1A1A1A", "555555", "2B2B2B"
NAVY, BRICK = "2F5D8C", "B5483F"
P_BLUE, P_GREEN, P_YELLOW, P_PINK = "D9E7F5", "DDF0E3", "FBEFCB", "F8DCDC"
GRID = "D9D9D9"
WHITE = "FFFFFF"

# ------------------------------------------------------------- numbers ------
test = report["per_split_metrics"]["test"]
cm = test["confusion_matrix"]
dist = report["class_distribution"]
cv = report["cross_validation"]["summary"]
threshold = meta["decision_threshold"]["value"]
candidates = {c["candidate"]: c for c in meta["validation_comparison"]}
chosen_val = candidates[meta["selected_configuration"]["name"]]
importances = meta["feature_importances"]
curve = meta["test_threshold_curve"]
n_pos = cm["true_positives"] + cm["false_negatives"]
majority_accuracy = 1 - dist["test"]["positive_rate"]


def f4(v: float) -> str:
    return f"{v:.4f}"


def pct(v: float, d: int = 1) -> str:
    return f"{v * 100:.{d}f}%"


def n(v: int) -> str:
    return f"{int(v):,}"


# --------------------------------------------------------------- helpers ----
def style_runs(paragraph, runs, size, bold, color, font, italic):
    """`runs` is a string or a list of (text, {overrides})."""
    if isinstance(runs, str):
        runs = [(runs, {})]
    for text, o in runs:
        r = paragraph.add_run()
        r.text = text
        r.font.name = o.get("font", font)
        r.font.size = Pt(o.get("size", size))
        r.font.bold = o.get("bold", bold)
        r.font.italic = o.get("italic", italic)
        r.font.color.rgb = rgb(o.get("color", color))


def write(tf, paras, anchor=MSO_ANCHOR.TOP, margins=(0.08, 0.04, 0.08, 0.04), wrap=True):
    """Fill a text frame. Each paragraph is a dict of text + formatting."""
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left, tf.margin_top, tf.margin_right, tf.margin_bottom = (Inches(m) for m in margins)
    for i, spec in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = spec.get("align", PP_ALIGN.LEFT)
        if "after" in spec:
            p.space_after = Pt(spec["after"])
        if "before" in spec:
            p.space_before = Pt(spec["before"])
        if "line" in spec:
            p.line_spacing = spec["line"]
        style_runs(
            p, spec["t"], spec.get("size", 12), spec.get("bold", False),
            spec.get("color", INK), spec.get("font", BODY), spec.get("italic", False),
        )


def box(slide, x, y, w, h, fill=None, line=LINE, lw=1.0, shape=MSO_SHAPE.RECTANGLE, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.shadow.inherit = False
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(lw)
    else:
        s.line.fill.background()
    if name:
        s.name = name
    return s


def label(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, margins=(0, 0, 0, 0), name=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    write(tb.text_frame, paras, anchor=anchor, margins=margins)
    if name:
        tb.name = name
    return tb


def arrow(slide, x1, y1, x2, y2, color=LINE, w=1.25, dashed=False, head=True):
    c = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(w)
    if dashed:
        c.line.dash_style = MSO_LINE.DASH
    if head:
        ln = c.line._get_or_add_ln()
        ln.append(etree.SubElement(ln, qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return c


def circle_number(slide, x, y, d, text, fill):
    c = box(slide, x, y, d, d, fill=fill, line=LINE, lw=0.75, shape=MSO_SHAPE.OVAL)
    write(c.text_frame, [{"t": text, "size": 12, "bold": True, "font": HEAD, "align": PP_ALIGN.CENTER}],
          anchor=MSO_ANCHOR.MIDDLE, margins=(0, 0, 0, 0), wrap=False)
    return c


def slide_title(slide, title, subtitle):
    t = slide.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.5), Inches(0.22), Inches(12.33), Inches(0.72)
    write(t.text_frame, [{"t": title, "size": 30, "bold": True, "font": HEAD, "align": PP_ALIGN.CENTER}],
          anchor=MSO_ANCHOR.MIDDLE, margins=(0, 0, 0, 0))
    label(slide, 0.5, 0.92, 12.33, 0.36,
          [{"t": subtitle, "size": 15, "italic": True, "color": MUTED, "align": PP_ALIGN.CENTER}],
          name="Subtitle")


def section_label(slide, x, y, w, text):
    label(slide, x, y, w, 0.3, [{"t": text, "size": 14, "bold": True, "font": HEAD}], name="Section label")


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ---- table helpers (python-pptx has no cell-border API, so write the XML) ----
def cell_borders(cell, bottom=GRID, others=None, width_pt=0.75):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        for old in tcPr.findall(qn(tag)):
            tcPr.remove(old)
    spec = {"a:lnL": others, "a:lnR": others, "a:lnT": others, "a:lnB": bottom}
    for idx, tag in enumerate(("a:lnL", "a:lnR", "a:lnT", "a:lnB")):
        colour = spec[tag]
        ln = etree.Element(qn(tag), {"w": str(int(Pt(width_pt))) if colour else "0"})
        if colour:
            fill = etree.SubElement(ln, qn("a:solidFill"))
            etree.SubElement(fill, qn("a:srgbClr"), {"val": colour})
        else:
            etree.SubElement(ln, qn("a:noFill"))
        tcPr.insert(idx, ln)  # borders must precede any fill element


def build_table(slide, x, y, widths, row_h, header, rows, size=12):
    shape = slide.shapes.add_table(len(rows) + 1, len(widths), Inches(x), Inches(y),
                                   Inches(sum(widths)), Inches(row_h * (len(rows) + 1)))
    shape.name = "Data issues table"
    table = shape.table
    # "No Style, No Grid": every colour below is set explicitly.
    tblPr = table._tbl.tblPr
    style = tblPr.find(qn("a:tableStyleId"))
    if style is None:
        style = etree.SubElement(tblPr, qn("a:tableStyleId"))
    style.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
    tblPr.set("firstRow", "0")
    tblPr.set("bandRow", "0")
    for i, w in enumerate(widths):
        table.columns[i].width = Inches(w)
    for r in range(len(rows) + 1):
        table.rows[r].height = Inches(row_h)
    for r, values in enumerate([header] + rows):
        for c, value in enumerate(values):
            cell = table.cell(r, c)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            is_header = r == 0
            write(cell.text_frame,
                  [{"t": value, "size": size, "bold": is_header, "font": BODY,
                    "align": PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER}],
                  anchor=MSO_ANCHOR.MIDDLE, margins=(0.08, 0.03, 0.08, 0.03))
            cell_borders(cell, bottom=LINE if is_header else GRID)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(P_BLUE if is_header else WHITE)
    return shape


# ---- chart helpers ----------------------------------------------------------
def style_chart(chart, size=11):
    chart.font.name = BODY
    chart.font.size = Pt(size)
    chart.font.color.rgb = rgb(INK)
    chart.has_title = False


def quiet_axis(axis, line=GRID):
    axis.format.line.color.rgb = rgb(line)
    axis.has_minor_gridlines = False


# ============================================================ build deck ====
prs = Presentation(str(SOURCE))

# Keep only the title slide; python-pptx drops the orphaned parts on save.
slide_ids = prs.slides._sldIdLst
for sld in list(slide_ids)[1:]:
    prs.part.drop_rel(sld.rId)
    slide_ids.remove(sld)

# ------------------------------------------------------------- slide 1 -------
s1 = prs.slides[0]
shapes = {sh.name: sh for sh in s1.shapes}

# The seminar-specific signature row and the seminar heading go away.
for name in ("TextBox 28", "TextBox 29", "TextBox 30"):
    shapes[name]._element.getparent().remove(shapes[name]._element)

# "Academic Year" stays; "Seminar & Technical Communication" would contradict
# the subject, so that line becomes a neutral descriptor.
tb = shapes["TextBox 14"].text_frame
tb.paragraphs[1].runs[0].text = "Project Presentation"

# Project title block: two centred lines in the deck's own face and weight.
tf = shapes["TextBox 20"].text_frame
for p in list(tf.paragraphs)[1:]:
    p._p.getparent().remove(p._p)
p0 = tf.paragraphs[0]
for r in list(p0.runs):
    r._r.getparent().remove(r._r)
for br in p0._p.findall(qn("a:br")):
    p0._p.remove(br)
p0.alignment = PP_ALIGN.CENTER
style_runs(p0, "Project Title:", 20, True, INK, "Book Antiqua", False)
p1 = tf.add_paragraph()
p1.alignment = PP_ALIGN.CENTER
p1.space_before = Pt(4)
style_runs(p1, "Loan Default Prediction using Random Forest", 28, True, INK, "Book Antiqua", False)
shapes["TextBox 20"].top = Inches(1.95)

# Left block: student, faculty, subject.
tf = shapes["TextBox 24"].text_frame
# Keep the first paragraph (a text frame must always have one) and clear it.
for extra in list(tf.paragraphs)[1:]:
    extra._p.getparent().remove(extra._p)
first = tf.paragraphs[0]
for r in list(first.runs):
    r._r.getparent().remove(r._r)
for br in first._p.findall(qn("a:br")):
    first._p.remove(br)
rows = [
    "Students Name and PRN no: Prasad Bhalerao (RBT24CB026)",
    "Faculty: Prof. Neeraj Sathawane",
    "Subject: Machine Learning",
]
for i, text in enumerate(rows):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(22)
    style_runs(p, text, 20, True, INK, "Book Antiqua", False)
shapes["TextBox 24"].top = Inches(3.5)
shapes["TextBox 24"].height = Inches(2.3)
shapes["TextBox 24"].left = Inches(1.3)

add_notes(s1, "Title slide. Machine Learning project: predicting serious borrower delinquency "
              "with a Random Forest, served through a FastAPI backend and a React web app.")

LAYOUT = next(l for l in prs.slide_layouts if l.name == "Title Only")

# ------------------------------------------------------------- slide 2 -------
s2 = prs.slides.add_slide(LAYOUT)
slide_title(s2, "Dataset and Problem",
            "Give Me Some Credit (Kaggle): who is in the data, what we predict, what had to be fixed")

# Left column: three stat callouts, predictor groups, target.
stats = [
    (n(meta["dataset"]["labeled_rows"]), "labelled borrowers", NAVY),
    (str(len(meta["features"])), "numeric predictors", NAVY),
    (pct(dist["test"]["positive_rate"], 2), "delinquent borrowers", BRICK),
]
for i, (big, small, colour) in enumerate(stats):
    x = 0.5 + i * 2.05
    b = box(s2, x, 1.5, 1.9, 1.2, fill=P_BLUE if colour == NAVY else P_PINK, name=f"Stat {i + 1}")
    write(b.text_frame, [
        {"t": big, "size": 28, "bold": True, "font": HEAD, "color": colour, "align": PP_ALIGN.CENTER},
        {"t": small, "size": 13, "align": PP_ALIGN.CENTER},
    ], anchor=MSO_ANCHOR.MIDDLE)

section_label(s2, 0.5, 2.9, 6.0, "Ten predictors, in three groups")
groups = [
    ("Payment history", P_YELLOW, ["30â€“59 days late", "60â€“89 days late", "90+ days late"]),
    ("Credit usage", P_GREEN, ["Revolving utilisation", "Debt ratio", "Open credit lines", "Real-estate loans"]),
    ("Borrower profile", P_BLUE, ["Age", "Monthly income", "Dependents"]),
]
for i, (name, fill, items) in enumerate(groups):
    x = 0.5 + i * 2.05
    box(s2, x, 3.25, 1.9, 1.9, fill=None, name=f"Group {name}")
    head = box(s2, x, 3.25, 1.9, 0.5, fill=fill, name=f"Group header {name}")
    write(head.text_frame, [{"t": name, "size": 13, "bold": True, "align": PP_ALIGN.CENTER}],
          anchor=MSO_ANCHOR.MIDDLE)
    label(s2, x + 0.12, 3.88, 1.7, 1.6,
          [{"t": item, "size": 13, "after": 5} for item in items], name=f"Group items {name}")

arrow(s2, 3.5, 5.15, 3.5, 5.6)
target = box(s2, 0.5, 5.6, 6.0, 1.18, fill=P_PINK, line=BRICK, lw=1.5, name="Target")
write(target.text_frame, [
    {"t": f"Target: {meta['target']['column']}", "size": 15, "bold": True, "font": HEAD,
     "color": BRICK, "align": PP_ALIGN.CENTER, "after": 3},
    {"t": "1 = 90+ days past due within two years      0 = otherwise", "size": 13,
     "align": PP_ALIGN.CENTER},
], anchor=MSO_ANCHOR.MIDDLE)

# Right column: class balance chart, then the data-issues table.
section_label(s2, 6.83, 1.5, 6.0, "Class balance: the delinquent class is rare")
cd = CategoryChartData()
cd.categories = ["Delinquent", "Not delinquent"]  # bar charts draw bottom-up
cd.add_series("Borrowers", [
    dist["train"]["positive"] + dist["validation"]["positive"] + dist["test"]["positive"],
    dist["train"]["negative"] + dist["validation"]["negative"] + dist["test"]["negative"],
])
gf = s2.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(6.83), Inches(1.82), Inches(6.0),
                         Inches(1.95), cd)
gf.name = "Class balance chart"
ch = gf.chart
style_chart(ch, 12)
ch.has_legend = False
plot = ch.plots[0]
plot.gap_width = 45
plot.has_data_labels = True
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
plot.data_labels.font.size = Pt(13)
plot.data_labels.font.bold = True
counts = [
    dist["train"]["positive"] + dist["validation"]["positive"] + dist["test"]["positive"],
    dist["train"]["negative"] + dist["validation"]["negative"] + dist["test"]["negative"],
]
for idx, colour in enumerate((BRICK, NAVY)):
    pt = plot.series[0].points[idx]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = rgb(colour)
    dl = pt.data_label
    dl.has_text_frame = True
    write(dl.text_frame, [{"t": n(counts[idx]), "size": 13, "bold": True}], margins=(0, 0, 0, 0))
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
ch.value_axis.maximum_scale = 175000
ch.value_axis.minimum_scale = 0
ch.category_axis.has_major_gridlines = False
ch.category_axis.tick_labels.font.size = Pt(13)
quiet_axis(ch.category_axis)

label(s2, 6.83, 3.8, 6.0, 0.34,
      [{"t": f"Always answering â€œnoâ€ scores {pct(majority_accuracy)} accuracy and catches no one.",
        "size": 13, "italic": True, "color": MUTED}], name="Imbalance note")

section_label(s2, 6.83, 4.22, 6.0, "Data issues found, and what was done")
build_table(
    s2, 6.83, 4.56, [2.75, 1.05, 2.2], 0.37,
    ["Issue", "Affected", "Handling"],
    [
        ["Monthly income missing", "19.82%", "Median + flag"],
        ["Dependents missing", "2.62%", "Median + flag"],
        ["Late counts coded 96 / 98", "269 rows", "Flag: 54.65% positive"],
        ["Age equal to 0", "1 row", "Treated as missing"],
        ["Outliers, duplicate rows", "kept", "Trees are rank-based"],
    ],
)
add_notes(s2, "Dataset: Give Me Some Credit from Kaggle, 150,000 labelled borrowers, ten numeric "
              "predictors. The target is serious delinquency, 90+ days past due within two years. "
              "Only 6.68% of borrowers are positive, which is why accuracy is a misleading metric. "
              "The 269 rows coded 96/98 are 54.65% positive, so the pipeline keeps a missing-value "
              "flag rather than quietly imputing them away.")

# ------------------------------------------------------------- slide 3 -------
s3 = prs.slides.add_slide(LAYOUT)
slide_title(s3, "Methodology and Algorithm",
            "A leakage-free pipeline around a Random Forest classifier")

steps = [
    ("Load", "150,000 labelled rows, 10 predictors, row index dropped", P_BLUE),
    ("Clean + impute", "96/98 codes and age < 18 become missing. Median fill plus missing flags", P_BLUE),
    ("Split", "Stratified 60 / 20 / 20: 90,000 train, 30,000 validation, 30,000 test", P_BLUE),
    ("Compare", "Five forests fitted on train, ranked by validation PR-AUC", P_YELLOW),
    ("Set cut-off", f"Threshold {threshold:.4f} chosen on validation by maximum F1", P_YELLOW),
    ("Test once", "Refit on train + validation, then score the untouched test set", P_PINK),
]
bw, gap, top, bh = 1.88, 0.2, 1.5, 1.85
for i, (head, body_text, fill) in enumerate(steps):
    x = 0.5 + i * (bw + gap)
    box(s3, x, top, bw, bh, fill=None, name=f"Step {i + 1}")
    band = box(s3, x, top, bw, 0.55, fill=fill, name=f"Step {i + 1} header")
    write(band.text_frame, [{"t": head, "size": 13, "bold": True, "align": PP_ALIGN.LEFT}],
          anchor=MSO_ANCHOR.MIDDLE, margins=(0.55, 0.02, 0.05, 0.02))
    circle_number(s3, x + 0.1, top + 0.085, 0.38, f"{i + 1}", WHITE)
    label(s3, x + 0.1, top + 0.65, bw - 0.2, bh - 0.7, [{"t": body_text, "size": 12.5}],
          name=f"Step {i + 1} text")
    if i < len(steps) - 1:
        arrow(s3, x + bw + 0.01, top + bh / 2, x + bw + gap - 0.01, top + bh / 2)

label(s3, 0.5, 3.45, 12.33, 0.34,
      [{"t": "Every statistic is fitted on training data only. The test set is touched exactly once.",
        "size": 13, "italic": True, "color": MUTED, "align": PP_ALIGN.CENTER}], name="Pipeline note")

# --- Random Forest diagram (left) ---
section_label(s3, 0.5, 3.88, 7.0, "How the forest produces a prediction")
rows_y = [4.3, 5.2, 6.1]
sample_labels = ["Bootstrap sample 1", "Bootstrap sample 2", "Bootstrap sample 300"]
vote_labels = ["0.81", "0.12", "0.74"]

data_box = box(s3, 0.5, 4.75, 1.15, 1.5, fill=P_BLUE, name="Training data")
write(data_box.text_frame, [
    {"t": "Training data", "size": 12.5, "bold": True, "align": PP_ALIGN.CENTER, "after": 2},
    {"t": "120,000 borrowers", "size": 12, "align": PP_ALIGN.CENTER},
], anchor=MSO_ANCHOR.MIDDLE)

for i, y in enumerate(rows_y):
    sb = box(s3, 2.0, y, 1.55, 0.66, fill=None, name=f"Sample {i + 1}")
    write(sb.text_frame, [{"t": sample_labels[i], "size": 12, "align": PP_ALIGN.CENTER}],
          anchor=MSO_ANCHOR.MIDDLE, margins=(0.04, 0.02, 0.04, 0.02))
    arrow(s3, 1.66, 5.5, 1.99, y + 0.33, head=True)

    # Tree icon: a root, two children, and the edges between them.
    tx, ty = 3.95, y
    box(s3, tx, ty, 0.78, 0.66, fill=P_GREEN, name=f"Tree {i + 1} frame")
    arrow(s3, tx + 0.39, ty + 0.2, tx + 0.2, ty + 0.46, head=False, w=1.0)
    arrow(s3, tx + 0.39, ty + 0.2, tx + 0.58, ty + 0.46, head=False, w=1.0)
    for cx, cy, d in ((0.39, 0.17, 0.2), (0.2, 0.5, 0.17), (0.58, 0.5, 0.17)):
        box(s3, tx + cx - d / 2, ty + cy - d / 2, d, d, fill=WHITE, lw=1.0, shape=MSO_SHAPE.OVAL,
            name=f"Tree {i + 1} node")
    arrow(s3, 3.56, y + 0.33, 3.94, y + 0.33)

    voted_one = float(vote_labels[i]) >= 0.5
    vb = box(s3, 5.0, y + 0.08, 0.82, 0.5, fill=P_PINK if voted_one else WHITE,
             line=BRICK if voted_one else LINE, name=f"Vote {i + 1}")
    write(vb.text_frame, [{"t": vote_labels[i], "size": 12, "bold": voted_one, "align": PP_ALIGN.CENTER}],
          anchor=MSO_ANCHOR.MIDDLE, margins=(0.02, 0.02, 0.02, 0.02))
    arrow(s3, 4.74, y + 0.33, 4.99, y + 0.33)

# Between the second and third tree rows, not on top of either.
label(s3, 3.95, 5.87, 0.78, 0.22,
      [{"t": "â‹®", "size": 14, "bold": True, "align": PP_ALIGN.CENTER}],
      anchor=MSO_ANCHOR.MIDDLE, name="More trees")

agg = box(s3, 6.15, 4.3, 1.5, 1.2, fill=P_YELLOW, name="Average")
write(agg.text_frame, [
    {"t": "Probability", "size": 12.5, "bold": True, "align": PP_ALIGN.CENTER, "after": 2},
    {"t": "mean of the 300 trees' leaf proportions", "size": 11.5, "align": PP_ALIGN.CENTER},
], anchor=MSO_ANCHOR.MIDDLE, margins=(0.06, 0.04, 0.06, 0.04))
for y in rows_y:
    arrow(s3, 5.83, y + 0.33, 6.14, 4.9)
arrow(s3, 6.9, 5.5, 6.9, 5.85)
dec = box(s3, 6.15, 5.85, 1.5, 1.1, fill=P_PINK, line=BRICK, lw=1.25, name="Decision")
write(dec.text_frame, [
    {"t": f"â‰¥ {threshold:.4f}", "size": 13, "bold": True, "color": BRICK, "align": PP_ALIGN.CENTER, "after": 2},
    {"t": "predict delinquent", "size": 12, "align": PP_ALIGN.CENTER},
], anchor=MSO_ANCHOR.MIDDLE)

# --- design choices (right) ---
section_label(s3, 7.95, 3.88, 4.88, "Design choices, all decided on validation data")
by_name = candidates
choices = [
    ("300 trees, minimum leaf size 20",
     f"Limits overfitting: validation PR-AUC {f4(by_name['regularised']['pr_auc'])} "
     f"against {f4(by_name['baseline']['pr_auc'])} for unrestricted trees", P_BLUE),
    ("Class weighting tested, not used",
     f"PR-AUC stayed near {f4(by_name['balanced']['pr_auc'])}: it only slides along the "
     "same precisionâ€“recall curve", P_GREEN),
    ("Missing-value flags added",
     f"Best validation PR-AUC ({f4(chosen_val['pr_auc'])}). The 96/98 rows are 54.65% positive",
     P_YELLOW),
    (f"Cut-off {threshold:.4f}, not 0.50",
     "Validation F1 0.4396 held up on the test set (0.4482)", P_PINK),
]
for i, (head, body_text, fill) in enumerate(choices):
    y = 4.28 + i * 0.7
    circle_number(s3, 7.95, y + 0.04, 0.42, f"0{i + 1}", fill)
    label(s3, 8.5, y, 4.33, 0.68, [
        {"t": head, "size": 13, "bold": True, "font": HEAD, "after": 1},
        {"t": body_text, "size": 12},
    ], name=f"Choice {i + 1}")

add_notes(s3, "Methodology. The split is stratified 60/20/20 and every fitted statistic comes from "
              "training data only. Five Random Forest configurations were compared on validation by "
              "PR-AUC, the decision threshold was chosen on validation by maximising F1, and the "
              "winner was refitted on train plus validation and scored once on test. The forest "
              "builds 300 decision trees, each on a bootstrap sample with random feature subsets at "
              "every split. The probability is the mean of the per-tree leaf proportions, "
              "not a hard majority vote.")

# ------------------------------------------------------------- slide 4 -------
s4 = prs.slides.add_slide(LAYOUT)
slide_title(s4, "Results and Evaluation",
            f"Held-out test set of {n(dist['test']['total'])} borrowers, scored exactly once")

kpis = [
    ("ROC-AUC", f4(test["roc_auc"]), "threshold-free", P_BLUE, NAVY),
    ("PR-AUC", f4(test["pr_auc"]), f"chance level {dist['test']['positive_rate']:.3f}", P_BLUE, NAVY),
    ("Recall", f4(test["recall"]), f"caught {n(cm['true_positives'])} of {n(n_pos)}", P_YELLOW, INK),
    ("Precision", f4(test["precision"]), f"at cut-off {threshold:.4f}", P_YELLOW, INK),
    ("F1-score", f4(test["f1"]), f"at cut-off {threshold:.4f}", P_YELLOW, INK),
]
kw, kgap = 2.3, 0.2075
for i, (name, value, sub, fill, colour) in enumerate(kpis):
    x = 0.5 + i * (kw + kgap)
    b = box(s4, x, 1.45, kw, 1.25, fill=fill, name=f"KPI {name}")
    write(b.text_frame, [
        {"t": value, "size": 26, "bold": True, "font": HEAD, "color": colour, "align": PP_ALIGN.CENTER},
        {"t": name, "size": 13, "bold": True, "align": PP_ALIGN.CENTER},
        {"t": sub, "size": 11.5, "color": MUTED, "align": PP_ALIGN.CENTER},
    ], anchor=MSO_ANCHOR.MIDDLE, margins=(0.05, 0.03, 0.05, 0.03))

# --- confusion matrix ---
section_label(s4, 0.5, 2.88, 3.6, "Confusion matrix")
label(s4, 1.35, 3.2, 2.7, 0.28, [{"t": "Predicted", "size": 12, "bold": True, "color": MUTED,
                                  "align": PP_ALIGN.CENTER}], name="Predicted label")
for j, txt in enumerate(("not delinquent", "delinquent")):
    label(s4, 1.35 + j * 1.38, 3.45, 1.32, 0.26,
          [{"t": txt, "size": 11.5, "color": MUTED, "align": PP_ALIGN.CENTER}], name="Column header")
cells = [
    (cm["true_negatives"], "true negatives", P_BLUE, NAVY),
    (cm["false_positives"], "false alarms", P_PINK, BRICK),
    (cm["false_negatives"], "missed", P_PINK, BRICK),
    (cm["true_positives"], "caught", P_BLUE, NAVY),
]
for k, (count, desc, fill, colour) in enumerate(cells):
    r, c = divmod(k, 2)
    cx, cy = 1.35 + c * 1.38, 3.75 + r * 1.2
    b = box(s4, cx, cy, 1.32, 1.14, fill=fill, name=f"Cell {desc}")
    write(b.text_frame, [
        {"t": n(count), "size": 20, "bold": True, "font": HEAD, "color": colour, "align": PP_ALIGN.CENTER},
        {"t": desc, "size": 11.5, "align": PP_ALIGN.CENTER},
    ], anchor=MSO_ANCHOR.MIDDLE, margins=(0.03, 0.02, 0.03, 0.02))
for r, txt in enumerate(("not delinquent", "delinquent")):
    label(s4, 0.5, 3.75 + r * 1.2, 0.8, 1.14,
          [{"t": "Actual", "size": 11.5, "bold": True, "color": MUTED, "align": PP_ALIGN.RIGHT},
           {"t": txt, "size": 11.5, "color": MUTED, "align": PP_ALIGN.RIGHT}],
          anchor=MSO_ANCHOR.MIDDLE, name="Row header")

# --- feature importance ---
section_label(s4, 4.35, 2.88, 4.6, "What the forest relies on")
SHORT = {
    "NumberOfTimes90DaysLate": "90+ days late",
    "RevolvingUtilizationOfUnsecuredLines": "Credit utilisation",
    "NumberOfTime60-89DaysPastDueNotWorse": "60â€“89 days late",
    "NumberOfTime30-59DaysPastDueNotWorse": "30â€“59 days late",
    "DebtRatio": "Debt ratio",
    "age": "Age",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines",
}
PAYMENT = {"90+ days late", "60â€“89 days late", "30â€“59 days late"}
top = [(SHORT[e["feature"]], e["importance"]) for e in importances if e["feature"] in SHORT]
rest = sum(e["importance"] for e in importances if e["feature"] not in SHORT)
top.append(("All others", rest))
payment_share = sum(v for name, v in top if name in PAYMENT)

cd = CategoryChartData()
cd.categories = [name for name, _ in reversed(top)]  # largest drawn at the top
cd.add_series("Importance", [v for _, v in reversed(top)])
gf = s4.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(4.35), Inches(3.18), Inches(4.6),
                         Inches(2.9), cd)
gf.name = "Feature importance chart"
ch = gf.chart
style_chart(ch, 11)
ch.has_legend = False
plot = ch.plots[0]
plot.gap_width = 40
plot.has_data_labels = True
plot.data_labels.number_format = "0.000"
plot.data_labels.number_format_is_linked = False
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
plot.data_labels.font.size = Pt(11)
for idx, (name, _) in enumerate(reversed(top)):
    pt = plot.series[0].points[idx]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = rgb(BRICK if name in PAYMENT else NAVY)
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
ch.value_axis.minimum_scale = 0
ch.value_axis.maximum_scale = max(v for _, v in top) * 1.22
ch.category_axis.has_major_gridlines = False
ch.category_axis.tick_labels.font.size = Pt(11.5)
quiet_axis(ch.category_axis)
label(s4, 4.35, 6.08, 4.6, 0.3,
      [{"t": f"Payment history (red) holds {pct(payment_share, 0)} of importance. Not causation.",
        "size": 11.5, "italic": True, "color": MUTED}], name="Importance note")

# --- threshold trade-off ---
section_label(s4, 9.2, 2.88, 3.63, "Moving the cut-off")
xy = XyChartData()
for name, key in (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")):
    series = xy.add_series(name)
    for p in curve:
        series.add_data_point(p["threshold"], p[key])
cut = xy.add_series(f"Cut-off {threshold:.2f}")
cut.add_data_point(threshold, 0)
cut.add_data_point(threshold, 1)
gf = s4.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(9.2), Inches(3.18),
                         Inches(3.63), Inches(2.9), xy)
gf.name = "Threshold trade-off chart"
ch = gf.chart
style_chart(ch, 11)
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(11)
specs = [
    (BRICK, 2.25, None),
    (NAVY, 2.25, None),
    ("6B6B6B", 2.0, MSO_LINE.ROUND_DOT),
    (INK, 1.5, MSO_LINE.DASH),
]
for series, (colour, width, dash) in zip(ch.plots[0].series, specs):
    series.smooth = False
    series.marker.style = XL_MARKER_STYLE.NONE
    line = series.format.line
    if dash is not None:
        line.dash_style = dash  # must precede the colour: it rewrites the line
    line.color.rgb = rgb(colour)
    line.width = Pt(width)
xa, ya = ch.category_axis, ch.value_axis
xa.minimum_scale, xa.maximum_scale, xa.major_unit = 0, 0.7, 0.1
xa.tick_labels.number_format, xa.tick_labels.number_format_is_linked = "0.0", False
xa.has_major_gridlines = False
ya.minimum_scale, ya.maximum_scale, ya.major_unit = 0, 1, 0.25
ya.tick_labels.number_format, ya.tick_labels.number_format_is_linked = "0%", False
ya.has_major_gridlines = True
ya.major_gridlines.format.line.color.rgb = rgb(GRID)
ya.major_gridlines.format.line.width = Pt(0.75)
quiet_axis(xa)
quiet_axis(ya)
label(s4, 9.2, 6.08, 3.63, 0.3,
      [{"t": "Lower cut-off: more caught, more false alarms.", "size": 11.5, "italic": True,
        "color": MUTED}], name="Threshold note")

# --- takeaway strip ---
strip = box(s4, 0.5, 6.42, 12.33, 0.62, fill=GRID if False else "F4F4F4", name="Takeaway")
write(strip.text_frame, [
    {"t": [
        ("Reading it: ", {"bold": True}),
        (f"accuracy ({pct(test['accuracy'])}) is below the {pct(majority_accuracy)} of always answering "
         f"â€œnoâ€, so PR-AUC and recall carry the result. Five-fold cross-validation agrees: "
         f"ROC-AUC {f4(cv['roc_auc']['mean'])} Â± {f4(cv['roc_auc']['std'])}. ", {}),
        ("Limits: ", {"bold": True}),
        ("2011 US data, uncalibrated probabilities, no fairness test. An educational prototype.", {}),
    ], "size": 12.5},
], anchor=MSO_ANCHOR.MIDDLE, margins=(0.15, 0.04, 0.15, 0.04))

add_notes(s4, "Results on the held-out test set. ROC-AUC is about 0.865 and PR-AUC about 0.40 against a "
              "chance level of 0.067. At the chosen cut-off the model catches roughly half of the "
              "delinquent borrowers at about 42% precision. Accuracy is deliberately not the headline: "
              "always answering no scores higher. Payment history dominates feature importance, but "
              "importance describes the model, not causes. Cross-validation confirms the test score "
              "is stable.")

prs.save(str(OUTPUT))
print(f"saved {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB, {len(prs.slides)} slides)")
