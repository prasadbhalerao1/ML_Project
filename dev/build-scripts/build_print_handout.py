"""Build the single print-ready PDF: 4 slides on page 1, technical sheet on page 2.

Both pages are A4 landscape so the sheet prints double-sided with the slides.
Rendered through headless Chrome, so the layout is exactly what prints. Every
number is read from the trained model's saved files.

Usage: python build_combined.py <project_root> <slides_png_dir> <output.html>
"""

from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

ROOT, PNG_DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ART = ROOT / "backend" / "ml" / "artifacts"
meta = json.loads((ART / "metadata.json").read_text(encoding="utf-8"))
rep = json.loads((ART / "evaluation_report.json").read_text(encoding="utf-8"))

test = rep["per_split_metrics"]["test"]
train = rep["per_split_metrics"]["train"]
cm = test["confusion_matrix"]
dist = rep["class_distribution"]
cv = rep["cross_validation"]["summary"]
thr = meta["decision_threshold"]["value"]
cand = {c["candidate"]: c for c in meta["validation_comparison"]}
curve = meta["test_threshold_curve"]
imp = meta["feature_importances"]
base = dist["test"]["positive_rate"]
n_pos = cm["true_positives"] + cm["false_negatives"]
n_neg = cm["true_negatives"] + cm["false_positives"]
leaf = meta["selected_configuration"]["params"]["min_samples_leaf"]
trees = meta["selected_configuration"]["params"]["n_estimators"]
# The winning configuration is refitted on train + validation before the
# final test score, so that is the row count each tree resamples from.
fit_rows = dist["train"]["total"] + dist["validation"]["total"]


def f4(v):
    return f"{v:.4f}"


def pc(v, d=2):
    return f"{v * 100:.{d}f}%"


def img_src(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


slides = [img_src(PNG_DIR / f"slide{i:02d}.png") for i in (1, 2, 3, 4)]
SHOTS = ROOT / "shots"
shot_result = img_src(SHOTS / "print-result.png")
shot_metrics = img_src(SHOTS / "print-metrics.png")

# ------------------------------------------------------- threshold curve ---
CW, CH = 196, 68
PAD_L, PAD_T, PAD_B = 20, 8, 13
x0, x1 = curve[0]["threshold"], curve[-1]["threshold"]
cx = lambda t: PAD_L + (t - x0) / (x1 - x0) * (CW - PAD_L - 4)
cy = lambda v: PAD_T + (1 - v) * (CH - PAD_T - PAD_B)
path = lambda k: " ".join(
    f"{'M' if i == 0 else 'L'} {cx(p['threshold']):.1f} {cy(p[k]):.1f}"
    for i, p in enumerate(curve)
)

# ------------------------------------------------------------ feature bars -
SHORT = {
    "NumberOfTimes90DaysLate": "90+ days late",
    "RevolvingUtilizationOfUnsecuredLines": "Credit utilisation",
    "NumberOfTime60-89DaysPastDueNotWorse": "60–89 days late",
    "NumberOfTime30-59DaysPastDueNotWorse": "30–59 days late",
    "DebtRatio": "Debt ratio",
    "age": "Age",
}
PAYMENT = {"90+ days late", "60–89 days late", "30–59 days late"}
top_feats = [(SHORT[e["feature"]], e["importance"]) for e in imp if e["feature"] in SHORT][:6]
fmax = max(v for _, v in top_feats)
payment_share = sum(e["importance"] for e in imp if SHORT.get(e["feature"]) in PAYMENT)
feature_rows = "".join(
    f'<div class="fbar"><span>{n}</span>'
    f'<span class="ftrack"><i style="width:{v / fmax * 100:.0f}%;'
    f'background:{"var(--brick)" if n in PAYMENT else "var(--navy)"}"></i></span>'
    f'<span class="fval">{v:.3f}</span></div>'
    for n, v in top_feats
)

# Type scale for the whole document. Landscape is wider than portrait, so the
# same content runs shorter; this fills the page. Tune this one number.
SCALE = 1.24


def scale_pt(css: str, factor: float) -> str:
    return re.sub(
        r"(\d+(?:\.\d+)?)pt",
        lambda m: f"{float(m.group(1)) * factor:.2f}pt",
        css,
    )


CSS = """
  @page { size: A4 landscape; margin: 6mm 7mm; }
  * { box-sizing: border-box; }
  :root {
    --ink:#161616; --mute:#5c5c5c; --line:#30363d;
    --navy:#2F5D8C; --brick:#B5483F;
    --blue:#E4EDF7; --green:#E2F2E7; --yellow:#FCF2D6; --pink:#FAE2E2;
  }
  html,body { margin:0; padding:0; }
  body { font-family:"Times New Roman",serif; color:var(--ink);
         font-size:7.5pt; line-height:1.2; }
  .page { page-break-after:always; }
  .page:last-child { page-break-after:auto; }

  /* ---------- page 1: slide handout ---------- */
  .slides { display:grid; grid-template-columns:1fr 1fr; gap:3mm; }
  .slides img { width:100%; display:block; border:1px solid var(--line); }
  .slidecap { text-align:center; font-size:7.4pt; color:var(--mute); margin-top:.8mm; }
  .hband { text-align:center; margin-bottom:2.5mm; }
  .hband .t { font-size:12pt; font-weight:700; }
  .hband .s { font-size:8pt; font-style:italic; color:var(--mute); margin-top:.5mm; }

  /* ---------- page 2: technical sheet ---------- */
  .sheet { display:flex; flex-direction:column; gap:2mm; zoom:0.72; }
  header { text-align:center; border-bottom:1.4px solid var(--line); padding-bottom:.9mm; }
  h1 { font-size:12.5pt; margin:0; font-weight:700; }
  h1 small { color:var(--navy); font-weight:700; font-size:12.5pt; }
  .by { font-size:7.6pt; font-style:italic; color:var(--mute); margin-top:.3mm; }

  .panel { border:1px solid var(--line); padding:1.4mm 1.8mm; }
  .cap { font-size:8.4pt; font-weight:700; color:var(--navy); margin:0 0 .9mm; }
  .row { display:grid; gap:2.2mm; }
  p { margin:0 0 .7mm; }
  .mute { color:var(--mute); }

  .chips { display:flex; gap:1.6mm; }
  .chip { flex:1; text-align:center; padding:.8mm .3mm; border:1px solid var(--line); }
  .chip .n { display:block; font-size:11pt; font-weight:700; line-height:1.05; }
  .chip .l { display:block; font-size:6.9pt; color:var(--mute); }

  .fx { display:grid; grid-template-columns:repeat(4,1fr); gap:1.2mm; }
  .fx .cell { border:1px solid var(--line); padding:.9mm 1.1mm; background:#fff; }
  .fx .t { font-size:7.3pt; font-weight:700; color:var(--navy); }
  .fx .e { font-size:9pt; font-style:italic; margin-top:.3mm; white-space:nowrap; }
  .fx .n { font-size:6.7pt; color:var(--mute); margin-top:.2mm; line-height:1.15; }
  .frac { display:inline-block; text-align:center; vertical-align:middle; }
  .frac .num { display:block; border-bottom:1px solid var(--ink); padding:0 .5mm; }
  .frac .den { display:block; padding:0 .5mm; }

  .cmwrap { display:grid; grid-template-columns:11mm 1fr 1fr; gap:.9mm; }
  .cmhead { font-size:6.9pt; color:var(--mute); text-align:center; }
  .cmside { font-size:6.9pt; color:var(--mute); display:flex; align-items:center;
            justify-content:flex-end; padding-right:.6mm; }
  .cmcell { border:1px solid var(--line); text-align:center; padding:1mm .3mm; }
  .cmcell .n { display:block; font-size:11pt; font-weight:700; line-height:1.05; }
  .cmcell .l { display:block; font-size:6.7pt; color:var(--mute); }
  .ok { background:var(--blue); } .bad { background:var(--pink); }

  .res { display:flex; gap:1.2mm; }
  .res .m { flex:1; border:1px solid var(--line); text-align:center; padding:.7mm .2mm; }
  .res .m .v { display:block; font-size:9.6pt; font-weight:700; line-height:1.05; }
  .res .m .k { display:block; font-size:6.7pt; color:var(--mute); }
  .hi { background:var(--blue); }

  .fbar { display:grid; grid-template-columns:23mm 1fr 8mm; gap:1mm;
          align-items:center; margin-bottom:.5mm; font-size:7pt; }
  .ftrack { background:#EDF0F3; height:2.2mm; display:block; }
  .ftrack i { display:block; height:2.2mm; }
  .fval { text-align:right; color:var(--mute); font-size:6.8pt; }

  ul { margin:0; padding-left:3mm; }
  li { margin-bottom:.45mm; }
  .shot { width:100%; height:auto; border:1px solid var(--line);
          display:block; }
  .row > * { min-width:0; }
  .foot { border:1px solid var(--line); background:#F6F7F8; padding:1.1mm 1.8mm; font-size:7.1pt; }
"""

DIAGRAM = f"""<svg viewBox="0 0 346 92" style="width:100%;height:auto">
  <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6"
    markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#30363d"/></marker></defs>
  <g font-family="Times New Roman,serif" font-size="8" fill="#161616">
    <rect x="1" y="28" width="48" height="32" fill="#E4EDF7" stroke="#30363d"/>
    <text x="25" y="41" text-anchor="middle" font-weight="700">Training</text>
    <text x="25" y="52" text-anchor="middle" font-size="7.2">120,000 rows</text>
    <g stroke="#30363d" fill="none" marker-end="url(#a)">
      <path d="M50 44 L63 15"/><path d="M50 44 L63 44"/><path d="M50 44 L63 74"/>
    </g>
    <g font-size="7.4">
      <rect x="64" y="4" width="64" height="21" fill="#fff" stroke="#30363d"/>
      <text x="96" y="14" text-anchor="middle" font-size="7">data copy 1</text><text x="96" y="21.5" text-anchor="middle" font-size="5.9" fill="#5B5B5B">rows drawn at random</text>
      <rect x="64" y="34" width="64" height="21" fill="#fff" stroke="#30363d"/>
      <text x="96" y="44" text-anchor="middle" font-size="7">data copy 2</text><text x="96" y="51.5" text-anchor="middle" font-size="5.9" fill="#5B5B5B">some twice, some left out</text>
      <rect x="64" y="64" width="64" height="21" fill="#fff" stroke="#30363d"/>
      <text x="96" y="74" text-anchor="middle" font-size="7">data copy {trees}</text><text x="96" y="81.5" text-anchor="middle" font-size="5.9" fill="#5B5B5B">every tree gets its own</text>
    </g>
    <g stroke="#30363d" fill="none" marker-end="url(#a)">
      <path d="M129 14 L140 14"/><path d="M129 44 L140 44"/><path d="M129 74 L140 74"/>
    </g>
    <g>
      <rect x="141" y="3" width="28" height="23" fill="#E2F2E7" stroke="#30363d"/>
      <rect x="141" y="33" width="28" height="23" fill="#E2F2E7" stroke="#30363d"/>
      <rect x="141" y="63" width="28" height="23" fill="#E2F2E7" stroke="#30363d"/>
      <g stroke="#30363d" fill="#fff">
        <path d="M155 10 L149 19 M155 10 L161 19"/><circle cx="155" cy="9" r="2.5"/>
        <circle cx="149" cy="20" r="2.1"/><circle cx="161" cy="20" r="2.1"/>
        <path d="M155 40 L149 49 M155 40 L161 49"/><circle cx="155" cy="39" r="2.5"/>
        <circle cx="149" cy="50" r="2.1"/><circle cx="161" cy="50" r="2.1"/>
        <path d="M155 70 L149 79 M155 70 L161 79"/><circle cx="155" cy="69" r="2.5"/>
        <circle cx="149" cy="80" r="2.1"/><circle cx="161" cy="80" r="2.1"/>
      </g>
    </g>
    <g stroke="#30363d" fill="none" marker-end="url(#a)">
      <path d="M170 14 L188 14"/><path d="M170 44 L188 44"/><path d="M170 74 L188 74"/>
    </g>
    <g>
      <rect x="189" y="4" width="27" height="20" fill="#FAE2E2" stroke="#B5483F"/>
      <text x="202" y="17" text-anchor="middle" font-weight="700">0.81</text>
      <rect x="189" y="34" width="27" height="20" fill="#fff" stroke="#30363d"/>
      <text x="202" y="47" text-anchor="middle">0.12</text>
      <rect x="189" y="64" width="27" height="20" fill="#FAE2E2" stroke="#B5483F"/>
      <text x="202" y="77" text-anchor="middle" font-weight="700">0.74</text>
    </g>
    <text x="202" y="61" text-anchor="middle" font-size="11" fill="#5c5c5c">&#8942;</text>
    <g stroke="#30363d" fill="none" marker-end="url(#a)">
      <path d="M217 14 L252 38"/><path d="M217 44 L252 44"/><path d="M217 74 L252 50"/>
    </g>
    <rect x="253" y="28" width="42" height="32" fill="#FCF2D6" stroke="#30363d"/>
    <text x="274" y="41" text-anchor="middle" font-weight="700">mean</text>
    <text x="274" y="52" text-anchor="middle">= p&#770;</text>
    <g stroke="#30363d" fill="none" marker-end="url(#a)"><path d="M296 44 L308 44"/></g>
    <rect x="309" y="28" width="36" height="32" fill="#FAE2E2" stroke="#B5483F"/>
    <text x="327" y="39" text-anchor="middle" font-size="7.4" font-weight="700"
          fill="#B5483F">p&#770; &#8805; {thr:.2f}</text>
    <text x="327" y="48" text-anchor="middle" font-size="6.5">predict</text>
    <text x="327" y="55" text-anchor="middle" font-size="6.5">delinquent</text>
  </g>
</svg>"""

CURVE = f"""<svg viewBox="0 0 {CW} {CH + 8}" style="width:100%;height:auto"
     font-family="Times New Roman,serif" font-size="6.2" fill="#5c5c5c">
  <line x1="{PAD_L}" y1="{cy(0):.0f}" x2="{CW - 4}" y2="{cy(0):.0f}" stroke="#C9CFD6"/>
  <line x1="{PAD_L}" y1="{cy(0.5):.0f}" x2="{CW - 4}" y2="{cy(0.5):.0f}" stroke="#EDF0F3"/>
  <line x1="{PAD_L}" y1="{cy(1):.0f}" x2="{CW - 4}" y2="{cy(1):.0f}" stroke="#EDF0F3"/>
  <text x="{PAD_L - 2}" y="{cy(1) + 2:.0f}" text-anchor="end">100%</text>
  <text x="{PAD_L - 2}" y="{cy(0.5) + 2:.0f}" text-anchor="end">50%</text>
  <text x="{PAD_L - 2}" y="{cy(0) + 2:.0f}" text-anchor="end">0%</text>
  <path d="{path('precision')}" fill="none" stroke="#B5483F" stroke-width="1.5"/>
  <path d="{path('recall')}" fill="none" stroke="#2F5D8C" stroke-width="1.5"/>
  <path d="{path('f1')}" fill="none" stroke="#6B6B6B" stroke-width="1.2" stroke-dasharray="2 1.6"/>
  <line x1="{cx(thr):.1f}" y1="{PAD_T}" x2="{cx(thr):.1f}" y2="{cy(0):.0f}"
        stroke="#161616" stroke-width="1" stroke-dasharray="2.5 2"/>
  <text x="{cx(thr) + 2:.0f}" y="{cy(0.86):.0f}" fill="#161616">{thr:.2f}</text>
  <text x="{PAD_L}" y="{CH + 2}">{x0:.2f}</text>
  <text x="{CW - 4}" y="{CH + 2}" text-anchor="end">{x1:.2f}</text>
  <text x="{(PAD_L + CW) / 2:.0f}" y="{CH + 7}" text-anchor="middle">decision threshold</text>
  <g font-size="6">
    <rect x="{PAD_L + 42}" y="{PAD_T - 6}" width="6" height="1.5" fill="#B5483F"/>
    <text x="{PAD_L + 50}" y="{PAD_T - 4}">precision</text>
    <rect x="{PAD_L + 80}" y="{PAD_T - 6}" width="6" height="1.5" fill="#2F5D8C"/>
    <text x="{PAD_L + 88}" y="{PAD_T - 4}">recall</text>
    <rect x="{PAD_L + 112}" y="{PAD_T - 6}" width="6" height="1.5" fill="#6B6B6B"/>
    <text x="{PAD_L + 120}" y="{PAD_T - 4}">F1</text>
  </g>
</svg>"""

SLIDE_TITLES = ["Title", "Dataset and Problem", "Methodology and Algorithm",
                "Results and Evaluation"]
slide_cells = "".join(
    f'<div><img src="{src}" alt="Slide {i + 1}"/>'
    f'<div class="slidecap">{i + 1} · {SLIDE_TITLES[i]}</div></div>'
    for i, src in enumerate(slides)
)

HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Loan Default Prediction — Handout</title>
<style>{scale_pt(CSS, SCALE)}</style></head><body>

<!-- ======================= PAGE 1 — slides ======================= -->
<div class="page">
  <div class="hband">
    <div class="t">Loan Default Prediction using Random Forest</div>
    <div class="s">Prasad Bhalerao (RBT24CB026) · Faculty: Prof. Neeraj Sathwale ·
      Subject: Machine Learning</div>
  </div>
  <div class="slides">{slide_cells}</div>
</div>

<!-- ======================= PAGE 2 — technical sheet ======================= -->
<div class="page"><div class="sheet">

<header>
  <h1>Technical Sheet <small>— algorithm, metrics and results</small></h1>
  <div class="by">Loan Default Prediction using Random Forest ·
    Prasad Bhalerao (RBT24CB026)</div>
</header>

<div class="row" style="grid-template-columns:1.06fr 1fr;">
  <!-- left column -->
  <div style="display:flex;flex-direction:column;gap:2mm">
    <section class="panel">
      <div class="cap">1 · Problem</div>
      <p><b>Supervised binary classification.</b> From 10 numeric borrower attributes,
      estimate P(serious delinquency within 2 years), then threshold it to a class.
      <span class="mute">Not regression (the output is a category) and not clustering
      (the groups come from labels).</span></p>
      <div class="chips" style="margin-top:1mm">
        <span class="chip"><span class="n">{meta['dataset']['labeled_rows']:,}</span><span class="l">borrowers</span></span>
        <span class="chip"><span class="n">{len(meta['features'])}</span><span class="l">features</span></span>
        <span class="chip" style="background:var(--pink)"><span class="n" style="color:var(--brick)">{pc(base)}</span><span class="l">positive</span></span>
      </div>
    </section>

    <section class="panel">
      <div class="cap">2 · Random Forest — how a prediction is made</div>
      {DIAGRAM}
      <div class="row" style="grid-template-columns:1fr 1fr;margin-top:1mm">
        <ul>
          <li><b>Make {trees} copies of the data.</b> Each copy is {fit_rows:,} rows drawn at
              random <i>with replacement</i> — a row can appear twice or not at all. Every
              tree therefore learns from slightly different data.
              <span class="mute">(The statistics name for this resampling is a
              <i>bootstrap</i>; building one tree per copy is <i>bagging</i>.)</span></li>
          <li><b>Grow one tree per copy.</b> At each node, look at a <i>random handful of
              the 10 features</i> — not all of them — and take the split that separates
              the classes best. Random features keep the trees from being near-copies of
              each other, which is what makes averaging them worthwhile.</li>
          <li><b>Stop</b> when a leaf holds fewer than {leaf} borrowers.</li>
        </ul>
        <div class="fx" style="grid-template-columns:1fr;gap:1mm">
          <div class="cell" style="background:var(--green)">
            <span class="t">Choosing a split — Gini impurity</span>
            <div class="e">G = 1 − Σ p<sub>k</sub>²</div>
            <div class="n">0 = pure node. Take the split with the largest weighted
            drop in G:</div>
            <div class="e" style="margin-top:.4mm">ΔG = G<sub>parent</sub> −
            <span style="white-space:nowrap">(n<sub>L</sub>/n)G<sub>L</sub></span> −
            <span style="white-space:nowrap">(n<sub>R</sub>/n)G<sub>R</sub></span></div>
          </div>
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="cap">3 · The prediction, step by step</div>
      <div class="fx" style="grid-template-columns:1fr 1fr 1fr;gap:1.2mm">
        <div class="cell">
          <span class="t">1 · Each tree answers</span>
          <div class="e"><span class="frac"><span class="num">n⁺ in leaf</span><span
            class="den">n in leaf</span></span> = p<sub>t</sub>(x)</div>
          <div class="n">Borrower x falls into one leaf. That tree reports the share of
          that leaf's training borrowers who defaulted.</div>
        </div>
        <div class="cell" style="background:var(--yellow)">
          <span class="t">2 · The forest averages</span>
          <div class="e">p̂(x) = <span class="frac"><span class="num">1</span><span
            class="den">T</span></span> Σ<sub>t=1</sub><sup>T</sup> p<sub>t</sub>(x)</div>
          <div class="n">T = {trees}. The <b>mean of the {trees} proportions</b> — not a
          majority vote of {trees} yes/no answers.</div>
        </div>
        <div class="cell" style="background:var(--pink)">
          <span class="t">3 · Compare to the cut-off</span>
          <div class="e">ŷ = 1 if p̂(x) ≥ τ</div>
          <div class="n">τ = <b>{thr:.4f}</b>, fixed on validation data — not the
          usual 0.50. Below τ → predict “no delinquency”.</div>
        </div>
      </div>
      <p class="mute" style="margin-top:1.2mm;font-size:7.1pt"><b>Worked example</b>
      (the diagram above): the trees return 0.81, 0.12, … , 0.74. Their mean is p̂ = 0.43.
      Since 0.43 ≥ {thr:.4f}, the borrower is flagged. The reported figure is that 0.43,
      a <i>ranking score</i> — not a claim that 43 in 100 such borrowers default.</p>
    </section>
  </div>

  <!-- right column -->
  <div style="display:flex;flex-direction:column;gap:2mm">
    <section class="panel">
      <div class="cap">4 · Evaluation metrics — from TP / FP / TN / FN</div>
      <div class="fx">
        <div class="cell"><span class="t">Accuracy</span>
          <div class="e"><span class="frac"><span class="num">TP + TN</span><span class="den">N</span></span></div>
          <div class="n">Misleading when one class is rare</div></div>
        <div class="cell"><span class="t">Precision</span>
          <div class="e"><span class="frac"><span class="num">TP</span><span class="den">TP + FP</span></span></div>
          <div class="n">Of those flagged, how many were right</div></div>
        <div class="cell"><span class="t">Recall (TPR)</span>
          <div class="e"><span class="frac"><span class="num">TP</span><span class="den">TP + FN</span></span></div>
          <div class="n">Of the truly delinquent, how many found</div></div>
        <div class="cell"><span class="t">F1</span>
          <div class="e">2 · <span class="frac"><span class="num">P · R</span><span class="den">P + R</span></span></div>
          <div class="n">Harmonic mean — the weaker one dominates</div></div>
        <div class="cell"><span class="t">FPR</span>
          <div class="e"><span class="frac"><span class="num">FP</span><span class="den">FP + TN</span></span></div>
          <div class="n">Divides by {n_neg:,} negatives → flattering</div></div>
        <div class="cell"><span class="t">ROC-AUC</span>
          <div class="e">P(s⁺ &gt; s⁻)</div>
          <div class="n">Area under TPR–FPR. Chance = 0.5</div></div>
        <div class="cell" style="background:var(--yellow)"><span class="t">PR-AUC</span>
          <div class="e">∫ P dR</div>
          <div class="n">Baseline = positive rate = {base:.4f}</div></div>
        <div class="cell" style="background:var(--yellow)"><span class="t">Why PR-AUC here</span>
          <div class="n" style="margin-top:.8mm">FPR's huge denominator hides false
          alarms. Precision divides only by what was flagged, so it stays honest.</div></div>
      </div>
    </section>

    <section class="panel">
      <div class="cap">5 · Results — held-out test set ({dist['test']['total']:,} borrowers, scored once)</div>
      <div class="row" style="grid-template-columns:1fr 52mm;">
        <div>
          <div class="cmwrap">
            <div></div><div class="cmhead">pred. 0</div><div class="cmhead">pred. 1</div>
            <div class="cmside">actual 0</div>
            <div class="cmcell ok"><span class="n" style="color:var(--navy)">{cm['true_negatives']:,}</span><span class="l">TN</span></div>
            <div class="cmcell bad"><span class="n" style="color:var(--brick)">{cm['false_positives']:,}</span><span class="l">FP · false alarm</span></div>
            <div class="cmside">actual 1</div>
            <div class="cmcell bad"><span class="n" style="color:var(--brick)">{cm['false_negatives']:,}</span><span class="l">FN · missed</span></div>
            <div class="cmcell ok"><span class="n" style="color:var(--navy)">{cm['true_positives']:,}</span><span class="l">TP · caught</span></div>
          </div>
          <div class="res" style="margin-top:1.2mm">
            <span class="m hi"><span class="v" style="color:var(--navy)">{f4(test['roc_auc'])}</span><span class="k">ROC-AUC</span></span>
            <span class="m hi"><span class="v" style="color:var(--navy)">{f4(test['pr_auc'])}</span><span class="k">PR-AUC</span></span>
            <span class="m"><span class="v">{f4(test['recall'])}</span><span class="k">Recall</span></span>
            <span class="m"><span class="v">{f4(test['precision'])}</span><span class="k">Precision</span></span>
            <span class="m"><span class="v">{f4(test['f1'])}</span><span class="k">F1</span></span>
          </div>
          <p style="margin-top:1.2mm"><b>Accuracy {f4(test['accuracy'])}</b>
          <span class="mute">— below the {pc(1 - base, 1)} of always answering “no”.
          That is the point.</span> <span class="mute">Caught {cm['true_positives']:,} of
          {n_pos:,} delinquents. 5-fold CV ROC-AUC {f4(cv['roc_auc']['mean'])} ±
          {f4(cv['roc_auc']['std'])}; gap train−test +{train['roc_auc'] - test['roc_auc']:.4f}
          ROC, +{train['pr_auc'] - test['pr_auc']:.4f} PR.</span></p>
        </div>
        <div>
          <div style="font-size:7.1pt;font-weight:700;margin-bottom:.5mm">Feature importance</div>
          {feature_rows}
          <p class="mute" style="font-size:6.8pt">Payment history (red) = {pc(payment_share, 0)}.
          Describes splits, not causation.</p>
        </div>
      </div>
    </section>
    <section class="panel">
      <div class="cap">6 · Class imbalance and the decision threshold</div>
      <div class="row" style="grid-template-columns:1fr 58mm;">
        <ul>
          <li>Only <b>{pc(base)}</b> positive. Three remedies: resampling, class
              weighting, moving the threshold.</li>
          <li><b>Class weighting tested, rejected</b> — recall
              {f4(cand['regularised']['recall'])} → {f4(cand['balanced']['recall'])} but
              PR-AUC {f4(cand['regularised']['pr_auc'])} → {f4(cand['balanced']['pr_auc'])}:
              it slides <i>along</i> the curve, it does not lift it.</li>
          <li><b>Threshold chosen on validation</b> by maximising F1 → <b>{thr:.4f}</b>,
              not 0.50. Validation F1 0.4396 → test {f4(test['f1'])}.</li>
          <li><b>Protocol.</b> Stratified 60/20/20; statistics fitted on train only;
              5 configurations compared on validation; test touched once.</li>
        </ul>
        <div>{CURVE}</div>
      </div>
    </section>
  </div>
</div>

<section class="panel" style="margin-top:2mm">
  <div class="cap">7 · The same arithmetic, running live &nbsp;·&nbsp;
    <span style="font-weight:400">delinquency-risk.vercel.app</span></div>
  <div class="row" style="grid-template-columns:1.25fr 1fr;gap:3mm;align-items:start">
    <div>
      <img src="{shot_result}" alt="Prediction result" class="shot"/>
      <p class="mute" style="font-size:6.9pt;margin:.5mm 0 0"><b>Panel 3, executed.</b>
      86% utilisation, four late episodes → the {trees} trees average to
      <b>p̂ = 0.501</b>; 0.501 ≥ {thr:.4f}, so class 1.</p>
    </div>
    <div>
      <img src="{shot_metrics}" alt="Live metrics" class="shot"/>
      <p class="mute" style="font-size:6.9pt;margin:.5mm 0 0"><b>Panel 5, served.</b>
      Read from the model file at start-up — nothing here is typed by hand.</p>
    </div>
  </div>
</section>

<div class="foot"><b>Limitations.</b> US data from ~2011 with no date column, so no
out-of-time validation — present-day performance unknown. Probabilities are
<b>uncalibrated</b>: 0.30 is a ranking score, not “30 in 100 default”. No protected
attributes, so fairness cannot be tested — absence of measured bias is not evidence of
fairness. Maximising F1 weights a miss and a false alarm equally; lending economics do
not. <b>Educational prototype — not for real credit decisions.</b></div>

</div></div>
</body></html>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
