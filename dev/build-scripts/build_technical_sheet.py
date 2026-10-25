"""Build the one-page technical sheet (A4, print-ready PDF).

A reference sheet rather than prose: diagrams, boxed formulas and real numbers,
with text cut back to labels and short phrases. Rendered through headless
Chrome, so the layout is exactly what prints. Every number is read from the
trained model's saved files.

Usage: python build_sheet.py <project_root> <output.html>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
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


def f4(v):
    return f"{v:.4f}"


def pc(v, d=2):
    return f"{v * 100:.{d}f}%"


# ------------------------------------------------------- threshold curve ---
CW, CH = 196, 70
PAD_L, PAD_B, PAD_T = 20, 13, 9
x0, x1 = curve[0]["threshold"], curve[-1]["threshold"]


def cx(t):
    return PAD_L + (t - x0) / (x1 - x0) * (CW - PAD_L - 4)


def cy(v):
    return PAD_T + (1 - v) * (CH - PAD_T - PAD_B)


def path(key):
    return " ".join(
        f"{'M' if i == 0 else 'L'} {cx(p['threshold']):.1f} {cy(p[key]):.1f}"
        for i, p in enumerate(curve)
    )


# --------------------------------------------------------- feature bars ----
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
    f'<div class="fbar"><span>{name}</span>'
    f'<span class="ftrack"><i style="width:{v / fmax * 100:.0f}%;'
    f'background:{"var(--brick)" if name in PAYMENT else "var(--navy)"}"></i></span>'
    f'<span class="fval">{v:.3f}</span></div>'
    for name, v in top_feats
)

CSS = """
  @page { size: A4 portrait; margin: 7mm 8mm; }
  * { box-sizing: border-box; }
  :root {
    --ink:#161616; --mute:#5c5c5c; --line:#30363d;
    --navy:#2F5D8C; --brick:#B5483F;
    --blue:#E4EDF7; --green:#E2F2E7; --yellow:#FCF2D6; --pink:#FAE2E2;
  }
  html,body { margin:0; padding:0; }
  body { font-family:"Times New Roman",serif; color:var(--ink);
         font-size:8.1pt; line-height:1.22; }
  /* One factor fills the page; tune this alone, not 25 separate sizes. */
  .sheet { display:flex; flex-direction:column; gap:2mm; zoom:1.03; }

  header { text-align:center; border-bottom:1.4px solid var(--line); padding-bottom:1mm; }
  h1 { font-size:13.5pt; margin:0; font-weight:700; }
  h1 small { color:var(--navy); font-weight:700; font-size:13.5pt; }
  .by { font-size:7.9pt; font-style:italic; color:var(--mute); margin-top:.4mm; }

  .panel { border:1px solid var(--line); padding:1.5mm 2mm; }
  .cap { font-size:8.9pt; font-weight:700; color:var(--navy); margin:0 0 1mm; }
  .row { display:grid; gap:2.4mm; }
  p { margin:0 0 .8mm; }
  .mute { color:var(--mute); }

  .chips { display:flex; gap:1.8mm; }
  .chip { flex:1; text-align:center; padding:.9mm .4mm; border:1px solid var(--line); }
  .chip .n { display:block; font-size:12pt; font-weight:700; line-height:1.05; }
  .chip .l { display:block; font-size:7.3pt; color:var(--mute); }

  .fx { display:grid; grid-template-columns:repeat(4,1fr); gap:1.4mm; }
  .fx .cell { border:1px solid var(--line); padding:1mm 1.3mm; background:#fff; }
  .fx .t { font-size:7.7pt; font-weight:700; color:var(--navy); }
  .fx .e { font-size:9.6pt; font-style:italic; margin-top:.4mm; white-space:nowrap; }
  .fx .n { font-size:7pt; color:var(--mute); margin-top:.3mm; line-height:1.18; }
  .frac { display:inline-block; text-align:center; vertical-align:middle; }
  .frac .num { display:block; border-bottom:1px solid var(--ink); padding:0 .6mm; }
  .frac .den { display:block; padding:0 .6mm; }

  .cmwrap { display:grid; grid-template-columns:12mm 1fr 1fr; gap:1mm; }
  .cmhead { font-size:7.2pt; color:var(--mute); text-align:center; }
  .cmside { font-size:7.2pt; color:var(--mute); display:flex; align-items:center;
            justify-content:flex-end; padding-right:.7mm; }
  .cmcell { border:1px solid var(--line); text-align:center; padding:1.1mm .4mm; }
  .cmcell .n { display:block; font-size:12pt; font-weight:700; line-height:1.05; }
  .cmcell .l { display:block; font-size:7pt; color:var(--mute); }
  .ok { background:var(--blue); } .bad { background:var(--pink); }

  .res { display:flex; gap:1.4mm; }
  .res .m { flex:1; border:1px solid var(--line); text-align:center; padding:.8mm .3mm; }
  .res .m .v { display:block; font-size:10.5pt; font-weight:700; line-height:1.05; }
  .res .m .k { display:block; font-size:7pt; color:var(--mute); }
  .hi { background:var(--blue); }

  .fbar { display:grid; grid-template-columns:25mm 1fr 9mm; gap:1.1mm;
          align-items:center; margin-bottom:.6mm; font-size:7.4pt; }
  .ftrack { background:#EDF0F3; height:2.4mm; display:block; }
  .ftrack i { display:block; height:2.4mm; }
  .fval { text-align:right; color:var(--mute); font-size:7.1pt; }

  ul { margin:0; padding-left:3.2mm; }
  li { margin-bottom:.5mm; }
  .foot { border:1px solid var(--line); background:#F6F7F8; padding:1.2mm 2mm; font-size:7.5pt; }
"""

DIAGRAM = f"""<svg viewBox="0 0 346 92" style="width:100%;height:auto">
  <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6"
    markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#30363d"/></marker></defs>
  <g font-family="Times New Roman,serif" font-size="8" fill="#161616">
    <rect x="1" y="28" width="48" height="32" fill="#E4EDF7" stroke="#30363d"/>
    <text x="25" y="41" text-anchor="middle" font-weight="700">Training</text>
    <text x="25" y="52" text-anchor="middle" font-size="7.2">120,000 rows</text>
    <g stroke="#30363d" fill="none" marker-end="url(#a)">
      <path d="M50 44 L70 15"/><path d="M50 44 L70 44"/><path d="M50 44 L70 74"/>
    </g>
    <g font-size="7.4">
      <rect x="71" y="4" width="50" height="21" fill="#fff" stroke="#30363d"/>
      <text x="96" y="18" text-anchor="middle">data copy 1</text>
      <rect x="71" y="34" width="50" height="21" fill="#fff" stroke="#30363d"/>
      <text x="96" y="48" text-anchor="middle">data copy 2</text>
      <rect x="71" y="64" width="50" height="21" fill="#fff" stroke="#30363d"/>
      <text x="96" y="78" text-anchor="middle">data copy {trees}</text>
    </g>
    <g stroke="#30363d" fill="none" marker-end="url(#a)">
      <path d="M122 14 L140 14"/><path d="M122 44 L140 44"/><path d="M122 74 L140 74"/>
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
     font-family="Times New Roman,serif" font-size="6.4" fill="#5c5c5c">
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
  <g font-size="6.2">
    <rect x="{PAD_L + 42}" y="{PAD_T - 6}" width="6" height="1.5" fill="#B5483F"/>
    <text x="{PAD_L + 50}" y="{PAD_T - 4}">precision</text>
    <rect x="{PAD_L + 80}" y="{PAD_T - 6}" width="6" height="1.5" fill="#2F5D8C"/>
    <text x="{PAD_L + 88}" y="{PAD_T - 4}">recall</text>
    <rect x="{PAD_L + 112}" y="{PAD_T - 6}" width="6" height="1.5" fill="#6B6B6B"/>
    <text x="{PAD_L + 120}" y="{PAD_T - 4}">F1</text>
  </g>
</svg>"""

HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Technical Sheet</title>
<style>{CSS}</style></head>
<body><div class="sheet">

<header>
  <h1>Loan Default Prediction using Random Forest <small>— Technical Sheet</small></h1>
  <div class="by">Prasad Bhalerao (RBT24CB026) · Faculty: Prof. Neeraj Sathwale ·
    Subject: Machine Learning</div>
</header>

<section class="panel">
  <div class="cap">1 · Problem</div>
  <div class="row" style="grid-template-columns:1fr 84mm; align-items:center;">
    <p style="margin:0"><b>Supervised binary classification.</b> From 10 numeric borrower
    attributes, estimate P(serious delinquency within 2 years), then threshold it to a
    class. <span class="mute">Not regression (the output is a category) and not clustering
    (the groups come from labels).</span></p>
    <div class="chips">
      <span class="chip"><span class="n">{meta['dataset']['labeled_rows']:,}</span><span class="l">borrowers</span></span>
      <span class="chip"><span class="n">{len(meta['features'])}</span><span class="l">features</span></span>
      <span class="chip" style="background:var(--pink)"><span class="n" style="color:var(--brick)">{pc(base)}</span><span class="l">positive</span></span>
    </div>
  </div>
</section>

<section class="panel">
  <div class="cap">2 · Random Forest — how a prediction is made</div>
  <div class="row" style="grid-template-columns:116mm 1fr;">
    {DIAGRAM}
    <div>
      <ul>
        <li><b>{trees} copies of the data,</b> each drawn at random <i>with replacement</i> — a row may appear twice or not at all, so every tree sees different data. <span class="mute">(This resampling is a <i>bootstrap</i>.)</span></li>
        <li><b>Split.</b> At each node search a <i>random feature subset</i>; take the
            split that most reduces Gini.</li>
        <li><b>Stop.</b> Leaf must hold ≥ {leaf} samples.</li>
        <li><b>Predict.</b> Each tree gives its leaf's positive proportion; the forest
            <b>averages</b> them — not a majority vote.</li>
      </ul>
      <div class="fx" style="grid-template-columns:1fr;margin-top:1.2mm">
        <div class="cell" style="background:var(--green)">
          <span class="t">Gini impurity</span>
          <div class="e">G = 1 − Σ p<sub>k</sub>²</div>
          <div class="n">Lower = purer node. The split maximises the weighted drop in G.</div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="panel">
  <div class="cap">3 · Evaluation metrics — from TP / FP / TN / FN</div>
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
      <div class="n" style="margin-top:1mm">FPR's huge denominator hides false alarms.
      Precision divides only by what was flagged, so it stays honest.</div></div>
  </div>
</section>

<section class="panel">
  <div class="cap">4 · Results — held-out test set ({dist['test']['total']:,} borrowers, scored once)</div>
  <div class="row" style="grid-template-columns:74mm 1fr 58mm;">
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
      <p class="mute" style="margin-top:1mm">Caught {cm['true_positives']:,} of {n_pos:,}
      delinquents at cut-off {thr:.4f}.</p>
    </div>
    <div>
      <div class="res" style="margin-bottom:1.2mm">
        <span class="m hi"><span class="v" style="color:var(--navy)">{f4(test['roc_auc'])}</span><span class="k">ROC-AUC</span></span>
        <span class="m hi"><span class="v" style="color:var(--navy)">{f4(test['pr_auc'])}</span><span class="k">PR-AUC</span></span>
      </div>
      <div class="res">
        <span class="m"><span class="v">{f4(test['recall'])}</span><span class="k">Recall</span></span>
        <span class="m"><span class="v">{f4(test['precision'])}</span><span class="k">Precision</span></span>
        <span class="m"><span class="v">{f4(test['f1'])}</span><span class="k">F1</span></span>
      </div>
      <p style="margin-top:1.2mm"><b>Accuracy {f4(test['accuracy'])}</b>
      <span class="mute">— below the {pc(1 - base, 1)} of always answering “no”.
      That is the point.</span></p>
      <p class="mute">5-fold CV: ROC-AUC {f4(cv['roc_auc']['mean'])} ± {f4(cv['roc_auc']['std'])}.
      Gap train−test: +{train['roc_auc'] - test['roc_auc']:.4f} ROC, +{train['pr_auc'] - test['pr_auc']:.4f} PR.</p>
    </div>
    <div>
      <div style="font-size:7.5pt;font-weight:700;margin-bottom:.5mm">Feature importance</div>
      {feature_rows}
      <p class="mute" style="font-size:7.1pt">Payment history (red) = {pc(payment_share, 0)}.
      Describes splits, not causation.</p>
    </div>
  </div>
</section>

<section class="panel">
  <div class="cap">5 · Class imbalance and the decision threshold</div>
  <div class="row" style="grid-template-columns:1fr 66mm;">
    <ul>
      <li>Only <b>{pc(base)}</b> positive. Three remedies: resampling, class weighting,
          moving the threshold.</li>
      <li><b>Class weighting tested, rejected</b> — recall {f4(cand['regularised']['recall'])}
          → {f4(cand['balanced']['recall'])} but PR-AUC {f4(cand['regularised']['pr_auc'])}
          → {f4(cand['balanced']['pr_auc'])}: it slides <i>along</i> the curve, it does not
          lift it.</li>
      <li><b>Threshold chosen on validation</b> by maximising F1 → <b>{thr:.4f}</b>, not 0.50.
          Validation F1 0.4396 → test {f4(test['f1'])}, so it generalised.</li>
      <li><b>Protocol.</b> Stratified 60/20/20; statistics fitted on train only;
          5 configurations compared on validation; test touched once.</li>
    </ul>
    <div>{CURVE}</div>
  </div>
</section>

<div class="foot"><b>Limitations.</b> US data from ~2011 with no date column, so no
out-of-time validation — present-day performance unknown. Probabilities are
<b>uncalibrated</b>: 0.30 is a ranking score, not “30 in 100 default”. No protected
attributes, so fairness cannot be tested — absence of measured bias is not evidence of
fairness. Maximising F1 weights a miss and a false alarm equally; lending economics do
not. <b>Educational prototype — not for real credit decisions.</b></div>

</div></body></html>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT}")
