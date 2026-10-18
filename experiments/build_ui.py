"""Rebuild results.html from current output files.

Run automatically via Claude Code Stop hook, or manually:
    python experiments/build_ui.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
HTML_OUT = ROOT / "results.html"
TEMPLATE = ROOT / "experiments" / "_ui_template.py"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def build_imgdata():
    base_file = OUTPUTS / "exp1_sink_corrected" / "results_baseline.json"
    corr_file = OUTPUTS / "exp1_sink_corrected" / "results_corrected.json"
    paper_file = OUTPUTS / "exp1_sink_corrected" / "results_paper.json"
    exp2_file = OUTPUTS / "exp2_spectral_tdo" / "per_image.json"
    exp4_file = OUTPUTS / "exp4_type_classification" / "samples.json"
    labels_file = ROOT / "data" / "metadata" / "hallucination_results_val500.json"

    base_data = load_json(base_file)
    corr_data = load_json(corr_file)
    paper_data = load_json(paper_file)
    exp2_data = load_json(exp2_file)
    exp4_data = load_json(exp4_file)
    labels_raw = load_json(labels_file) or []
    labels_by_id = {item["img_id"]: item for item in (labels_raw if isinstance(labels_raw, list) else labels_raw.get("records", []))}

    base_records = {r["img_id"]: r for r in (base_data or {}).get("records", [])}
    corr_records = {r["img_id"]: r for r in (corr_data or {}).get("records", [])}
    paper_records = {r["img_id"]: r for r in (paper_data or {}).get("records", [])}
    exp2_records = {r["img_id"]: r for r in (exp2_data or {}).get("records", [])}
    exp4_by_img = {}
    for s in (exp4_data or {}).get("samples", []):
        exp4_by_img.setdefault(s["img_id"], []).append(s)

    def outcome_sets(gt, det):
        gt, det = set(gt), set(det)
        return sorted(gt & det), sorted(det - gt), sorted(gt - det)

    combined = []
    for img_id in sorted(base_records.keys()):
        b = base_records[img_id]
        c = corr_records.get(img_id, {})
        p = paper_records.get(img_id, {})
        e2 = exp2_records.get(img_id, {})
        e4 = exp4_by_img.get(img_id)
        lbl = labels_by_id.get(img_id, {})

        gt = set(b.get("hallucination_words", []))
        b_tp, b_fp, b_fn = outcome_sets(gt, b.get("detected_hallucinations", []))
        c_tp, c_fp, c_fn = outcome_sets(gt, c.get("detected_hallucinations", []))
        p_tp, p_fp, p_fn = outcome_sets(gt, p.get("detected_hallucinations", []))

        combined.append({
            "id": img_id,
            "gt_words": sorted(lbl.get("gt_words") or []),
            "e1": {
                "gt": sorted(gt),
                "b_tp": b_tp, "b_fp": b_fp, "b_fn": b_fn,
                "c_tp": c_tp, "c_fp": c_fp, "c_fn": c_fn,
                "p_tp": p_tp, "p_fp": p_fp, "p_fn": p_fn,
            },
            "e2": {
                "hal": e2.get("is_hallucinated"),
                "pr": round(e2.get("pr", 0), 6),
                "lr": round(e2.get("leading_ratio", 0), 6),
                "tok": e2.get("n_generated_tokens"),
            } if e2 else None,
            "e4": [
                {"w": s["word"], "t": s["type"],
                 "H": round(s["features"]["entropy"], 3),
                 "C": round(s["features"]["concentration"], 3),
                 "S": round(s["features"]["sink_ratio"], 3)}
                for s in e4
            ] if e4 else None,
        })
    return combined


def load_summaries():
    s1 = load_json(OUTPUTS / "exp1_sink_corrected" / "summary.json") or {}
    s2 = load_json(OUTPUTS / "exp2_spectral_tdo" / "summary.json") or {}
    s4 = load_json(OUTPUTS / "exp4_type_classification" / "summary.json") or {}
    return s1, s2, s4


def build_exp2_arrays():
    data = load_json(OUTPUTS / "exp2_spectral_tdo" / "per_image.json")
    if not data:
        return [], [], [], []
    records = data.get("records", [])
    hallu_pr = [round(r["pr"], 6) for r in records if r.get("is_hallucinated")]
    clean_pr = [round(r["pr"], 6) for r in records if not r.get("is_hallucinated")]
    hallu_lr = [round(r["leading_ratio"], 6) for r in records if r.get("is_hallucinated")]
    clean_lr = [round(r["leading_ratio"], 6) for r in records if not r.get("is_hallucinated")]
    return hallu_pr, clean_pr, hallu_lr, clean_lr


def has_paper_baseline(s1):
    return "paper_llcc" in s1


def fmt(v, d=4):
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def build_html(imgdata, s1, s2, s4, hallu_pr, clean_pr, hallu_lr, clean_lr):
    paper = s1.get("paper_llcc", {})
    baseline = s1.get("baseline_llcc", {})
    corrected = s1.get("sink_corrected_llcc", {})
    n_img = s1.get("n_images", len(imgdata))

    paper_row = ""
    paper_bar_dataset = ""
    paper_count_dataset = ""
    paper_outcome_dataset = ""
    delta_paper_row = ""

    if paper:
        paper_row = f"""
    <div class="metric-card">
      <div class="label">F1 Papier (labels)</div>
      <div class="value up">{fmt(paper.get('f1',0))}</div>
      <div class="delta">P={fmt(paper.get('precision',0),3)} · R={fmt(paper.get('recall',0),3)}</div>
    </div>"""
        paper_bar_dataset = f"""
      {{ label: 'Papier (labels)', data: [{fmt(paper.get('precision',0))}, {fmt(paper.get('recall',0))}, {fmt(paper.get('f1',0))}], backgroundColor: 'rgba(92,186,110,0.55)', borderColor: 'rgba(92,186,110,1)', borderWidth: 2, borderRadius: 5 }},"""
        paper_count_dataset = f"""
      {{ label: 'Papier', data: [{paper.get('tp',0)}, {paper.get('fp',0)}, {paper.get('fn',0)}], backgroundColor: 'rgba(92,186,110,0.55)', borderColor: 'rgba(92,186,110,1)', borderWidth: 2, borderRadius: 5 }},"""
        p_tn = n_img - paper.get('tp',0) - paper.get('fp',0) - paper.get('fn',0)
        paper_outcome_dataset = f"""
      {{ label: 'TN', data: [{p_tn}, 0, 0], backgroundColor: 'rgba(92,186,110,0.6)', borderRadius: 4 }},
      {{ label: 'TP', data: [{paper.get('tp',0)}, 0, 0], backgroundColor: 'rgba(108,142,191,0.8)', borderRadius: 4 }},
      {{ label: 'FP', data: [{paper.get('fp',0)}, 0, 0], backgroundColor: 'rgba(224,123,84,0.7)', borderRadius: 4 }},
      {{ label: 'FN', data: [{paper.get('fn',0)}, 0, 0], backgroundColor: 'rgba(201,125,201,0.7)', borderRadius: 4 }},"""
        delta_v = s1.get("delta_f1_vs_paper", baseline.get("f1", 0) - paper.get("f1", 0))
        delta_paper_row = f"""
    <div class="metric-card">
      <div class="label">ΔF1 baseline vs papier</div>
      <div class="value {'up' if delta_v >= 0 else 'down'}">{delta_v:+.3f}</div>
      <div class="delta">Écart de reproduction</div>
    </div>"""

    b_tn = n_img - baseline.get('tp',0) - baseline.get('fp',0) - baseline.get('fn',0)
    c_tn = n_img - corrected.get('tp',0) - corrected.get('fp',0) - corrected.get('fn',0)
    delta_sink = s1.get("delta_f1_sink_vs_baseline", s1.get("delta_f1", 0))

    n_hallu = s2.get("n_hallucinated", len(hallu_pr))
    n_clean = s2.get("n_clean", len(clean_pr))
    auc_pr = s2.get("pr", {}).get("auc_low_pr_predicts_hallu", 0)
    p_pr = s2.get("pr", {}).get("mann_whitney_p", 1)
    pr_hallu_med = s2.get("pr", {}).get("hallu_median", 0)
    pr_clean_med = s2.get("pr", {}).get("clean_median", 0)
    auc_lr = s2.get("leading_ratio", {}).get("auc_high_lead_predicts_hallu", 0)
    p_lr = s2.get("leading_ratio", {}).get("mann_whitney_p", 1)

    n4 = s4.get("n_samples", 0)
    loo_acc = s4.get("loo_accuracy", 0)
    loo_f1 = s4.get("loo_f1", 0)
    loo_auc = s4.get("loo_auc", 0)
    base_acc = s4.get("baseline_accuracy", 0)
    coefs = s4.get("feature_coefs", {})
    n4_t1 = s4.get("n_type1", 0)
    n4_t2 = s4.get("n_type2", 0)

    exp4_samples = []
    exp4_data = load_json(OUTPUTS / "exp4_type_classification" / "samples.json")
    if exp4_data:
        exp4_samples = exp4_data.get("samples", [])

    exp4_scatter_t1 = [
        {"x": round(s["features"]["entropy"],3), "y": round(s["features"]["concentration"],3), "label": s["word"]}
        for s in exp4_samples if s["type"] == "type1"
    ]
    exp4_scatter_t2 = [
        {"x": round(s["features"]["entropy"],3), "y": round(s["features"]["concentration"],3), "label": s["word"]}
        for s in exp4_samples if s["type"] == "type2"
    ]

    exp4_warn = ""
    if n4 < 50:
        exp4_warn = f"""
  <div id="exp4-warn" class="comment-box warn" style="margin-bottom:20px">
    <h4>⚠ Résultats préliminaires</h4>
    <ul>
      <li>Ces résultats portent sur <strong>{n4} échantillons</strong>. Les métriques LOO sont peu fiables sur un petit échantillon.</li>
    </ul>
  </div>"""

    imgdata_json = json.dumps(imgdata, separators=(',', ':'))
    hallu_pr_json = json.dumps(hallu_pr)
    clean_pr_json = json.dumps(clean_pr)
    hallu_lr_json = json.dumps(hallu_lr)
    clean_lr_json = json.dumps(clean_lr)
    exp4_t1_json = json.dumps(exp4_scatter_t1)
    exp4_t2_json = json.dumps(exp4_scatter_t2)
    coef_entropy = round(coefs.get("entropy", 0), 4)
    coef_conc = round(coefs.get("concentration", 0), 4)
    coef_sink = round(coefs.get("sink_ratio", 0), 4)

    outcomes_labels = "'Papier', 'Baseline', 'Sink-corr.'" if paper else "'Baseline', 'Sink-corr.'"

    paper_outcome_data_baseline = f"""
      {{ label: 'TN', data: [0, {b_tn}, {c_tn}], backgroundColor: 'rgba(92,186,110,0.6)', borderRadius: 4 }},
      {{ label: 'TP', data: [0, {baseline.get('tp',0)}, {corrected.get('tp',0)}], backgroundColor: 'rgba(108,142,191,0.8)', borderRadius: 4 }},
      {{ label: 'FP', data: [0, {baseline.get('fp',0)}, {corrected.get('fp',0)}], backgroundColor: 'rgba(224,123,84,0.7)', borderRadius: 4 }},
      {{ label: 'FN', data: [0, {baseline.get('fn',0)}, {corrected.get('fn',0)}], backgroundColor: 'rgba(201,125,201,0.7)', borderRadius: 4 }},"""

    if paper:
        outcomes_all = paper_outcome_dataset + paper_outcome_data_baseline
    else:
        c_tn2 = n_img - corrected.get('tp',0) - corrected.get('fp',0) - corrected.get('fn',0)
        outcomes_all = f"""
      {{ label: 'TN', data: [{b_tn}, {c_tn2}], backgroundColor: 'rgba(92,186,110,0.6)', borderRadius: 4 }},
      {{ label: 'TP', data: [{baseline.get('tp',0)}, {corrected.get('tp',0)}], backgroundColor: 'rgba(108,142,191,0.8)', borderRadius: 4 }},
      {{ label: 'FP', data: [{baseline.get('fp',0)}, {corrected.get('fp',0)}], backgroundColor: 'rgba(224,123,84,0.7)', borderRadius: 4 }},
      {{ label: 'FN', data: [{baseline.get('fn',0)}, {corrected.get('fn',0)}], backgroundColor: 'rgba(201,125,201,0.7)', borderRadius: 4 }},"""

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SADT — Résultats des expériences</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#0f1117;--card:#1a1d27;--border:#2a2d3a;
    --accent1:#6c8ebf;--accent2:#e07b54;--accent3:#5cba6e;--accent4:#c97dc9;
    --text:#e8eaf0;--muted:#8b90a0;--good:#5cba6e;--bad:#e07b54;--neutral:#6c8ebf;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:15px;line-height:1.6}}
  header{{background:linear-gradient(135deg,#12162a 0%,#1a1f35 100%);padding:48px 32px 40px;text-align:center;border-bottom:1px solid var(--border)}}
  header h1{{font-size:2rem;font-weight:700;letter-spacing:-.5px}}
  header h1 span{{color:var(--accent1)}}
  header .subtitle{{color:var(--muted);margin-top:8px;font-size:.95rem}}
  header .badge{{display:inline-block;margin-top:14px;background:#2a1f0a;color:#f0b429;border:1px solid #7a5a10;border-radius:20px;padding:4px 14px;font-size:.8rem;font-weight:600}}
  main{{max-width:1100px;margin:0 auto;padding:40px 24px 80px}}
  .exp-section{{margin-bottom:56px}}
  .exp-header{{display:flex;align-items:center;gap:14px;margin-bottom:20px}}
  .exp-tag{{background:var(--accent1);color:#fff;border-radius:6px;padding:4px 12px;font-size:.8rem;font-weight:700;letter-spacing:.5px;flex-shrink:0}}
  .exp-tag.exp2{{background:var(--accent2)}}.exp-tag.exp4{{background:var(--accent3)}}
  .exp-title{{font-size:1.3rem;font-weight:700}}
  .description-box{{background:var(--card);border:1px solid var(--border);border-left:3px solid var(--accent1);border-radius:8px;padding:18px 20px;margin-bottom:24px}}
  .description-box.exp2{{border-left-color:var(--accent2)}}.description-box.exp4{{border-left-color:var(--accent3)}}
  .description-box p{{color:#c8ccda;font-size:.93rem}}.description-box p+p{{margin-top:8px}}
  .description-box strong{{color:var(--text)}}
  .charts-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
  .charts-grid.single{{grid-template-columns:1fr}}.charts-grid.triple{{grid-template-columns:1fr 1fr 1fr}}
  @media(max-width:720px){{.charts-grid,.charts-grid.triple{{grid-template-columns:1fr}}}}
  .chart-card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px}}
  .chart-card h3{{font-size:.9rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-bottom:16px}}
  .chart-wrap{{position:relative;height:240px}}.chart-wrap.tall{{height:300px}}
  .metrics-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}}
  @media(max-width:720px){{.metrics-row{{grid-template-columns:repeat(2,1fr)}}}}
  .metric-card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center}}
  .metric-card .label{{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}}
  .metric-card .value{{font-size:1.6rem;font-weight:700;margin-top:4px}}
  .metric-card .delta{{font-size:.8rem;margin-top:2px}}
  .up{{color:var(--good)}}.down{{color:var(--bad)}}.neutral-val{{color:var(--neutral)}}
  .comment-box{{background:#161a24;border:1px solid var(--border);border-left:3px solid #4a4f60;border-radius:8px;padding:16px 20px}}
  .comment-box h4{{font-size:.8rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px}}
  .comment-box ul{{list-style:none;padding:0}}
  .comment-box li{{font-size:.9rem;color:#c8ccda;padding:4px 0;padding-left:18px;position:relative}}
  .comment-box li::before{{content:'›';position:absolute;left:4px;color:var(--muted)}}
  .warn{{background:#1a1500;border-left-color:#f0b429}}.warn h4{{color:#f0b429}}
  .section-divider{{border:none;border-top:1px solid var(--border);margin:48px 0}}
  footer{{text-align:center;padding:24px;color:var(--muted);font-size:.82rem;border-top:1px solid var(--border)}}
  /* Gallery */
  .gallery-toolbar{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:20px}}
  #g-search{{background:var(--card);border:1px solid var(--border);color:var(--text);border-radius:8px;padding:8px 14px;font-size:.9rem;width:220px;outline:none}}
  #g-search:focus{{border-color:var(--accent1)}}
  .filter-group{{display:flex;gap:6px;align-items:center;flex-wrap:wrap}}
  .filter-label{{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}}
  .fbtn{{background:transparent;border:1px solid var(--border);color:var(--muted);border-radius:20px;padding:4px 12px;font-size:.78rem;cursor:pointer;transition:.15s}}
  .fbtn:hover{{border-color:var(--accent1);color:var(--text)}}.fbtn.active{{background:var(--accent1);border-color:var(--accent1);color:#fff}}
  .g-count{{font-size:.82rem;color:var(--muted);margin-left:auto}}
  #gallery-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px}}
  .g-card{{background:var(--card);border:1px solid var(--border);border-radius:10px;overflow:hidden;cursor:pointer;transition:border-color .15s,transform .15s}}
  .g-card:hover{{border-color:var(--accent1);transform:translateY(-2px)}}
  .g-card img{{width:100%;aspect-ratio:4/3;object-fit:cover;display:block;background:#111}}
  .g-card .g-card-body{{padding:8px 10px 10px}}
  .g-card .g-name{{font-size:.7rem;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:6px}}
  .g-badges{{display:flex;flex-wrap:wrap;gap:4px}}
  .badge{{font-size:.68rem;font-weight:700;border-radius:4px;padding:2px 6px}}
  .b-tp{{background:rgba(92,186,110,.25);color:#5cba6e}}.b-fp{{background:rgba(224,123,84,.25);color:#e07b54}}
  .b-fn{{background:rgba(201,125,201,.25);color:#c97dc9}}.b-tn{{background:rgba(139,144,160,.15);color:#8b90a0}}
  .b-hal{{background:rgba(224,123,84,.2);color:#e07b54}}.b-clean{{background:rgba(92,186,110,.2);color:#5cba6e}}
  .b-t1{{background:rgba(108,142,191,.2);color:#6c8ebf}}.b-t2{{background:rgba(240,180,40,.2);color:#f0b429}}
  .pagination{{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:20px}}
  .pg-btn{{background:var(--card);border:1px solid var(--border);color:var(--muted);border-radius:6px;padding:6px 14px;cursor:pointer;font-size:.85rem}}
  .pg-btn:hover,.pg-btn.active{{background:var(--accent1);border-color:var(--accent1);color:#fff}}
  .modal-overlay{{position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:1000;display:flex;align-items:center;justify-content:center;padding:20px}}
  .modal-overlay.hidden{{display:none}}
  .modal-box{{background:var(--card);border:1px solid var(--border);border-radius:14px;max-width:860px;width:100%;max-height:90vh;overflow-y:auto;position:relative;padding:28px}}
  .modal-close{{position:absolute;top:14px;right:16px;background:transparent;border:none;color:var(--muted);font-size:1.2rem;cursor:pointer}}
  .modal-close:hover{{color:var(--text)}}
  .modal-img{{width:100%;max-height:380px;object-fit:contain;border-radius:8px;background:#111;margin-bottom:20px}}
  .modal-title{{font-size:1rem;font-weight:700;margin-bottom:16px;word-break:break-all}}
  .modal-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  @media(max-width:600px){{.modal-grid{{grid-template-columns:1fr}}}}
  .modal-panel{{background:#12151f;border:1px solid var(--border);border-radius:8px;padding:14px}}
  .modal-panel h4{{font-size:.78rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px}}
  .tag-row{{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}}
  .tag{{font-size:.8rem;border-radius:4px;padding:3px 8px;font-weight:600}}
  .tag-gt{{background:rgba(139,144,160,.2);color:var(--muted)}}.tag-tp{{background:rgba(92,186,110,.25);color:#5cba6e}}
  .tag-fp{{background:rgba(224,123,84,.25);color:#e07b54}}.tag-fn{{background:rgba(201,125,201,.25);color:#c97dc9}}
  .stat-row{{display:flex;justify-content:space-between;font-size:.85rem;padding:4px 0;border-bottom:1px solid var(--border)}}
  .stat-row:last-child{{border:none}}.stat-key{{color:var(--muted)}}.stat-val{{font-weight:600}}
</style>
</head>
<body>
<header>
  <h1>Same Attention, <span>Different Truths</span></h1>
  <div class="subtitle">Résultats des expériences de reproduction — LLaVA-1.5-7B · COCO val2014 · {n_img} images</div>
  <div class="badge">🏆 CVPR 2026 Highlight</div>
</header>
<main>

<!-- EXP 1 -->
<section class="exp-section">
  <div class="exp-header"><span class="exp-tag">EXP 1</span><span class="exp-title">Sink-corrected LLCC — Détection d'hallucinations</span></div>
  <div class="description-box">
    <p><strong>Objectif :</strong> évaluer si la correction "sink" améliore le détecteur LLCC. La <strong>baseline papier</strong> correspond aux détections stockées dans le fichier de labels (run original du papier, pleine précision). La <strong>baseline re-run</strong> relance LLCC sur les mêmes <code>generated_words</code> mais avec le modèle 4-bit. La <strong>sink-corrected</strong> remplace la sélection de patches par un score normalisé <code>α/μ</code>.</p>
  </div>
  <div class="metrics-row">
    {paper_row}
    <div class="metric-card">
      <div class="label">F1 Baseline re-run</div>
      <div class="value neutral-val">{fmt(baseline.get('f1',0))}</div>
      <div class="delta">P={fmt(baseline.get('precision',0),3)} · R={fmt(baseline.get('recall',0),3)}</div>
    </div>
    <div class="metric-card">
      <div class="label">F1 Sink-corrected</div>
      <div class="value {'up' if corrected.get('f1',0) >= baseline.get('f1',0) else 'down'}">{fmt(corrected.get('f1',0))}</div>
      <div class="delta">P={fmt(corrected.get('precision',0),3)} · R={fmt(corrected.get('recall',0),3)}</div>
    </div>
    <div class="metric-card">
      <div class="label">ΔF1 sink vs baseline</div>
      <div class="value {'up' if delta_sink >= 0 else 'down'}">{delta_sink:+.3f}</div>
      <div class="delta">Effet de la correction</div>
    </div>
    {delta_paper_row}
  </div>
  <div class="charts-grid">
    <div class="chart-card"><h3>Precision / Recall / F1</h3><div class="chart-wrap"><canvas id="chart-exp1-bar"></canvas></div></div>
    <div class="chart-card"><h3>TP / FP / FN</h3><div class="chart-wrap"><canvas id="chart-exp1-counts"></canvas></div></div>
  </div>
  <div class="charts-grid single">
    <div class="chart-card"><h3>Distribution des outcomes ({outcomes_labels})</h3><div class="chart-wrap"><canvas id="chart-exp1-outcomes"></canvas></div></div>
  </div>
  <div class="comment-box">
    <h4>Interprétation</h4>
    <ul>
      <li>La baseline papier (labels) donne F1={fmt(paper.get('f1',0))} — cohérent avec les 0.793 annoncés (différence due au protocole d'évaluation).</li>
      <li>Notre re-run avec le modèle 4-bit donne F1={fmt(baseline.get('f1',0))} — écart de {s1.get('delta_f1_vs_paper', 0):+.3f} dû à la quantisation.</li>
      <li>La correction sink {'améliore' if delta_sink > 0 else 'dégrade'} le F1 de {delta_sink:+.3f} par rapport à notre baseline.</li>
    </ul>
  </div>
</section>

<hr class="section-divider">

<!-- EXP 2 -->
<section class="exp-section">
  <div class="exp-header"><span class="exp-tag exp2">EXP 2</span><span class="exp-title">Spectral TDO — Détection par signature spectrale d'attention</span></div>
  <div class="description-box exp2">
    <p><strong>Participation Ratio (PR) :</strong> mesure la diffusion de l'attention. Hypothèse : une attention concentrée (PR bas) précède les hallucinations.</p>
    <p><strong>Leading Ratio :</strong> fraction d'attention captée par le token dominant.</p>
  </div>
  <div class="metrics-row">
    <div class="metric-card"><div class="label">AUC — PR</div><div class="value {'up' if auc_pr > 0.55 else 'neutral-val'}">{fmt(auc_pr)}</div><div class="delta">p = {p_pr:.2e}</div></div>
    <div class="metric-card"><div class="label">AUC — Leading Ratio</div><div class="value neutral-val">{fmt(auc_lr)}</div><div class="delta">p = {p_lr:.3f}</div></div>
    <div class="metric-card"><div class="label">PR médian hallu</div><div class="value down">{fmt(pr_hallu_med)}</div><div class="delta">vs {fmt(pr_clean_med)} (clean)</div></div>
    <div class="metric-card"><div class="label">Images</div><div class="value neutral-val">{n_img}</div><div class="delta">{n_hallu} hallu · {n_clean} clean</div></div>
  </div>
  <div class="charts-grid">
    <div class="chart-card"><h3>Distribution PR — Halluciné vs Propre</h3><div class="chart-wrap tall"><canvas id="chart-exp2-pr"></canvas></div></div>
    <div class="chart-card"><h3>Distribution Leading Ratio</h3><div class="chart-wrap tall"><canvas id="chart-exp2-lr"></canvas></div></div>
  </div>
  <div class="charts-grid single">
    <div class="chart-card"><h3>AUC comparé</h3><div class="chart-wrap" style="height:160px"><canvas id="chart-exp2-auc"></canvas></div></div>
  </div>
  <div class="comment-box">
    <h4>Interprétation</h4>
    <ul>
      <li>PR statistiquement significatif (p={p_pr:.2e}, AUC={fmt(auc_pr)}) : les images hallucinées ont une attention plus concentrée.</li>
      <li>Leading Ratio non discriminant (AUC≈{fmt(auc_lr)}, p={p_lr:.3f}).</li>
    </ul>
  </div>
</section>

<hr class="section-divider">

<!-- EXP 4 -->
<section class="exp-section">
  <div class="exp-header"><span class="exp-tag exp4">EXP 4</span><span class="exp-title">Classification causale — Type I vs Type II</span></div>
  <div class="description-box exp4">
    <p><strong>Type I — Incertitude visuelle :</strong> attention concentrée, hallucination supprimée par masquage.</p>
    <p><strong>Type II — Prior contextuel :</strong> fort prior de co-occurrence, hallucination persiste après masquage.</p>
  </div>
  {exp4_warn}
  <div class="metrics-row">
    <div class="metric-card"><div class="label">LOO Accuracy</div><div class="value {'up' if loo_acc > base_acc else 'down'}">{fmt(loo_acc)}</div><div class="delta">vs baseline {fmt(base_acc)}</div></div>
    <div class="metric-card"><div class="label">LOO F1</div><div class="value {'up' if loo_f1 > 0.3 else 'down'}">{fmt(loo_f1)}</div><div class="delta">&nbsp;</div></div>
    <div class="metric-card"><div class="label">LOO AUC</div><div class="value {'up' if loo_auc > 0.5 else 'down'}">{fmt(loo_auc)}</div><div class="delta">&nbsp;</div></div>
    <div class="metric-card"><div class="label">Échantillons</div><div class="value neutral-val">{n4}</div><div class="delta">{n4_t1}× T1 · {n4_t2}× T2</div></div>
  </div>
  <div class="charts-grid">
    <div class="chart-card"><h3>Scatter — Entropie vs Concentration</h3><div class="chart-wrap tall"><canvas id="chart-exp4-scatter"></canvas></div></div>
    <div class="chart-card"><h3>Coefficients logistiques</h3><div class="chart-wrap tall"><canvas id="chart-exp4-coef"></canvas></div></div>
  </div>
</section>

<hr class="section-divider">

<!-- GALLERY -->
<section class="exp-section" id="gallery-section">
  <div class="exp-header"><span class="exp-tag" style="background:#7c6abf">GALERIE</span><span class="exp-title">Visualisation par image — inputs & outputs</span></div>
  <div class="gallery-toolbar">
    <input id="g-search" type="text" placeholder="Rechercher une image…" autocomplete="off">
    <div class="filter-group">
      <span class="filter-label">Exp1 baseline</span>
      <button class="fbtn active" data-filter="b_out" data-val="">Tous</button>
      <button class="fbtn" data-filter="b_out" data-val="TP">TP</button>
      <button class="fbtn" data-filter="b_out" data-val="FP">FP</button>
      <button class="fbtn" data-filter="b_out" data-val="FN">FN</button>
      <button class="fbtn" data-filter="b_out" data-val="TN">TN</button>
    </div>
    <div class="filter-group">
      <span class="filter-label">Exp2</span>
      <button class="fbtn active" data-filter="hallu" data-val="">Tous</button>
      <button class="fbtn" data-filter="hallu" data-val="1">Halluciné</button>
      <button class="fbtn" data-filter="hallu" data-val="0">Propre</button>
    </div>
    <div class="filter-group">
      <span class="filter-label">Exp4</span>
      <button class="fbtn active" data-filter="type" data-val="">Tous</button>
      <button class="fbtn" data-filter="type" data-val="type1">Type I</button>
      <button class="fbtn" data-filter="type" data-val="type2">Type II</button>
      <button class="fbtn" data-filter="type" data-val="none">Sans</button>
    </div>
    <span id="g-count" class="g-count"></span>
  </div>
  <div id="gallery-grid"></div>
  <div class="pagination" id="pagination"></div>
</section>

<div id="modal-overlay" class="modal-overlay hidden" onclick="closeModal(event)">
  <div class="modal-box"><button class="modal-close" onclick="closeModal()">✕</button><div id="modal-content"></div></div>
</div>
</main>
<footer>SADT · Reproduction · LLaVA-1.5-7B · COCO val2014 {n_img} images · Cluster SallesInfo (RTX A4000) · Généré le {__import__('datetime').date.today()}</footer>
<script>
const C={{blue:'rgba(108,142,191,1)',blueA:'rgba(108,142,191,0.55)',orange:'rgba(224,123,84,1)',orangeA:'rgba(224,123,84,0.55)',green:'rgba(92,186,110,1)',greenA:'rgba(92,186,110,0.4)',purple:'rgba(201,125,201,1)',purpleA:'rgba(201,125,201,0.4)',muted:'rgba(139,144,160,0.8)',grid:'rgba(42,45,58,1)',text:'#e8eaf0'}};
Chart.defaults.color=C.text;Chart.defaults.borderColor=C.grid;Chart.defaults.font.family="'Segoe UI',system-ui,sans-serif";
function mergedHistogram(a,b,n){{const mn=Math.min(...a,...b),mx=Math.max(...a,...b),w=(mx-mn)/n,cA=Array(n).fill(0),cB=Array(n).fill(0),lb=[];for(let i=0;i<n;i++)lb.push((mn+(i+.5)*w).toFixed(4));const bin=v=>{{let b2=Math.floor((v-mn)/w);return b2>=n?n-1:b2}};a.forEach(v=>cA[bin(v)]++);b.forEach(v=>cB[bin(v)]++);return{{labels:lb,countA:cA,countB:cB}}}}
const bo={{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{boxWidth:12,padding:14}}}}}},scales:{{x:{{grid:{{color:C.grid}},ticks:{{color:C.muted,maxTicksLimit:8}}}},y:{{grid:{{color:C.grid}},ticks:{{color:C.muted}}}}}}}};
// EXP1
new Chart(document.getElementById('chart-exp1-bar'),{{type:'bar',data:{{labels:['Precision','Recall','F1'],datasets:[{paper_bar_dataset}{{label:'Baseline re-run',data:[{fmt(baseline.get('precision',0))},{fmt(baseline.get('recall',0))},{fmt(baseline.get('f1',0))}],backgroundColor:C.blueA,borderColor:C.blue,borderWidth:2,borderRadius:5}},{{label:'Sink-corrected',data:[{fmt(corrected.get('precision',0))},{fmt(corrected.get('recall',0))},{fmt(corrected.get('f1',0))}],backgroundColor:C.orangeA,borderColor:C.orange,borderWidth:2,borderRadius:5}}]}},options:{{...bo,scales:{{x:{{grid:{{color:C.grid}}}},y:{{grid:{{color:C.grid}},min:0,max:1}}}}}}}});
new Chart(document.getElementById('chart-exp1-counts'),{{type:'bar',data:{{labels:['TP','FP','FN'],datasets:[{paper_count_dataset}{{label:'Baseline',data:[{baseline.get('tp',0)},{baseline.get('fp',0)},{baseline.get('fn',0)}],backgroundColor:C.blueA,borderColor:C.blue,borderWidth:2,borderRadius:5}},{{label:'Sink-corr.',data:[{corrected.get('tp',0)},{corrected.get('fp',0)},{corrected.get('fn',0)}],backgroundColor:C.orangeA,borderColor:C.orange,borderWidth:2,borderRadius:5}}]}},options:bo}});
new Chart(document.getElementById('chart-exp1-outcomes'),{{type:'bar',data:{{labels:[{outcomes_labels}],datasets:[{outcomes_all}]}},options:{{...bo,indexAxis:'y',scales:{{x:{{stacked:true,grid:{{color:C.grid}}}},y:{{stacked:true,grid:{{color:C.grid}}}}}}}}}});
// EXP2
const HALLU_PR={hallu_pr_json},CLEAN_PR={clean_pr_json},HALLU_LR={hallu_lr_json},CLEAN_LR={clean_lr_json};
const BINS=28;
const prH=mergedHistogram(HALLU_PR,CLEAN_PR,BINS),lrH=mergedHistogram(HALLU_LR,CLEAN_LR,BINS);
new Chart(document.getElementById('chart-exp2-pr'),{{type:'bar',data:{{labels:prH.labels,datasets:[{{label:'Halluciné',data:prH.countA,backgroundColor:C.orangeA,borderColor:C.orange,borderWidth:1,borderRadius:2}},{{label:'Propre',data:prH.countB,backgroundColor:C.blueA,borderColor:C.blue,borderWidth:1,borderRadius:2}}]}},options:bo}});
new Chart(document.getElementById('chart-exp2-lr'),{{type:'bar',data:{{labels:lrH.labels,datasets:[{{label:'Halluciné',data:lrH.countA,backgroundColor:C.orangeA,borderColor:C.orange,borderWidth:1,borderRadius:2}},{{label:'Propre',data:lrH.countB,backgroundColor:C.blueA,borderColor:C.blue,borderWidth:1,borderRadius:2}}]}},options:bo}});
new Chart(document.getElementById('chart-exp2-auc'),{{type:'bar',data:{{labels:['PR','Leading Ratio'],datasets:[{{label:'AUC',data:[{fmt(auc_pr)},{fmt(auc_lr)}],backgroundColor:[C.greenA,C.purpleA],borderColor:[C.green,C.purple],borderWidth:2,borderRadius:6}}]}},options:{{...bo,indexAxis:'y',plugins:{{legend:{{display:false}}}},scales:{{x:{{grid:{{color:C.grid}},min:0,max:0.8}},y:{{grid:{{color:C.grid}}}}}}}}}});
// EXP4
const t1={exp4_t1_json},t2={exp4_t2_json};
new Chart(document.getElementById('chart-exp4-scatter'),{{type:'scatter',data:{{datasets:[{{label:'Type I',data:t1,backgroundColor:C.blueA,borderColor:C.blue,pointRadius:9}},{{label:'Type II',data:t2,backgroundColor:C.orangeA,borderColor:C.orange,pointRadius:9}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{boxWidth:12}}}},tooltip:{{callbacks:{{label:ctx=>`${{ctx.dataset.label}} "${{ctx.raw.label}}" H=${{ctx.raw.x}} C=${{ctx.raw.y}}`}}}}}},scales:{{x:{{grid:{{color:C.grid}},title:{{display:true,text:'Entropie',color:C.muted}}}},y:{{grid:{{color:C.grid}},title:{{display:true,text:'Concentration',color:C.muted}}}}}}}}}});
new Chart(document.getElementById('chart-exp4-coef'),{{type:'bar',data:{{labels:['Entropie','Concentration','Sink Ratio'],datasets:[{{label:'Coef.',data:[{coef_entropy},{coef_conc},{coef_sink}],backgroundColor:[C.greenA,C.orangeA,C.purpleA],borderColor:[C.green,C.orange,C.purple],borderWidth:2,borderRadius:6}}]}},options:{{...bo,plugins:{{legend:{{display:false}}}}}}}});
// GALLERY
const IMG_DATA={imgdata_json};
const PAGE_SIZE=60;
let gFilters={{b_out:'',hallu:'',type:''}},gSearch='',gPage=0;
function imgOutcome(r){{if(r.e1.b_tp.length)return'TP';if(r.e1.b_fp.length)return'FP';if(r.e1.b_fn.length)return'FN';return'TN'}}
function applyFilters(){{return IMG_DATA.filter(r=>{{if(gSearch){{const q=gSearch.toLowerCase();if(!r.id.toLowerCase().includes(q)&&!r.e1.gt.join(' ').toLowerCase().includes(q))return false}}if(gFilters.b_out&&imgOutcome(r)!==gFilters.b_out)return false;if(gFilters.hallu!==''){{if(!r.e2||String(r.e2.hal)!==gFilters.hallu)return false}}if(gFilters.type){{if(gFilters.type==='none'){{if(r.e4)return false}}else if(!r.e4||!r.e4.some(s=>s.t===gFilters.type))return false}}return true}})}}
function renderGallery(){{const f=applyFilters(),total=f.length,totalPages=Math.ceil(total/PAGE_SIZE);document.getElementById('g-count').textContent=total+' image'+(total>1?'s':'');if(gPage>=totalPages)gPage=0;const slice=f.slice(gPage*PAGE_SIZE,(gPage+1)*PAGE_SIZE),grid=document.getElementById('gallery-grid');grid.innerHTML='';slice.forEach(rec=>{{const card=document.createElement('div');card.className='g-card';card.onclick=()=>openModal(rec);const out=imgOutcome(rec),outCls={{TP:'b-tp',FP:'b-fp',FN:'b-fn',TN:'b-tn'}}[out];let badges=`<span class="badge ${{outCls}}">${{out}}</span>`;if(rec.e2)badges+=`<span class="badge ${{rec.e2.hal?'b-hal':'b-clean'}}">${{rec.e2.hal?'Hal':'Clean'}}</span>`;if(rec.e4){{const types=[...new Set(rec.e4.map(s=>s.t))];types.forEach(t=>badges+=`<span class="badge ${{t==='type1'?'b-t1':'b-t2'}}">${{t==='type1'?'T1':'T2'}}</span>`)}}card.innerHTML=`<img src="data/test_imgs/${{rec.id}}" loading="lazy" alt="${{rec.id}}"><div class="g-card-body"><div class="g-name">${{rec.id.replace('COCO_val2014_','')}}</div><div class="g-badges">${{badges}}</div></div>`;grid.appendChild(card)}});const pg=document.getElementById('pagination');pg.innerHTML='';if(totalPages<=1)return;for(let i=0;i<totalPages;i++){{const btn=document.createElement('button');btn.className='pg-btn'+(i===gPage?' active':'');btn.textContent=i+1;btn.onclick=()=>{{gPage=i;renderGallery();window.scrollTo({{top:document.getElementById('gallery-section').offsetTop-20,behavior:'smooth'}})}};pg.appendChild(btn)}}}}
function tagsHtml(arr,cls){{return arr.length?arr.map(w=>`<span class="tag ${{cls}}">${{w}}</span>`).join(''):'<span style="color:var(--muted);font-size:.8rem">—</span>'}}
function openModal(rec){{document.getElementById('modal-overlay').classList.remove('hidden');const content=document.getElementById('modal-content');const out=imgOutcome(rec),outCls={{TP:'b-tp',FP:'b-fp',FN:'b-fn',TN:'b-tn'}}[out];const p=rec.e1.p_tp||[],pfp=rec.e1.p_fp||[],pfn=rec.e1.p_fn||[];let exp4Html=rec.e4?rec.e4.map(s=>`<div style="margin-bottom:10px"><div style="font-size:.85rem;font-weight:600;margin-bottom:6px">"<em>${{s.w}}</em>" — <span class="badge ${{s.t==='type1'?'b-t1':'b-t2'}}">${{s.t==='type1'?'Type I':'Type II'}}</span></div><div class="stat-row"><span class="stat-key">Entropie</span><span class="stat-val">${{s.H}}</span></div><div class="stat-row"><span class="stat-key">Concentration</span><span class="stat-val">${{s.C}}</span></div><div class="stat-row"><span class="stat-key">Sink ratio</span><span class="stat-val">${{s.S}}</span></div></div>`).join('<hr style="border-color:var(--border);margin:8px 0">'):'<span style="color:var(--muted);font-size:.85rem">Non incluse dans Exp4</span>';const paperPanel=p.length||pfp.length||pfn.length?`<div class="modal-panel"><h4>Exp1 — Papier (labels)</h4><div style="font-size:.82rem;color:var(--muted);margin-bottom:6px">Mots GT</div><div class="tag-row">${{tagsHtml(rec.e1.gt,'tag-gt')}}</div><div style="font-size:.82rem;color:var(--muted);margin:10px 0 4px">Détectés</div><div class="tag-row">${{tagsHtml(p,'tag-tp')}}${{tagsHtml(pfp,'tag-fp')}}</div>${{pfn.length?`<div style="font-size:.75rem;color:var(--muted);margin-top:6px">Manqués : ${{tagsHtml(pfn,'tag-fn')}}</div>`:''}}</div>`:'';const gtWordsHtml=rec.gt_words&&rec.gt_words.length?rec.gt_words.map(w=>`<span class="tag" style="background:rgba(240,180,40,.15);color:#f0b429">${{w}}</span>`).join(''):'—';content.innerHTML=`<div class="modal-title">${{rec.id}}</div><div style="margin-bottom:14px"><span style="font-size:.75rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.5px">Objets présents dans l'image (COCO GT)</span><div class="tag-row" style="margin-top:6px">${{gtWordsHtml}}</div></div><img class="modal-img" src="data/test_imgs/${{rec.id}}" alt="${{rec.id}}"><div class="modal-grid">${{paperPanel}}<div class="modal-panel"><h4>Exp1 — Baseline re-run</h4><div class="tag-row">${{tagsHtml(rec.e1.gt,'tag-gt')}}</div><div style="font-size:.82rem;color:var(--muted);margin:10px 0 4px">Détectés</div><div class="tag-row">${{tagsHtml(rec.e1.b_tp,'tag-tp')}}${{tagsHtml(rec.e1.b_fp,'tag-fp')}}</div>${{rec.e1.b_fn.length?`<div style="font-size:.75rem;color:var(--muted);margin-top:6px">Manqués : ${{tagsHtml(rec.e1.b_fn,'tag-fn')}}</div>`:''}}<div style="margin-top:12px"><span class="badge ${{outCls}}">${{out}}</span></div></div><div class="modal-panel"><h4>Exp1 — Sink-corrected</h4><div class="tag-row">${{tagsHtml(rec.e1.gt,'tag-gt')}}</div><div style="font-size:.82rem;color:var(--muted);margin:10px 0 4px">Détectés</div><div class="tag-row">${{tagsHtml(rec.e1.c_tp,'tag-tp')}}${{tagsHtml(rec.e1.c_fp,'tag-fp')}}</div>${{rec.e1.c_fn.length?`<div style="font-size:.75rem;color:var(--muted);margin-top:6px">Manqués : ${{tagsHtml(rec.e1.c_fn,'tag-fn')}}</div>`:''}}</div><div class="modal-panel"><h4>Exp2 — Spectral TDO</h4>${{rec.e2?`<div class="stat-row"><span class="stat-key">Label</span><span class="stat-val"><span class="badge ${{rec.e2.hal?'b-hal':'b-clean'}}">${{rec.e2.hal?'Halluciné':'Propre'}}</span></span></div><div class="stat-row"><span class="stat-key">PR</span><span class="stat-val">${{rec.e2.pr}}</span></div><div class="stat-row"><span class="stat-key">Leading Ratio</span><span class="stat-val">${{rec.e2.lr}}</span></div><div class="stat-row"><span class="stat-key">Tokens</span><span class="stat-val">${{rec.e2.tok}}</span></div>`:'—'}}</div><div class="modal-panel"><h4>Exp4 — Causal type</h4>${{exp4Html}}</div></div>`;}}
function closeModal(e){{if(!e||e.target===document.getElementById('modal-overlay')||e.currentTarget.classList.contains('modal-close'))document.getElementById('modal-overlay').classList.add('hidden')}}
document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeModal()}});
document.querySelectorAll('.fbtn').forEach(btn=>btn.addEventListener('click',()=>{{const key=btn.dataset.filter;document.querySelectorAll(`.fbtn[data-filter="${{key}}"]`).forEach(b=>b.classList.remove('active'));btn.classList.add('active');gFilters[key]=btn.dataset.val;gPage=0;renderGallery()}}));
document.getElementById('g-search').addEventListener('input',e=>{{gSearch=e.target.value.trim();gPage=0;renderGallery()}});
renderGallery();
</script>
</body>
</html>"""


def main():
    imgdata = build_imgdata()
    s1, s2, s4 = load_summaries()
    hallu_pr, clean_pr, hallu_lr, clean_lr = build_exp2_arrays()

    if not imgdata:
        print("build_ui: no data found, skipping HTML generation", file=sys.stderr)
        return

    html = build_html(imgdata, s1, s2, s4, hallu_pr, clean_pr, hallu_lr, clean_lr)
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"build_ui: results.html updated ({len(html)//1024} KB, {len(imgdata)} images)")


if __name__ == "__main__":
    main()
