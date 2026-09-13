from pathlib import Path
import html


class ReportBuilder:
    def build(self, output_path, target, baseline, production, drift, diagnosis, research):
        def number(value):
            return "—" if value is None else f"{value:.4f}"

        status = diagnosis["status"]
        performance = "".join(
            f"<tr><td>{html.escape(metric.upper())}</td><td>{number(before)}</td>"
            f"<td>{number(production.get(metric))}</td></tr>" for metric, before in baseline.items()
        )
        drift_rows = "".join(
            f"<tr><td>{html.escape(item['feature'])}</td><td>{item['test']}</td><td>{item['p_value']:.4g}</td>"
            f"<td><span class='badge {'danger' if item['drift'] else 'ok'}'>{'Drift' if item['drift'] else 'Stable'}</span></td></tr>"
            for item in drift
        ) or "<tr><td colspan='4'>No comparable features.</td></tr>"
        hypotheses = "".join(
            f"<article class='hypothesis'><span class='badge warn'>{html.escape(item['confidence'])} confidence</span>"
            f"<h3>{html.escape(item['title'])}</h3><p>{html.escape(item['claim'])}</p>"
            f"<p class='muted'><strong>Next test:</strong> {html.escape(item['experiment']['goal'])}</p></article>"
            for item in research["hypotheses"]
        )
        experiment_section = ""
        if research.get("experiments"):
            experiment = max(research["experiments"], key=lambda item: item["improvement_pct"])
            candidate_rows = "".join(
                f"<tr><td>{html.escape(item['model'])}</td><td>{number(item['metrics'][experiment['selection_metric']])}</td></tr>"
                for item in experiment["candidates"]
            )
            importance_rows = "".join(
                f"<tr><td>{html.escape(item['feature'])}</td><td>{item['importance']:.4f}</td></tr>"
                for item in experiment.get("feature_importance", [])[:10]
            ) or "<tr><td colspan='2'>Feature importance is unavailable for this model.</td></tr>"
            history_rows = "".join(
                f"<tr><td>{item['iteration']}</td><td>{html.escape(item['action'])}</td><td>{html.escape(item['status'])}</td><td>{item.get('improvement_pct', 0):+.2f}%</td></tr>"
                for item in research.get("history", []) if item["action"] != "stop"
            )
            experiment_section = f"""<section class='panel'><div class='section-head'><div><p class='eyebrow'>HYPOTHESIS VALIDATION</p><h2>Best of {research.get('iterations_completed', len(research['experiments']))} scheduled experiments</h2></div><span class='badge {'ok' if experiment['status'] == 'validated' else 'danger'}'>{experiment['status'].upper()}</span></div>
            <div class='metrics'><div><span>Holdout improvement</span><strong>{experiment['improvement_pct']:+.2f}%</strong></div><div><span>Recommended model</span><strong>{html.escape(experiment['recommended_model'])}</strong></div><div><span>Unseen holdout rows</span><strong>{experiment['holdout_rows']}</strong></div></div>
            <p class='muted'>{html.escape(experiment['success_criterion'])}</p><table><thead><tr><th>Candidate</th><th>{html.escape(experiment['selection_metric'].upper())} on holdout</th></tr></thead><tbody><tr><td>baseline</td><td>{number(experiment['baseline_metrics'][experiment['selection_metric']])}</td></tr>{candidate_rows}</tbody></table>
            <h3>Top feature importance</h3><p class='muted'>Model-based importance for the winning candidate; this is not a causal explanation.</p><table><thead><tr><th>Transformed feature</th><th>Importance</th></tr></thead><tbody>{importance_rows}</tbody></table>
            <h3>Experiment timeline</h3><table><thead><tr><th>Iteration</th><th>Action</th><th>Status</th><th>Improvement</th></tr></thead><tbody>{history_rows}</tbody></table></section>"""

        body = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>ML-Sherlock Investigation</title><style>
        :root{{--ink:#172033;--muted:#63708a;--paper:#f5f7fb;--line:#e4e9f2;--blue:#4d74ff;--red:#c43e58;--green:#188a68;--amber:#a35c00}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:36px 24px 64px}}.hero{{background:linear-gradient(125deg,#17264c,#293f7a);border-radius:22px;padding:38px;color:#fff;box-shadow:0 16px 35px #253d761e}}.eyebrow{{font-size:11px;font-weight:800;letter-spacing:.12em;margin:0 0 6px;color:#91adff}}h1{{font-size:32px;line-height:1.15;margin:0 0 10px}}h2{{font-size:20px;margin:0}}h3{{font-size:16px;margin:8px 0}}.hero p{{margin:0;color:#d7e1ff}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0}}.card,.panel{{background:#fff;border:1px solid var(--line);border-radius:16px;padding:22px;box-shadow:0 5px 15px #1b30500a}}.card span,.metrics span{{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}.card strong,.metrics strong{{display:block;font-size:24px;margin-top:3px}}.panel{{margin-top:18px}}.section-head{{display:flex;justify-content:space-between;gap:12px;align-items:start;margin-bottom:14px}}table{{border-collapse:collapse;width:100%}}th{{text-align:left;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}}td,th{{padding:11px 8px;border-bottom:1px solid var(--line)}}.badge{{display:inline-block;padding:4px 9px;border-radius:999px;font-size:11px;font-weight:800;letter-spacing:.04em}}.ok{{background:#ddf6ed;color:var(--green)}}.danger{{background:#fae5e9;color:var(--red)}}.warn{{background:#fff0d7;color:var(--amber)}}.hypothesis{{border-left:3px solid var(--blue);padding:4px 0 4px 16px;margin:18px 0}}.hypothesis p{{margin:7px 0}}.muted{{color:var(--muted)}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;background:#f7f9fd;padding:15px;border-radius:12px;margin:12px 0 16px}}@media(max-width:700px){{main{{padding:18px 14px}}.hero{{padding:28px 22px}}.grid,.metrics{{grid-template-columns:1fr}}}}</style></head><body><main>
        <header class='hero'><p class='eyebrow'>ML-SHERLOCK / INVESTIGATION REPORT</p><h1>Production model health</h1><p>Target: <strong>{html.escape(target)}</strong> · Evidence-first analysis and experiment validation</p></header>
        <section class='grid'><div class='card'><span>System status</span><strong>{status.upper()}</strong></div><div class='card'><span>Degraded metrics</span><strong>{len(diagnosis['performance_degradation'])}</strong></div><div class='card'><span>Drifted features</span><strong>{len(diagnosis['drifted_features'])}</strong></div></section>
        <section class='panel'><div class='section-head'><div><p class='eyebrow'>DECISION</p><h2>{html.escape(research['recommendation'])}</h2></div><span class='badge {'danger' if status == 'degraded' else 'ok'}'>{status.upper()}</span></div><p class='muted'>{html.escape(diagnosis['summary'])}</p></section>
        <section class='panel'><p class='eyebrow'>MODEL PERFORMANCE</p><h2>Baseline versus production</h2><table><thead><tr><th>Metric</th><th>Baseline validation</th><th>Production</th></tr></thead><tbody>{performance}</tbody></table></section>
        <section class='panel'><p class='eyebrow'>DRIFT EVIDENCE</p><h2>Feature distribution comparison</h2><table><thead><tr><th>Feature</th><th>Test</th><th>p-value</th><th>Result</th></tr></thead><tbody>{drift_rows}</tbody></table></section>
        <section class='panel'><p class='eyebrow'>RESEARCH HYPOTHESES</p><h2>What the evidence suggests</h2>{hypotheses}</section>{experiment_section}</main></body></html>"""
        path = Path(output_path)
        path.write_text(body, encoding="utf-8")
        return str(path.resolve())
