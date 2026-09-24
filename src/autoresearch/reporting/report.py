"""Self-contained, evidence-led investigation reports."""

import base64
import io
import json
import math
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, FileSystemLoader, select_autoescape
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


MODEL_NAMES = {
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "baseline": "Başlangıç modeli",
}
ACTION_NAMES = {
    "retrain_recent_data": "Güncel veriyle eğitim",
    "model_search": "Model karşılaştırması",
    "drop_drifted_features": "Driftli özellikleri çıkarma",
}


def number(value, digits=2):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "Veri yok"
    return f"{value:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def experiment_chart(experiments, metric, dark=False):
    """Plot measured candidates and the accepted lineage on the selection split."""
    points = [e for e in experiments if isinstance(e.get("candidate_metrics", {}).get(metric), (int, float))
              and math.isfinite(e["candidate_metrics"][metric])]
    if not points:
        return None
    ink, muted, line = ("#eceef1", "#b5bdc9", "#3c414b") if dark else ("#22262e", "#606874", "#e2e5ea")
    fig = Figure(figsize=(10.5, 3.4), dpi=160, facecolor="none", layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set_facecolor("none")
    baseline = points[0].get("baseline_metrics", {}).get(metric)
    current = baseline
    xs, accepted = [], []
    if isinstance(baseline, (int, float)) and math.isfinite(baseline):
        xs.append(0)
        accepted.append(baseline)
        ax.axhline(baseline, color=muted, linestyle="--", linewidth=1, label="Başlangıç modeli")
    for item in points:
        score = item["candidate_metrics"][metric]
        good = item.get("status") == "validated"
        ax.scatter(item["iteration"], score, marker="o" if good else "x", s=38,
                   color="#258564" if good else "#c96364", zorder=4)
        if good:
            current = score
        if isinstance(current, (int, float)):
            xs.append(item["iteration"])
            accepted.append(current)
    ax.step(xs, accepted, where="post", color="#668be4" if dark else "#365dc5", linewidth=2.2,
            label="Korunan model")
    ticks = [p["iteration"] for p in points]
    ax.set_xticks(ticks[::max(1, len(ticks) // 15)])
    ax.set_xlabel("Deney", color=muted, fontsize=10)
    ax.set_ylabel(metric.upper(), color=muted, fontsize=10)
    ax.tick_params(colors=muted, labelsize=9, length=0, pad=8)
    ax.grid(axis="y", color=line, linewidth=.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, loc="best", labelcolor=ink, fontsize=9)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class ReportBuilder:
    def build(self, output_path, target, baseline, production, drift, diagnosis, research):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        decision = research.get("decision") or {}
        experiments = research.get("experiments", [])
        initial_candidates = research.get("initial_candidates", [])
        metric = decision.get("selection_metric", "rmse")
        final_before = decision.get("final_baseline_metrics", {})
        final_after = decision.get("final_candidate_metrics", {})
        accepted = [e for e in experiments if e.get("status") == "validated"]
        selected = next((e for e in experiments if e.get("iteration") == decision.get("iteration")), None)
        model = MODEL_NAMES.get(decision.get("model"), decision.get("model", "Henüz seçilmedi"))
        deployment = decision.get("deployment_status", "requires_independent_evaluation")
        titles = {
            "review_candidate": f"{model} ile kontrollü geçiş değerlendirilebilir.",
            "keep_baseline": "Başlangıç modelini koruyun.",
            "requires_independent_evaluation": "Bağımsız doğrulama tamamlanmalı.",
        }
        profiles = {p["name"]: p for p in decision.get("reference_profile", {}).get("features", [])}
        prod_profiles = {p["name"]: p for p in decision.get("production_profile", {}).get("features", [])}
        drift_rows = [dict(item, reference=profiles.get(item["feature"], {}),
                           production=prod_profiles.get(item["feature"], {}),
                           retained=item["feature"] in decision.get("used_features", [])) for item in drift]
        data = decision.get("training_data", {})
        adaptation_rows = experiments[0].get("adaptation_rows") if experiments else None
        selection_rows = experiments[0].get("holdout_rows") if experiments else None
        downloads = []
        for suffix, label in ((".json", "Karar JSON"), (".joblib", "Eğitilmiş model")):
            companion = path.with_suffix(suffix)
            if companion.is_file():
                downloads.append({"url": quote(companion.name), "label": label})
        env = Environment(loader=FileSystemLoader(Path(__file__).parent), autoescape=select_autoescape(["html"]))
        env.filters["number"] = number
        env.filters["model_name"] = lambda value: MODEL_NAMES.get(value, value)
        env.filters["action_name"] = lambda value: ACTION_NAMES.get(value, value)
        html = env.get_template("report.html").render(
            target=target, baseline=baseline, production=production, diagnosis=diagnosis,
            research=research, decision=decision, experiments=experiments, selected=selected,
            accepted=accepted, initial_candidates=initial_candidates,
            model=model, metric=metric, deployment=deployment,
            title=titles.get(deployment, titles["requires_independent_evaluation"]),
            final_before=final_before, final_after=final_after, drift_rows=drift_rows,
            drift_count=sum(bool(d["drift"]) for d in drift), data=data,
            adaptation_rows=adaptation_rows, selection_rows=selection_rows,
            downloads=downloads, chart=experiment_chart(experiments, metric),
            dark_chart=experiment_chart(experiments, metric, dark=True),
        )
        path.write_text(html, encoding="utf-8")
        snapshot = dict(target=target, baseline=baseline, production=production, drift=drift,
                        diagnosis=diagnosis, research=research)
        path.with_suffix(".report-data.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return str(path.resolve())
