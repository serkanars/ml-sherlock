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
EVIDENCE_NAMES = {
    "performance_degradation": "Performans düşüşü",
    "feature_drift": "Özellik drifti",
    "target_drift": "Hedef drifti",
    "prediction_drift": "Tahmin drifti",
    "residual_drift": "Artık hata drifti",
    "feature_error_relationship": "Özellik-hata ilişkisi",
    "segment_degradation": "Segment bozulması",
}
HYPOTHESIS_NAMES = {
    "covariate_shift": "Kovaryat değişimi",
    "segment_specific_degradation": "Segmente özgü bozulma",
    "target_relationship_shift": "Hedef ilişkisi değişimi",
    "unstable_feature": "Kararsız özellik",
    "model_family_robustness": "Model ailesi dayanıklılığı",
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
    def build(self, output_path, target, baseline, production, drift, diagnosis, research,
              *, evidence=None, ranked_evidence=None, segment_analysis=None,
              residual_analysis=None, feature_error_analysis=None):
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
        evidence_rows = _collect_evidence(
            evidence,
            drift,
            residual_analysis=residual_analysis,
            feature_error_analysis=feature_error_analysis,
        )
        evidence_by_id = {item["id"]: item for item in evidence_rows}
        ranked_rows = _ranked_rows(ranked_evidence, evidence_rows)
        top_evidence = ranked_rows[:8]
        feature_drift = [item for item in evidence_rows if item["type"] == "feature_drift"]
        distribution_drift = [
            item for item in evidence_rows
            if item["type"] in {"target_drift", "prediction_drift", "residual_drift"}
        ]
        feature_error = [
            item for item in evidence_rows if item["type"] == "feature_error_relationship"
        ]
        degraded_segments = sorted(
            (item for item in evidence_rows if item["type"] == "segment_degradation"),
            key=lambda item: (-_numeric(item.get("value")), item["id"]),
        )[:20]
        hypothesis_rows = _hypothesis_rows(
            research.get("hypotheses", []), evidence_by_id, experiments, metric
        )
        downloads = []
        for suffix, label in ((".json", "Karar JSON"), (".joblib", "Eğitilmiş model")):
            companion = path.with_suffix(suffix)
            if companion.is_file():
                downloads.append({"url": quote(companion.name), "label": label})
        env = Environment(loader=FileSystemLoader(Path(__file__).parent), autoescape=select_autoescape(["html"]))
        env.filters["number"] = number
        env.filters["model_name"] = lambda value: MODEL_NAMES.get(value, value)
        env.filters["action_name"] = lambda value: ACTION_NAMES.get(value, value)
        env.filters["evidence_name"] = lambda value: EVIDENCE_NAMES.get(value, value)
        env.filters["hypothesis_name"] = lambda value: HYPOTHESIS_NAMES.get(value, value)
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
            evidence_rows=evidence_rows, top_evidence=top_evidence,
            feature_drift=feature_drift, distribution_drift=distribution_drift,
            feature_error=feature_error, degraded_segments=degraded_segments,
            hypothesis_rows=hypothesis_rows, diagnoses=diagnosis.get("patterns", []),
            segment_analysis=segment_analysis,
        )
        path.write_text(html, encoding="utf-8")
        snapshot = dict(target=target, baseline=baseline, production=production, drift=drift,
                        evidence=evidence_rows, diagnosis=diagnosis, research=research)
        path.with_suffix(".report-data.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return str(path.resolve())


def _collect_evidence(evidence, drift, *, residual_analysis=None, feature_error_analysis=None):
    collected = {}
    for item in evidence or []:
        serialized = item.to_dict() if hasattr(item, "to_dict") else dict(item)
        collected[serialized["id"]] = _evidence_row(serialized)
    for item in drift:
        evidence_id = f"feature-drift-{item['feature']}"
        if any(row.get("feature") == item["feature"] and row["type"] == "feature_drift"
               for row in collected.values()):
            continue
        synthetic = {
            "id": evidence_id,
            "type": "feature_drift",
            "metric": "distribution_effect_size",
            "value": item.get("effect_size", item.get("statistic")),
            "feature": item["feature"],
            "segment": None,
            "severity": item.get("severity", "info"),
            "direction": None,
            "metadata": item,
        }
        collected[evidence_id] = _evidence_row(synthetic)
    for analysis in (residual_analysis, feature_error_analysis):
        if not analysis:
            continue
        items = analysis.get("evidence", []) if isinstance(analysis, dict) else []
        if items is None:
            continue
        if not isinstance(items, list):
            items = [items]
        for item in items:
            serialized = item.to_dict() if hasattr(item, "to_dict") else dict(item)
            collected.setdefault(serialized["id"], _evidence_row(serialized))
    return list(collected.values())


def _evidence_row(item):
    metadata = item.get("metadata") or {}
    subject = item.get("segment") or item.get("feature") or EVIDENCE_NAMES.get(
        item["type"], item["type"]
    )
    return {
        **item,
        "metadata": metadata,
        "subject": subject,
        "active": metadata.get("drift") is not False and (
            metadata.get("drift") is True or item.get("severity") != "info"
        ),
        "summary": _evidence_summary(item, subject),
    }


def _evidence_summary(item, subject):
    evidence_type = item["type"]
    if evidence_type == "feature_drift":
        return f"{subject} dağılımında ölçülmüş değişim."
    if evidence_type == "target_drift":
        return f"{subject} hedef dağılımı referans dönemden farklı."
    if evidence_type == "prediction_drift":
        return "Model tahminlerinin dağılımı referans dönemden farklı."
    if evidence_type == "residual_drift":
        return "Model artık hatalarının dağılımında değişim ölçüldü."
    if evidence_type == "feature_error_relationship":
        return f"{subject}, mutlak model hatasıyla tahmine dayalı ilişki gösteriyor."
    if evidence_type == "segment_degradation":
        return f"Model performansındaki bozulma {subject} segmentinde yoğunlaşıyor."
    return f"{subject} için ölçülmüş araştırma bulgusu."


def _ranked_rows(ranked_evidence, evidence_rows):
    by_id = {item["id"]: item for item in evidence_rows}
    ordered = []
    for item in ranked_evidence or []:
        evidence_id = item.id if hasattr(item, "id") else item.get("id")
        if evidence_id in by_id and by_id[evidence_id] not in ordered:
            ordered.append(by_id[evidence_id])
    severity = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
    remaining = [item for item in evidence_rows if item not in ordered]
    remaining.sort(
        key=lambda item: (
            -severity.get(item.get("severity"), 0),
            -abs(_numeric(item.get("value"))),
            item["id"],
        )
    )
    return [*ordered, *remaining]


def _hypothesis_rows(hypotheses, evidence_by_id, experiments, metric):
    rows = []
    for hypothesis in hypotheses:
        item = dict(hypothesis)
        supporting = [
            evidence_by_id[evidence_id]
            for evidence_id in item.get("evidence_ids", [])
            if evidence_id in evidence_by_id
        ]
        matches = [
            experiment for experiment in experiments
            if experiment.get("hypothesis_id") == item.get("id")
        ]
        if not matches and item.get("recommended_experiment"):
            matches = [
                experiment for experiment in experiments
                if experiment.get("action") == item["recommended_experiment"]
            ]
        item["supporting_evidence"] = supporting
        item["experiment_results"] = [
            {
                "iteration": experiment.get("iteration"),
                "status": experiment.get("status"),
                "action": experiment.get("action"),
                "improvement_pct": experiment.get("improvement_pct"),
                "candidate_model": experiment.get("candidate_model"),
                "candidate_metric": experiment.get("candidate_metrics", {}).get(metric),
            }
            for experiment in matches
        ]
        rows.append(item)
    return rows


def _numeric(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0
