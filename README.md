# ML-Sherlock 🕵️

Investigate why machine learning models fail in production.

MVP:
CSV → profiling → MLflow dataset/run → baseline model → production comparison → drift detection → diagnosis → HTML report.

MLflow uses a local SQLite database (`sqlite:///mlflow.db`) by default. To use a
different backend, pass `--tracking-uri` to the CLI or `tracking_uri` to
`AutoResearch`.

Install:
`pip install -e .`

Run (copy and edit `sherlock.example.yaml` for your data):
`ml-sherlock run --config sherlock.example.yaml`

The YAML file contains the dataset paths, target, tracking backend, model seed,
drift threshold, and report location. The report includes deterministic research
hypotheses and the next recommended experiment, each linked to measured evidence.

Optional LLM planning is configured under `llm` in the YAML file. Ollama requires
no additional Python package. For OpenAI or an OpenAI-compatible endpoint, install
`pip install -e ".[openai]"` and set the API-key environment variable named by
`llm.api_key_env`; never store the key in YAML.

Set `llm.enabled: true` to activate Ollama. The CLI then logs provider setup,
each planning request, selected action, and any fallback to deterministic planning.

```python
from autoresearch import AutoResearch, LLMConfig

research = AutoResearch(
    target="y", metric="rmse", tracking_uri="http://mlflow-server:5000",
    llm=LLMConfig(provider="ollama", model="qwen3:8b", base_url="http://localhost:11434"),
)
```

The MVP is deterministic first. LLM/provider and autonomous hypothesis loops come next.
