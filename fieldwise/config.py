"""All settings come from environment variables so keys never live in code.

Prices and model IDs are taken from the Meta Model API docs (dev.meta.ai,
read 27 Oct 2026). Re-check them before the final benchmark run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from .env into the environment. Variables already set win.

    No dependency, works the same on Windows. Handles comments, `export`, quotes and
    inline `  # comments` after unquoted values.
    """
    if not path.exists():
        return
    found: dict[str, str] = {}
    for raw in path.read_text("utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        val = val.strip()
        if val[:1] in ("'", '"') and val[:1] in val[1:]:
            val = val[1:val.index(val[0], 1)]
        else:
            val = val.split(" #", 1)[0].split("\t#", 1)[0].strip()
        if key:
            found[key] = val                       # a later line beats an earlier one
    for key, val in found.items():
        os.environ.setdefault(key, val)            # the real environment beats .env


if os.environ.get("FIELDWISE_DOTENV", "1") != "0":
    _load_dotenv(ROOT / ".env")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # --- Muse Glimmer (local, open weights, served by llama.cpp `llama serve`) ---
    glimmer_base_url: str = field(default_factory=lambda: _env("GLIMMER_BASE_URL", "http://127.0.0.1:8080/v1"))
    glimmer_api_key: str | None = field(default_factory=lambda: _env("GLIMMER_API_KEY"))
    glimmer_model: str = field(default_factory=lambda: _env("GLIMMER_MODEL", "muse-glimmer"))
    # Docs: reasoning can't be turned off; "low" spends the fewest thinking tokens.
    glimmer_reasoning_strength: str = field(default_factory=lambda: _env("GLIMMER_REASONING_STRENGTH", "low"))
    # Reasoning tokens count against max_tokens; too small returns empty content.
    glimmer_max_tokens: int = field(default_factory=lambda: _int("GLIMMER_MAX_TOKENS", 4096))
    glimmer_timeout_s: float = field(default_factory=lambda: _float("GLIMMER_TIMEOUT_S", 600))
    glimmer_seed: int = field(default_factory=lambda: _int("GLIMMER_SEED", 7))

    # --- Meta Model API (Muse Spark + SAM 3.1) ---
    meta_api_key: str | None = field(default_factory=lambda: _env("MODEL_API_KEY"))
    meta_base_url: str = field(default_factory=lambda: _env("META_BASE_URL", "https://api.meta.ai/v1"))
    # Standard mode on purpose: Contributor mode lets Meta train on what we send.
    spark_model: str = field(default_factory=lambda: _env("SPARK_MODEL", "muse-spark-1.3"))
    spark_reasoning_effort: str | None = field(default_factory=lambda: _env("SPARK_REASONING_EFFORT"))
    spark_timeout_s: float = field(default_factory=lambda: _float("SPARK_TIMEOUT_S", 120))
    # Account guard for the app: after this many content-filter refusals, stop calling Spark until the app
    # restarts (repeated refusals got the key restricted on 29 Oct). 0 = off; the benchmark uses its own limit.
    spark_max_refusals: int = field(default_factory=lambda: _int("FIELDWISE_SPARK_MAX_REFUSALS", 2))
    sam_model: str = field(default_factory=lambda: _env("SAM_MODEL", "sam-3.1"))
    sam_timeout_s: float = field(default_factory=lambda: _float("SAM_TIMEOUT_S", 90))

    # Pricing (USD), Standard mode, from the docs' pricing page.
    spark_usd_per_m_input: float = field(default_factory=lambda: _float("SPARK_USD_PER_M_INPUT", 1.25))
    spark_usd_per_m_cached: float = field(default_factory=lambda: _float("SPARK_USD_PER_M_CACHED", 0.15))
    spark_usd_per_m_output: float = field(default_factory=lambda: _float("SPARK_USD_PER_M_OUTPUT", 4.25))
    sam_usd_per_image: float = field(default_factory=lambda: _float("SAM_USD_PER_IMAGE", 2.50 / 1000))
    # Phrase SAM 3.1 uses to find the leaf (the denominator of "% damaged"); {crop} becomes maize/cassava.
    # Chosen with bench/sam_phrases.py on dev photos.
    sam_leaf_phrase: str = field(default_factory=lambda: _env("FIELDWISE_SAM_LEAF_PHRASE", "leaf"))
    # How the damaged area inside the leaf is found: "colour" (default; see fieldwise/lesions.py) or "sam"
    # (SAM 3.1 text prompt per disease, which failed validation on 30 Oct). Cut-offs come from bench/lesion_tune.py.
    lesion_method: str = field(default_factory=lambda: _env("FIELDWISE_LESION_METHOD", "colour"))
    lesion_hue_min: float = field(default_factory=lambda: _float("FIELDWISE_LESION_HUE_MIN", 85.0))
    lesion_sat_min: float = field(default_factory=lambda: _float("FIELDWISE_LESION_SAT_MIN", 0.30))
    # Show "% of the leaf damaged" only if it passed the test bar (median error <= 15 points on unused test photos);
    # FIELDWISE_SEVERITY_ERROR_PTS is that measured median error, shown next to the number.
    severity_show_percent: bool = field(default_factory=lambda: _env("FIELDWISE_SEVERITY_PERCENT", "0") == "1")
    severity_error_pts: float | None = field(default_factory=lambda: (
        float(v) if (v := _env("FIELDWISE_SEVERITY_ERROR_PTS", "").strip()) else None))

    # --- Routing ---
    # Placeholder until tuned on the dev split; never tune on test.
    qa_threshold: float = field(default_factory=lambda: _float("FIELDWISE_QA_THRESHOLD", 0.70))
    dx_threshold: float = field(default_factory=lambda: _float("FIELDWISE_DX_THRESHOLD", 0.70))
    retrieval_k: int = field(default_factory=lambda: _int("FIELDWISE_RETRIEVAL_K", 6))
    # Below this query-term coverage the guides are treated as not covering the question.
    min_coverage: float = field(default_factory=lambda: _float("FIELDWISE_MIN_COVERAGE", 0.34))

    # --- Demo controls ---
    force_offline: bool = field(default_factory=lambda: _env("FIELDWISE_FORCE_OFFLINE", "0") == "1")
    cache_only: bool = field(default_factory=lambda: _env("FIELDWISE_CACHE_ONLY", "0") == "1")
    # Demo only, for a hub too small to run Glimmer 30B live: show Glimmer's real outputs recorded by the
    # benchmark run (bench/results/*_local.jsonl) for those questions and photos. The UI labels every
    # replayed answer. The benchmark refuses to run with this on.
    glimmer_replay: bool = field(default_factory=lambda: _env("FIELDWISE_GLIMMER_REPLAY", "0") == "1")

    # --- Paths ---
    data_dir: Path = field(default_factory=lambda: Path(_env("FIELDWISE_DATA_DIR", str(ROOT / "data"))))
    # Downloaded benchmark datasets (CCMT, PlantSeg). Gitignored; never redistributed.
    datasets_dir: Path = field(default_factory=lambda: Path(_env("FIELDWISE_DATASETS_DIR", str(ROOT / "datasets"))))

    @property
    def guides_dir(self) -> Path:
        return self.data_dir / "guides"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index" / "guides_index.json"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def queue_db(self) -> Path:
        return self.data_dir / "queue.sqlite3"

    @property
    def outbox_dir(self) -> Path:
        return self.data_dir / "outbox"


settings = Settings()
