"""Multi-seed aggregation + reporting, ported/generalized from dfl-ablation/{spo_vs_sfge,sweep_lr}.py.

Used by every experiment to turn per-seed regret lists into {mean,std,values} records, paired
significance tests, and JSON + Markdown artifacts under ``exp_results/``.
"""
from __future__ import annotations
import json
import math
import os

import numpy as np

try:
    from scipy.stats import wilcoxon as _wilcoxon
except Exception:  # scipy always present in the project .venv, but degrade gracefully
    _wilcoxon = None


def summarize(values):
    """List of per-seed metrics -> {mean, std, n, values} (NaNs ignored)."""
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))],
                     dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "values": list(values)}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size),
            "values": [None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
                       for v in values]}


def wilcoxon_pair(a, b, alternative="less"):
    """Paired Wilcoxon signed-rank p-value for H1: a < b (alternative='less'). None if undefined.

    Returns None when scipy is missing, n<2 paired finite samples, or all differences are zero.
    """
    if _wilcoxon is None:
        return None
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2 or np.allclose(a, b):
        return None
    try:
        return float(_wilcoxon(a, b, alternative=alternative, zero_method="wilcox").pvalue)
    except Exception:
        return None


def fmt_mean_std(rec, places=4):
    if rec is None or rec.get("n", 0) == 0:
        return "n/a"
    return f"{rec['mean']:.{places}f}±{rec['std']:.{places}f}"


def md_table(headers, rows):
    """headers: list[str]; rows: list[list[str]] -> GitHub-flavored markdown table string."""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def write_md(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path
