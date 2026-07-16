"""E4: еволюційна адаптація політик — prequential-порівняння:
статична ГА-політика / адаптивна (перенавчання за сигналом дрейфу) /
інкрементальний SGD. 5 потоків; графік кривих для одного потоку.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .datasets import make_drift_stream
from .policy_adapt import (prequential_accuracy, run_adaptive,
                           run_incremental, run_static)

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def main(n_streams=5, train_n=1500):
    rows = {"static": [], "adaptive": [], "incremental": []}
    curves_saved = False
    for s in range(n_streams):
        seed = 300 + s
        X, y, drifts = make_drift_stream(n=12000, seed=seed)
        c_static = run_static(X, y, train_n, seed)
        c_adapt, retrains = run_adaptive(X, y, train_n, seed)
        c_incr = run_incremental(X, y, train_n, seed)
        rows["static"].append(float(c_static.mean()))
        rows["adaptive"].append(float(c_adapt.mean()))
        rows["incremental"].append(float(c_incr.mean()))
        print(f"stream {s}: static={c_static.mean():.3f} "
              f"adaptive={c_adapt.mean():.3f} (retrains at {retrains}) "
              f"sgd={c_incr.mean():.3f}")
        if not curves_saved:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            w = 500
            fig, ax = plt.subplots(figsize=(10, 4.5))
            xs = np.arange(train_n, len(y))
            ax.plot(xs, prequential_accuracy(c_static, w), label="Статична політика", lw=1.2)
            ax.plot(xs, prequential_accuracy(c_adapt, w), label="Еволюційна адаптація", lw=1.2)
            ax.plot(xs, prequential_accuracy(c_incr, w), label="Інкрементальний SGD", lw=1.2, alpha=0.8)
            for pos, typ in drifts:
                ax.axvline(pos, color="gray", ls="--", lw=0.8)
            for r in retrains:
                ax.axvline(r, color="green", ls=":", lw=0.8)
            ax.set_xlabel("Позиція у потоці")
            ax.set_ylabel("Prequential accuracy (вікно 500)")
            ax.legend(loc="lower left")
            ax.set_ylim(0.4, 1.0)
            fig.tight_layout()
            fig.savefig(RESULTS / "e4_prequential.png", dpi=150)
            curves_saved = True
    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                   "runs": v} for k, v in rows.items()}
    summary["improvement_vs_static_pct"] = float(
        (np.mean(rows["adaptive"]) - np.mean(rows["static"]))
        / np.mean(rows["static"]) * 100)
    with open(RESULTS / "e4_policy.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved -> e4_policy.json")


if __name__ == "__main__":
    main()
