"""Якісніший графік prequential-кривих для розділу 4 (потік 1)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .datasets import make_drift_stream
from .policy_adapt import (prequential_accuracy, run_adaptive,
                           run_incremental, run_static)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main(seed=301, train_n=1500):
    X, y, drifts = make_drift_stream(n=12000, seed=seed)
    c_static = run_static(X, y, train_n, seed)
    c_adapt, retrains = run_adaptive(X, y, train_n, seed)
    c_incr = run_incremental(X, y, train_n, seed)
    w = 500
    xs = np.arange(train_n, len(y))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(xs, prequential_accuracy(c_static, w), color="#888888",
            label="Статична політика", lw=1.4)
    ax.plot(xs, prequential_accuracy(c_incr, w), color="#2c7fb8",
            label="Інкрементальна логістична регресія", lw=1.4, alpha=0.9)
    ax.plot(xs, prequential_accuracy(c_adapt, w), color="#d95f02",
            label="Еволюційна адаптація", lw=1.6)
    for k, (pos, typ) in enumerate(drifts):
        ax.axvline(pos, color="black", ls="--", lw=0.9,
                   label="Точки дрейфу" if k == 0 else None)
    ax.plot(retrains, [0.32] * len(retrains), marker="^", ls="none",
            color="#1b9e77", markersize=7, label="Перенавчання ГА")
    ax.set_xlabel("Позиція у потоці подій")
    ax.set_ylabel("Prequential-точність (вікно 500)")
    ax.set_ylim(0.30, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "e4_prequential.png", dpi=150)
    print("saved plot, means:",
          f"static={c_static.mean():.3f}",
          f"adaptive={c_adapt.mean():.3f}",
          f"sgd={c_incr.mean():.3f}")


if __name__ == "__main__":
    main()
