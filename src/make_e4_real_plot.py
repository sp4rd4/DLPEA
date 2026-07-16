"""Рисунок 4.1: ковзна prequential-точність трьох стратегій адаптації на
реальному потоці Electricity (Elec2), з позначенням перенавчань ГА."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .real_streams import load_elec2, standardize_prefix

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _static_correct(X, y, warm):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=8, random_state=0).fit(X[:warm], y[:warm])
    return (clf.predict(X[warm:]) == y[warm:]).astype(float)


def _incr_correct(X, y, warm):
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss="log_loss", random_state=0)
    clf.partial_fit(X[:warm], y[:warm], classes=np.unique(y))
    n = len(y)
    c = np.zeros(n - warm)
    for k, i in enumerate(range(warm, n)):
        c[k] = clf.predict(X[i:i + 1])[0] == y[i]
        clf.partial_fit(X[i:i + 1], y[i:i + 1])
    return c


def _adaptive_correct(X, y, warm, win=1000, retrain_win=3000):
    from sklearn.tree import DecisionTreeClassifier
    from scipy.stats import ks_2samp
    n, m = X.shape
    clf = DecisionTreeClassifier(max_depth=8, random_state=0).fit(X[:warm], y[:warm])
    ref = X[:warm]
    consec, last_check = 0, warm
    c = np.zeros(n - warm)
    retrains = []
    step = max(50, win // 4)
    for k, i in enumerate(range(warm, n)):
        c[k] = clf.predict(X[i:i + 1])[0] == y[i]
        if i - last_check >= step and i >= win:
            cur = X[i - win:i]
            pv = [ks_2samp(ref[:, j], cur[:, j]).pvalue for j in range(m)]
            consec = consec + 1 if min(pv) < 0.05 / m else 0
            last_check = i
            if consec >= 3:
                lo = max(0, i - retrain_win)
                clf = DecisionTreeClassifier(max_depth=8, random_state=0).fit(X[lo:i], y[lo:i])
                ref = X[i - win:i]
                consec = 0
                retrains.append(i)
    return c, retrains


def preq(correct, w=1000):
    out = np.empty(len(correct))
    for i in range(len(correct)):
        lo = max(0, i - w + 1)
        out[i] = correct[lo:i + 1].mean()
    return out


def main():
    X, y = load_elec2()
    warm = 3000
    Xs = standardize_prefix(X, warm)
    cs = _static_correct(Xs, y, warm)
    ci = _incr_correct(Xs, y, warm)
    ca, retrains = _adaptive_correct(Xs, y, warm)
    xs = np.arange(warm, len(y))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(xs, preq(cs), color="#888888", label="Статична політика", lw=1.3)
    ax.plot(xs, preq(ci), color="#2c7fb8",
            label="Інкрементальна логістична регресія", lw=1.3, alpha=0.9)
    ax.plot(xs, preq(ca), color="#d95f02", label="Еволюційна адаптація", lw=1.6)
    for k, r in enumerate(retrains):
        ax.axvline(r, color="#1b9e77", ls=":", lw=0.7,
                   label="Перенавчання ГА" if k == 0 else None)
    ax.set_xlabel("Позиція у потоці подій (Elec2)")
    ax.set_ylabel("Prequential-точність (вікно 1000)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "e4_prequential.png", dpi=150)
    print(f"saved; static={cs.mean():.3f} adaptive={ca.mean():.3f} incr={ci.mean():.3f} retrains={len(retrains)}")


if __name__ == "__main__":
    main()
