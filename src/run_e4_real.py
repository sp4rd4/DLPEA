"""E4 (реальні дані): еволюційна адаптація політики класифікації на реальному
нестаціонарному потоці Electricity (Elec2). Порівнюються три стратегії за
prequential-точністю (схема «перевірка-потім-навчання»):

  static      - модель навчено один раз на початковому вікні, далі не оновлюється;
  adaptive    - запропонована схема: детектор дрейфу KS+підтвердження керує
                перенавчанням набору правил на свіжому вікні (режим дослідження);
  incremental - інкрементальна логістична регресія (пасивна адаптація).

Elec2 має природний добовий і сезонний дрейф попиту/цін, тож придатний для
оцінки утримання якості в нестаціонарному режимі.
"""
from __future__ import annotations

import json
import numpy as np

from .real_streams import load_elec2, standardize_prefix
from .drift import ks_confirm_detector


def prequential_static(X, y, warm):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=8, random_state=0)
    clf.fit(X[:warm], y[:warm])
    correct = clf.predict(X[warm:]) == y[warm:]
    return float(np.mean(correct))


def prequential_incremental(X, y, warm):
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss="log_loss", random_state=0)
    clf.partial_fit(X[:warm], y[:warm], classes=np.unique(y))
    n = len(y)
    correct = np.zeros(n - warm, dtype=bool)
    for k, i in enumerate(range(warm, n)):
        correct[k] = clf.predict(X[i:i + 1])[0] == y[i]
        clf.partial_fit(X[i:i + 1], y[i:i + 1])
    return float(np.mean(correct))


def prequential_adaptive(X, y, warm, win, retrain_win=2000):
    """Детектор дрейфу керує перенавчанням дерева рішень (сурогат набору правил).
    Після підтвердженої тривоги модель перенавчається на останньому вікні."""
    from sklearn.tree import DecisionTreeClassifier
    from scipy.stats import ks_2samp
    n, m = X.shape
    clf = DecisionTreeClassifier(max_depth=8, random_state=0)
    clf.fit(X[:warm], y[:warm])
    ref = X[:warm]
    consec = 0
    correct = np.zeros(n - warm, dtype=bool)
    step = max(50, win // 4)
    last_check = warm
    n_retrains = 0
    for k, i in enumerate(range(warm, n)):
        correct[k] = clf.predict(X[i:i + 1])[0] == y[i]
        if i - last_check >= step and i >= win:
            cur = X[i - win:i]
            pvals = [ks_2samp(ref[:, j], cur[:, j]).pvalue for j in range(m)]
            signal = min(pvals) < 0.05 / m
            consec = consec + 1 if signal else 0
            last_check = i
            if consec >= 3:
                lo = max(0, i - retrain_win)
                clf = DecisionTreeClassifier(max_depth=8, random_state=0)
                clf.fit(X[lo:i], y[lo:i])
                ref = X[i - win:i]
                consec = 0
                n_retrains += 1
    return float(np.mean(correct)), n_retrains


def _eval_stream(name, X, y, warm, win, retrain_win=3000):
    Xs = standardize_prefix(X, warm)
    acc_static = prequential_static(Xs, y, warm)
    acc_inc = prequential_incremental(Xs, y, warm)
    acc_adp, n_ret = prequential_adaptive(Xs, y, warm, win, retrain_win)
    improvement = 100.0 * (acc_adp - acc_static) / acc_static
    print(f"{name} n={len(y)}")
    print(f"  static      = {acc_static:.3f}")
    print(f"  adaptive    = {acc_adp:.3f}  (перенавчань: {n_ret})")
    print(f"  incremental = {acc_inc:.3f}")
    print(f"  приріст adaptive vs static: {improvement:+.1f}%")
    return {
        "dataset": name,
        "n_samples": int(len(y)),
        "warm": warm,
        "static": {"prequential_acc": acc_static},
        "adaptive": {"prequential_acc": acc_adp, "n_retrains": n_ret},
        "incremental": {"prequential_acc": acc_inc},
        "improvement_vs_static_pct": improvement,
    }


def run():
    out = {}
    Xe, ye = load_elec2()
    out["elec2"] = _eval_stream("Elec2 (Electricity)", Xe, ye, warm=3000, win=1000)
    from .real_streams import load_insects
    Xi, yi, _ = load_insects("abrupt_balanced")
    out["insects_abrupt"] = _eval_stream("INSECTS-Abr (abrupt)", Xi, yi,
                                         warm=3000, win=1500, retrain_win=4000)
    with open("results/e4_real.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run()
