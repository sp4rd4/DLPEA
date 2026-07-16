"""Еволюційна адаптація політик за умов дрейфу (п. 3.2): prequential-оцінювання
адаптивної стратегії (перенавчання ГА-правил за сигналом детектора дрейфу)
проти статичної політики та інкрементального базлайна.
"""
from __future__ import annotations

import numpy as np
from .ga_classifier import GARuleClassifier
from .drift import ks_confirm_detector


def prequential_accuracy(correct, window=500):
    """Ковзна prequential-точність (test-then-train)."""
    acc = np.zeros(len(correct))
    c = np.asarray(correct, dtype=float)
    cs = np.cumsum(c)
    for i in range(len(c)):
        lo = max(0, i - window + 1)
        total = cs[i] - (cs[lo - 1] if lo > 0 else 0.0)
        acc[i] = total / (i - lo + 1)
    return acc


def _fit_ga(X, y, seed, lam=0.05):
    ga = GARuleClassifier(max_rules=6, max_conds=3, pop_size=50,
                          generations=50, lam=lam, patience=12, seed=seed)
    ga.fit(X, y)
    return ga


def run_static(X, y, train_n, seed=0):
    ga = _fit_ga(X[:train_n], y[:train_n], seed)
    correct = []
    for i in range(train_n, len(y)):
        pred = ga.predict(X[i:i + 1])[0]
        correct.append(int(pred == y[i]))
    return np.array(correct)


def run_adaptive(X, y, train_n, seed=0, refit_window=300,
                 det_window=250, det_step=50, explore_refits=(0, 300, 600)):
    """Адаптивна політика з режимами експлуатації/дослідження (п. 3.2):
    у режимі експлуатації детектор (KS + підтвердження) стежить за дрейфом;
    після тривоги система переходить у режим дослідження: серія перенавчань
    ГА на свіжих вікнах (одразу, +300, +600 зразків), референс детектора
    скидається на вікно, що спрацювало."""
    ga = _fit_ga(X[:train_n], y[:train_n], seed)
    correct, retrain_points = [], []
    from scipy.stats import ks_2samp
    ref = X[:train_n][-det_window:]
    consec, cooldown_until = 0, 0
    pending = []                                   # заплановані перенавчання
    for i in range(train_n, len(y)):
        pred = ga.predict(X[i:i + 1])[0]
        correct.append(int(pred == y[i]))
        if pending and i >= pending[0]:
            pending.pop(0)
            lo = max(0, i - refit_window)
            ga = _fit_ga(X[lo:i], y[lo:i], seed + i)
            retrain_points.append(i)
            if not pending:        # кінець режиму дослідження
                ref = X[i - det_window:i]
        if (i - train_n) % det_step == 0 and i - train_n >= det_window \
                and i >= cooldown_until and not pending:
            cur = X[i - det_window:i]
            pv = [ks_2samp(ref[:, j], cur[:, j]).pvalue
                  for j in range(X.shape[1])]
            signal = min(pv) < 0.02 / X.shape[1]
            consec = consec + 1 if signal else 0
            if consec >= 2:
                # режим дослідження: серія перенавчань на свіжих даних
                pending = [i + d for d in explore_refits]
                ref = cur                          # чистий пост-дрейфовий референс
                consec = 0
                cooldown_until = i + explore_refits[-1] + 400
    return np.array(correct), retrain_points


def run_incremental(X, y, train_n, seed=0):
    """Інкрементальний базлайн: SGD-логістична регресія, partial_fit."""
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss="log_loss", random_state=seed)
    clf.fit(X[:train_n], y[:train_n])
    correct = []
    for i in range(train_n, len(y)):
        pred = clf.predict(X[i:i + 1])[0]
        correct.append(int(pred == y[i]))
        clf.partial_fit(X[i:i + 1], y[i:i + 1])
    return np.array(correct)
