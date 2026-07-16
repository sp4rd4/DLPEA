"""Стандартні синтетичні потоки з коваріантним дрейфом для оцінки детектора
зміни концепції. Використано генератор RandomRBF з бібліотеки river
(Montiel et al., 2021): різні набори центроїдів дають різні розподіли ознак,
тож зміна центроїдів у заданій точці породжує контрольований зсув P(x) з
точно відомою позицією. Абруптний варіант: миттєва заміна концепту;
поступовий: імовірнісне змішування двох концептів у вікні переходу.
"""
from __future__ import annotations

import itertools
import numpy as np


def _segment(seed_model, n, n_feat, n_centroids, seed_sample):
    import river.datasets.synth as synth
    s = synth.RandomRBF(seed_model=seed_model, seed_sample=seed_sample,
                        n_classes=2, n_features=n_feat, n_centroids=n_centroids)
    xs, ys = [], []
    for x, y in itertools.islice(s, n):
        xs.append(list(x.values()))
        ys.append(int(y))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int)


def make_abrupt(n_concepts=4, seg_len=3000, n_feat=10, n_centroids=20, seed=0):
    """Потік з n_concepts концептами; абруптні зсуви на межах сегментів."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 10_000, n_concepts)
    Xs, ys, cps = [], [], []
    for k, sm in enumerate(seeds):
        X, y = _segment(int(sm), seg_len, n_feat, n_centroids, seed_sample=seed + 1)
        if k > 0:
            cps.append(k * seg_len)
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys), cps


def make_gradual(n_concepts=4, seg_len=3000, width=800, n_feat=10,
                 n_centroids=20, seed=0):
    """Поступові переходи: у вікні width зразки з ймовірністю змішуються між
    сусідніми концептами (лінійне наростання частки нового концепту)."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 10_000, n_concepts)
    segs = [_segment(int(sm), seg_len, n_feat, n_centroids, seed_sample=seed + 1)
            for sm in seeds]
    Xs, ys, cps = [], [], []
    mix_rng = np.random.default_rng(seed + 7)
    for k in range(n_concepts):
        X, y = segs[k]
        Xs.append(X.copy())
        ys.append(y.copy())
        cps.append(k * seg_len) if k > 0 else None
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    # накласти перехідне змішування навколо кожної точки
    for k in range(1, n_concepts):
        pos = k * seg_len
        prev, cur = segs[k - 1], segs[k]
        for i in range(pos - width // 2, pos + width // 2):
            if i < 0 or i >= len(X):
                continue
            p_new = (i - (pos - width // 2)) / width
            if mix_rng.random() > p_new:
                j = mix_rng.integers(0, len(prev[0]))
                X[i] = prev[0][j]
                y[i] = prev[1][j]
    return X, y, [c for c in [k * seg_len for k in range(1, n_concepts)]]
