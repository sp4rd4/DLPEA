"""Завантаження реальних потоків для експериментів з дрейфом концепції
та адаптацією політик: INSECTS (Souza et al., 2020) та Electricity (Elec2).

Позиції дрейфу для INSECTS узято з першоджерела (таблиця 2):
Souza V. M. A., Reis D. M., Maletzke A. G., Batista G. E. Challenges in
Benchmarking Stream Learning Algorithms with Real-world Data. 2020.
"""
from __future__ import annotations

import numpy as np

# Позиції зміни концепції (індекси зразків) для збалансованих варіантів INSECTS.
# Ключі відповідають назвам варіантів у бібліотеці river.
INSECTS_CHANGE_POINTS = {
    "abrupt_balanced": [14352, 19500, 33240, 38682, 39510],
    "gradual_balanced": [14028],
    "incremental_reoccurring_balanced": [26568, 53364],
    "incremental_abrupt_balanced": [26568, 53364],
}


def load_insects(variant="abrupt_balanced", max_n=None):
    """Повертає (X, y, change_points). X стандартизовано за навчальним префіксом."""
    from river import datasets
    ds = datasets.Insects(variant=variant)
    xs, ys = [], []
    for i, (x, y) in enumerate(ds):
        if max_n and i >= max_n:
            break
        xs.append(list(x.values()))
        ys.append(y)
    X = np.asarray(xs, dtype=float)
    # мітки класів -> цілі
    classes = sorted(set(ys))
    cmap = {c: k for k, c in enumerate(classes)}
    y = np.asarray([cmap[c] for c in ys], dtype=int)
    cps = INSECTS_CHANGE_POINTS[variant]
    if max_n:
        cps = [c for c in cps if c < max_n]
    return X, y, cps


def load_elec2(max_n=None):
    """Electricity (Elec2): 8 ознак, бінарна мітка UP/DOWN. Повертає (X, y)."""
    from river import datasets
    ds = datasets.Elec2()
    xs, ys = [], []
    for i, (x, y) in enumerate(ds):
        if max_n and i >= max_n:
            break
        xs.append(list(x.values()))
        ys.append(int(y))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int)


def standardize_prefix(X, warm):
    """Стандартизація за статистикою навчального префікса (без витоку майбутнього)."""
    mu = X[:warm].mean(axis=0)
    sd = X[:warm].std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd
