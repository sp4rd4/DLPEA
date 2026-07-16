"""Поведінкове профілювання (метод п. 3.3): адаптивний профіль з
експоненційним забуванням + зважена агрегація z-відхилень; конфігурація
(маска, ваги, поріг, швидкість забування) оптимізується ГА.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def profile_scores(X, rho, warmup=10):
    """Онлайн-профіль: EWMA середнє/дисперсія на користувача;
    повертає z+ (тільки перевищення) для кожного користувача-дня."""
    n_users, n_days, F = X.shape
    mu = X[:, :warmup, :].mean(axis=1)
    var = X[:, :warmup, :].var(axis=1) + 1e-6
    Z = np.zeros_like(X)
    for d in range(warmup, n_days):
        x = X[:, d, :]
        z = (x - mu) / np.sqrt(var)
        Z[:, d, :] = np.maximum(0.0, z)
        # оновлення профілю (після оцінки дня)
        mu = rho * mu + (1 - rho) * x
        var = rho * var + (1 - rho) * (x - mu) ** 2 + 1e-9
    return Z


def detector_alarms(Z, mask, weights, tau):
    w = mask * weights
    denom = w.sum() + 1e-9
    score = np.tensordot(Z, w, axes=([2], [0])) / denom
    return (score > tau).astype(int), score


class GABehaviorOptimizer:
    """ГА над конфігурацією детектора: хромосома = [маска(F), ваги(F), tau, rho]."""

    def __init__(self, n_feat, pop_size=40, generations=50, seed=0,
                 fp_penalty=0.15, patience=12):
        self.F = n_feat
        self.pop_size, self.generations = pop_size, generations
        self.rng = np.random.default_rng(seed)
        self.fp_penalty, self.patience = fp_penalty, patience

    def _random(self):
        return {
            "mask": (self.rng.random(self.F) < 0.6).astype(float),
            "w": self.rng.uniform(0.1, 1.0, self.F),
            "tau": float(self.rng.uniform(1.0, 4.0)),
            "rho": float(self.rng.uniform(0.85, 0.99)),
        }

    def _fitness(self, c, X, y, warmup):
        Z = self._z_cache.setdefault(round(c["rho"], 3),
                                     profile_scores(X, c["rho"], warmup))
        alarms, _ = detector_alarms(Z, c["mask"], c["w"], c["tau"])
        a, t = alarms[:, warmup:].ravel(), y[:, warmup:].ravel()
        f1 = f1_score(t, a, zero_division=0)
        fpr = ((a == 1) & (t == 0)).sum() / max((t == 0).sum(), 1)
        return f1 - self.fp_penalty * fpr

    def _mutate(self, c):
        c = {"mask": c["mask"].copy(), "w": c["w"].copy(),
             "tau": c["tau"], "rho": c["rho"]}
        r = self.rng.random()
        if r < 0.3:
            i = self.rng.integers(self.F)
            c["mask"][i] = 1.0 - c["mask"][i]
        elif r < 0.6:
            i = self.rng.integers(self.F)
            c["w"][i] = float(np.clip(c["w"][i] + self.rng.normal(0, 0.2),
                                      0.01, 1.5))
        elif r < 0.85:
            c["tau"] = float(np.clip(c["tau"] + self.rng.normal(0, 0.3),
                                     0.5, 6.0))
        else:
            c["rho"] = float(np.clip(c["rho"] + self.rng.normal(0, 0.02),
                                     0.80, 0.995))
        return c

    def _crossover(self, a, b):
        pick = self.rng.random(self.F) < 0.5
        return {"mask": np.where(pick, a["mask"], b["mask"]),
                "w": np.where(pick, a["w"], b["w"]),
                "tau": a["tau"] if self.rng.random() < 0.5 else b["tau"],
                "rho": a["rho"] if self.rng.random() < 0.5 else b["rho"]}

    def fit(self, X, y, warmup=10):
        self._z_cache = {}
        pop = [self._random() for _ in range(self.pop_size)]
        fits = np.array([self._fitness(c, X, y, warmup) for c in pop])
        best_i = int(np.argmax(fits))
        best, best_fit, stall = pop[best_i], fits[best_i], 0
        for _ in range(self.generations):
            order = np.argsort(fits)[::-1]
            new = [pop[i] for i in order[:2]]
            while len(new) < self.pop_size:
                i1, i2 = self.rng.integers(0, len(pop), 2)
                j1, j2 = self.rng.integers(0, len(pop), 2)
                p1 = pop[i1] if fits[i1] > fits[i2] else pop[i2]
                p2 = pop[j1] if fits[j1] > fits[j2] else pop[j2]
                child = self._crossover(p1, p2)
                if self.rng.random() < 0.5:
                    child = self._mutate(child)
                new.append(child)
            pop = new
            fits = np.array([self._fitness(c, X, y, warmup) for c in pop])
            gen_i = int(np.argmax(fits))
            if fits[gen_i] > best_fit + 1e-6:
                best, best_fit, stall = pop[gen_i], fits[gen_i], 0
            else:
                stall += 1
            if stall >= self.patience:
                break
        self.best_ = best
        return self


def evaluate_split(X, y, cfg, split_day, warmup=10):
    """Метрики на тестовій половині періоду (після split_day)."""
    Z = profile_scores(X, cfg["rho"], warmup)
    alarms, score = detector_alarms(Z, cfg["mask"], cfg["w"], cfg["tau"])
    a = alarms[:, split_day:].ravel()
    t = y[:, split_day:].ravel()
    return {
        "f1": f1_score(t, a, zero_division=0),
        "precision": precision_score(t, a, zero_division=0),
        "recall": recall_score(t, a, zero_division=0),
        "fnr": 1.0 - recall_score(t, a, zero_division=0),
        "fpr": float(((a == 1) & (t == 0)).sum() / max((t == 0).sum(), 1)),
    }


def baseline_metrics(X, y, split_day, kind, seed=0):
    """Базлайни без персональних профілів: Isolation Forest / LOF / OCSVM
    на сирих векторах користувач-день (типова практика «глобальної» моделі)."""
    n_users, n_days, F = X.shape
    Xtr = X[:, :split_day, :].reshape(-1, F)
    Xte = X[:, split_day:, :].reshape(-1, F)
    yte = y[:, split_day:].ravel()
    contamination = max(float(y[:, :split_day].mean()), 0.005)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    if kind == "isolation_forest":
        from sklearn.ensemble import IsolationForest
        m = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=seed).fit(Xtr_s)
        pred = (m.predict(Xte_s) == -1).astype(int)
    elif kind == "lof":
        from sklearn.neighbors import LocalOutlierFactor
        m = LocalOutlierFactor(n_neighbors=30, novelty=True,
                               contamination=contamination).fit(Xtr_s)
        pred = (m.predict(Xte_s) == -1).astype(int)
    elif kind == "ocsvm":
        from sklearn.svm import OneClassSVM
        m = OneClassSVM(nu=contamination, gamma="scale").fit(Xtr_s)
        pred = (m.predict(Xte_s) == -1).astype(int)
    else:
        raise ValueError(kind)
    return {
        "f1": f1_score(yte, pred, zero_division=0),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "fnr": 1.0 - recall_score(yte, pred, zero_division=0),
        "fpr": float(((pred == 1) & (yte == 0)).sum() / max((yte == 0).sum(), 1)),
    }
