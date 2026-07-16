"""E2 (реальні дані): виявлення інсайдерів на CERT Insider Threat r4.2.
Оцінювання на рівні користувача (ідентифікація інсайдера), як прийнято для
цього набору: кожному користувачеві призначаємо ризик-оцінку за тестовий
період, після чого ранжуємо користувачів. Запропонований метод (індивідуальний
адаптивний профіль, конфігурацію якого оптимізує ГА) проти глобальних
детекторів аномалій Isolation Forest, LOF, One-Class SVM.

Ключова відмінність: глобальні методи оцінюють, наскільки рядок «користувач-
день» рідкісний для всієї популяції; персональний профіль оцінює відхилення
від власної норми користувача, характерне для інсайдера, чия активність
залишається звичайною на тлі всієї організації.
"""
from __future__ import annotations

import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from .behavior import GABehaviorOptimizer, profile_scores, detector_alarms
from .cert_features import build, FEATURE_COLS


def to_tensor(rows):
    users = sorted(set(r["user"] for r in rows))
    dates = sorted(set(r["date"] for r in rows))
    uidx = {u: i for i, u in enumerate(users)}
    didx = {d: i for i, d in enumerate(dates)}
    U, D, F = len(users), len(dates), len(FEATURE_COLS)
    X = np.zeros((U, D, F), dtype=float)
    y = np.zeros((U, D), dtype=int)
    for r in rows:
        i, j = uidx[r["user"]], didx[r["date"]]
        X[i, j] = [r[c] for c in FEATURE_COLS]
        y[i, j] = r["label"]
    return X, y, users, dates


def user_metrics(user_score, user_label, budgets=(20, 50, 100)):
    """Average precision, recall@budget, precision@budget для ранжування
    користувачів. За екстремального дисбалансу (інсайдери це кілька відсотків)
    саме показники на робочому бюджеті тривог, а не ROC-AUC, відображають
    практичну придатність, тож AUC наводимо лише як довідковий."""
    order = np.argsort(user_score)[::-1]
    n_ins = int(user_label.sum())
    out = {"auc": float(roc_auc_score(user_label, user_score)),
           "ap": float(average_precision_score(user_label, user_score)),
           "n_insiders": n_ins}
    for b in budgets:
        top = order[:b]
        det = int(user_label[top].sum())
        out[f"recall@{b}"] = det / n_ins if n_ins else 0.0
        out[f"precision@{b}"] = det / b
    return out


def profile_user_scores(X, cfg, split, warmup):
    """Ризик-оцінка користувача це пікова денна аномалія його профілю на
    тестовому періоді (найсильніше відхилення від власної норми)."""
    Z = profile_scores(X, cfg["rho"], warmup)
    _, score = detector_alarms(Z, cfg["mask"], cfg["w"], cfg["tau"])
    return score[:, split:].max(axis=1)


def global_user_scores(X, split, kind, seed=0):
    from sklearn.preprocessing import StandardScaler
    U, D, F = X.shape
    Xtr = X[:, :split, :].reshape(-1, F)
    sc = StandardScaler().fit(Xtr)
    Xte = sc.transform(X[:, split:, :].reshape(-1, F)).reshape(U, D - split, F)
    Xtr_s = sc.transform(Xtr)
    if kind == "isolation_forest":
        from sklearn.ensemble import IsolationForest
        m = IsolationForest(n_estimators=300, random_state=seed).fit(Xtr_s)
        day = -m.score_samples(Xte.reshape(-1, F)).reshape(U, D - split)
    elif kind == "lof":
        from sklearn.neighbors import LocalOutlierFactor
        m = LocalOutlierFactor(n_neighbors=30, novelty=True).fit(Xtr_s)
        day = -m.score_samples(Xte.reshape(-1, F)).reshape(U, D - split)
    elif kind == "ocsvm":
        from sklearn.svm import OneClassSVM
        m = OneClassSVM(nu=0.05, gamma="scale").fit(Xtr_s)
        day = -m.score_samples(Xte.reshape(-1, F)).reshape(U, D - split)
    else:
        raise ValueError(kind)
    return day.max(axis=1)


def run(seed=0):
    rows = build()
    X, y, users, dates = to_tensor(rows)
    U, D, F = X.shape
    user_label = (y.sum(axis=1) > 0).astype(int)
    n_ins = int(user_label.sum())
    print(f"CERT r4.2: користувачів={U}, днів={D}, ознак={F}, інсайдерів={n_ins}, "
          f"шкідливих user-days={int(y.sum())}")

    warmup, split = 30, int(D * 0.6)
    opt = GABehaviorOptimizer(F, pop_size=40, generations=50, seed=seed, fp_penalty=0.15)
    opt.fit(X[:, :split, :], y[:, :split], warmup=warmup)
    cfg = opt.best_

    scores = {"GA-profile": profile_user_scores(X, cfg, split, warmup)}
    for kind in ["isolation_forest", "lof", "ocsvm"]:
        scores[kind] = global_user_scores(X, split, kind, seed=seed)

    res = {k: user_metrics(v, user_label) for k, v in scores.items()}
    summary = {
        "dataset": "CERT Insider Threat r4.2",
        "n_users": U, "n_days": D, "n_features": F,
        "n_insiders": n_ins, "n_malicious_userdays": int(y.sum()),
        "split_day": split, "warmup": warmup,
        "active_features": [FEATURE_COLS[i] for i in range(F) if cfg["mask"][i] > 0.5],
        "tau": cfg["tau"], "rho": cfg["rho"],
        "features": FEATURE_COLS, "results": res,
    }
    with open("results/e2_cert.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    names = {"GA-profile": "ГА-профіль", "isolation_forest": "Isolation Forest",
             "lof": "LOF", "ocsvm": "One-Class SVM"}
    print()
    for k in ["GA-profile", "isolation_forest", "lof", "ocsvm"]:
        r = res[k]
        print(f"  {names[k]:18s} AUC={r['auc']:.3f} AP={r['ap']:.3f} "
              f"R@50={r['recall@50']:.3f} R@100={r['recall@100']:.3f} P@50={r['precision@50']:.3f}")
    print("  активні ознаки ГА:", summary["active_features"])


if __name__ == "__main__":
    run()
