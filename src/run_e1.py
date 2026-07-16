"""E1: класифікація документів — ГА-правила проти базлайнів.

5-кратна стратифікована крос-валідація на двох датасетах (Synthetic PII,
20 Newsgroups binary). Метрики якості + інтерпретованість + час (для E5).
Додатково: чутливість до коефіцієнта регуляризації lambda.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .datasets import build_features, load_20ng_binary, make_pii_corpus
from .ga_classifier import GARuleClassifier

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

BASELINES = {
    "RandomForest": lambda s: RandomForestClassifier(n_estimators=200, random_state=s),
    "GradientBoosting": lambda s: GradientBoostingClassifier(random_state=s),
    "LinearSVM": lambda s: LinearSVC(random_state=s),
    "LogisticRegression": lambda s: LogisticRegression(max_iter=2000, random_state=s),
    "NaiveBayes": lambda s: MultinomialNB(),
}


def eval_fold(model, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(Xte)
    pred_t = time.perf_counter() - t0
    return {
        "f1": f1_score(yte, pred, zero_division=0),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "fit_time": fit_t,
        "predict_time_per_1k": pred_t / len(yte) * 1000,
    }


def run_dataset(name, texts, y, seed=42, ga_params=None):
    ga_params = ga_params or {}
    print(f"== {name}: {len(texts)} docs, pos={y.mean():.2%}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    per_model = {m: [] for m in list(BASELINES) + ["GA-rules"]}
    interp, ga_rules_dump = [], []
    texts = np.array(texts, dtype=object)
    for fold, (tr, te) in enumerate(skf.split(texts, y)):
        Xtr, Xte, feat_names = build_features(list(texts[tr]), list(texts[te]))
        # NB вимагає невід'ємних ознак — зсуваємо метадані до [0..]
        shift = np.minimum(Xtr.min(axis=0), 0)
        for mname, factory in BASELINES.items():
            m = factory(seed + fold)
            if mname == "NaiveBayes":
                per_model[mname].append(
                    eval_fold(m, Xtr - shift, y[tr], Xte - shift, y[te]))
            else:
                per_model[mname].append(eval_fold(m, Xtr, y[tr], Xte, y[te]))
        ga = GARuleClassifier(seed=seed + fold, **ga_params)
        res = eval_fold(ga, Xtr, y[tr], Xte, y[te])
        per_model["GA-rules"].append(res)
        interp.append(ga.interpretability(Xte))
        ga_rules_dump.append(ga.describe(feat_names))
        print(f"  fold {fold}: GA f1={res['f1']:.3f} "
              f"rules={interp[-1]['n_rules']} | RF f1={per_model['RandomForest'][-1]['f1']:.3f}")
    summary = {}
    for mname, folds in per_model.items():
        summary[mname] = {k: {"mean": float(np.mean([f[k] for f in folds])),
                              "std": float(np.std([f[k] for f in folds]))}
                          for k in folds[0]}
    summary["GA-rules"]["interpretability"] = {
        k: {"mean": float(np.mean([i[k] for i in interp])),
            "std": float(np.std([i[k] for i in interp]))}
        for k in interp[0]}
    summary["GA-rules"]["example_rules"] = ga_rules_dump[-1]
    return summary


def lambda_sensitivity(texts, y, seed=42, ga_params=None):
    """Сітка lambda: компроміс точність-складність (3 фолди для швидкості)."""
    from sklearn.model_selection import StratifiedKFold
    ga_params = dict(ga_params or {})
    ga_params.pop("lam", None)
    out = {}
    texts = np.array(texts, dtype=object)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for lam in [0.0, 0.02, 0.05, 0.1, 0.2]:
        f1s, nrules = [], []
        for fold, (tr, te) in enumerate(skf.split(texts, y)):
            Xtr, Xte, _ = build_features(list(texts[tr]), list(texts[te]))
            ga = GARuleClassifier(lam=lam, seed=seed + fold,
                                  **ga_params).fit(Xtr, y[tr])
            f1s.append(f1_score(y[te], ga.predict(Xte), zero_division=0))
            nrules.append(len(ga.rules_))
        out[str(lam)] = {"f1_mean": float(np.mean(f1s)),
                         "f1_std": float(np.std(f1s)),
                         "rules_mean": float(np.mean(nrules))}
        print(f"  lambda={lam}: F1={np.mean(f1s):.3f} rules={np.mean(nrules):.1f}")
    return out


GA_20NG = {"max_rules": 12, "max_conds": 3, "pop_size": 80, "generations": 120,
           "lam": 0.02, "patience": 25}


def main():
    results = {}
    pii_texts, pii_y = make_pii_corpus(n=4000, seed=42)
    results["synthetic_pii"] = run_dataset(
        "Synthetic PII", pii_texts, pii_y,
        ga_params={"lam": 0.05, "pop_size": 100, "generations": 100,
                   "patience": 25})
    ng_texts, ng_y = load_20ng_binary(seed=42, max_docs=4000)
    results["20ng_binary"] = run_dataset("20NG binary", ng_texts, ng_y,
                                         ga_params=GA_20NG)
    print("== lambda sensitivity (20NG)")
    results["lambda_sensitivity_20ng"] = lambda_sensitivity(
        ng_texts, ng_y, ga_params={k: v for k, v in GA_20NG.items()})
    with open(RESULTS / "e1_classification.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved -> e1_classification.json")


if __name__ == "__main__":
    main()
