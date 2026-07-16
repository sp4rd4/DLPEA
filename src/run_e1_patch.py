"""E1 (перевага керованості): оновлення політики експертом без перенавчання.

Сценарій: у документообігу з'являється новий тип критичного ідентифікатора
(міжнародний номер рахунку IBAN), для якого ще немає розмічених прикладів.
Навчальний корпус не містить IBAN-позитивів; у тестовому періоді третина
конфіденційних документів визначається саме IBAN.

Порівнюються реакції:
  - ГА-правила: експерт дописує ОДНЕ правило "IF pat_iban > 0 THEN confidential"
    (секунди, нуль розмічених прикладів), решта набору правил не змінюється;
  - RF / SVM / LR: пряме втручання неможливе (модель непрозора), варіант
    відповіді лише донавчання, для якого потрібні розмічені IBAN-документи;
    вимірюємо, скільки таких документів потрібно, щоб наздогнати патч.
"""
from __future__ import annotations

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.svm import LinearSVC

from .datasets import (make_pii_corpus, build_features, PATTERN_NAMES,
                       _person, FIRST, LAST)
from .ga_classifier import GARuleClassifier, Rule

MODELS = {
    "RandomForest": lambda s: RandomForestClassifier(n_estimators=200, random_state=s),
    "LinearSVM": lambda s: LinearSVC(random_state=s),
    "LogisticRegression": lambda s: LogisticRegression(max_iter=2000, random_state=s),
}
GA_PARAMS = {"lam": 0.05, "pop_size": 100, "generations": 100, "patience": 25}


def _make_iban(rng):
    return "UA" + "".join(str(rng.integers(0, 10)) for _ in range(27))


def add_iban_positives(texts, y, share, seed):
    """Новий тип ідентифікатора: частину НЕГАТИВІВ перетворюємо на
    IBAN-позитиви (єдиний критичний вміст документа це IBAN), решту
    негативів частково засмічуємо приманкою (шаблон переказу без IBAN).
    share задає частку IBAN-позитивів серед підсумкових позитивів."""
    rng = np.random.default_rng(seed)
    texts = list(texts)
    y = np.array(y)
    n_pos = int((y == 1).sum())
    k = int(n_pos * share / (1 - share))     # щоб частка серед позитивів була share
    neg_idx = np.where(y == 0)[0]
    chosen = rng.choice(neg_idx, size=k, replace=False)
    for i in chosen:
        base = texts[i].split(". ")
        base.insert(rng.integers(0, len(base) + 1),
                    f"wire transfer for {_person(rng)} iban {_make_iban(rng)}")
        texts[i] = ". ".join(base)
        y[i] = 1
    # близнюкова приманка: той самий переказ з особою, але замість IBAN
    # 27-цифровий розрахунковий референс (не відповідає формату IBAN)
    rest = np.setdiff1d(neg_idx, chosen)
    for i in rng.choice(rest, size=k, replace=False):
        base = texts[i].split(". ")
        ref = "".join(str(rng.integers(0, 10)) for _ in range(27))
        base.insert(rng.integers(0, len(base) + 1),
                    f"wire transfer for {_person(rng)} settlement ref {ref}")
        texts[i] = ". ".join(base)
    return texts, y, chosen


def run(seeds=(0, 1, 2)):
    res = {m: {"before": [], "after_nolabel": []} for m in MODELS}
    res["GA-rules"] = {"before": [], "patched": []}
    catchup = {str(k): [] for k in (10, 50, 200)}
    iban_recall = {"GA-patched": [], "RF": []}

    for seed in seeds:
        tr_texts, tr_y = make_pii_corpus(n=3000, seed=400 + seed, domain="fin")
        te_texts, te_y = make_pii_corpus(n=1500, seed=500 + seed, domain="fin")
        te_texts, te_y, iban_idx = add_iban_positives(te_texts, te_y, 0.33, 600 + seed)
        Xtr, Xte, names = build_features(tr_texts, te_texts)
        iidx = names.index("pat_iban")

        fitted = {}
        for mname, factory in MODELS.items():
            m = factory(seed)
            m.fit(Xtr, tr_y)
            fitted[mname] = m
            f1 = f1_score(te_y, m.predict(Xte), zero_division=0)
            res[mname]["before"].append(f1)
            res[mname]["after_nolabel"].append(f1)  # втручання без даних неможливе

        ga = GARuleClassifier(seed=seed, **GA_PARAMS).fit(Xtr, tr_y)
        res["GA-rules"]["before"].append(
            f1_score(te_y, ga.predict(Xte), zero_division=0))
        # втручання експерта: одне правило, нуль розмічених прикладів
        ga.rules_.append(Rule([(iidx, +1, 0.0)]))
        res["GA-rules"]["patched"].append(
            f1_score(te_y, ga.predict(Xte), zero_division=0))
        mask = np.zeros(len(te_y), dtype=bool)
        mask[iban_idx] = True
        iban_recall["GA-patched"].append(
            recall_score(te_y[mask], ga.predict(Xte[mask]), zero_division=0))
        iban_recall["RF"].append(
            recall_score(te_y[mask], fitted["RandomForest"].predict(Xte[mask]),
                         zero_division=0))

        # скільки розмічених IBAN-документів потрібно RF, щоб наздогнати
        extra_texts, extra_y = make_pii_corpus(n=1200, seed=700 + seed, domain="fin")
        extra_texts, extra_y, _ = add_iban_positives(extra_texts, extra_y, 0.33, 800 + seed)
        Xex, _, _ = build_features(extra_texts, te_texts[:1])
        # той самий ознаковий простір: перera-обчислення разом з train
        for k in (10, 50, 200):
            rng = np.random.default_rng(900 + seed + k)
            take = rng.choice(len(extra_y), size=k, replace=False)
            Xtr2, Xte2, _ = build_features(
                tr_texts + [extra_texts[i] for i in take], te_texts)
            y2 = np.concatenate([tr_y, extra_y[take]])
            m = MODELS["RandomForest"](seed)
            m.fit(Xtr2, y2)
            catchup[str(k)].append(f1_score(te_y, m.predict(Xte2), zero_division=0))
        print(f"seed={seed}: GA before={res['GA-rules']['before'][-1]:.3f} "
              f"patched={res['GA-rules']['patched'][-1]:.3f} | "
              f"RF={res['RandomForest']['before'][-1]:.3f} "
              f"RF+200lab={catchup['200'][-1]:.3f}")

    def agg(v):
        return {"mean": float(np.mean(v)), "std": float(np.std(v))}

    out = {
        "scenario": "поява нового типу ідентифікатора (IBAN), 33% позитивів тесту",
        "GA-rules": {k: agg(v) for k, v in res["GA-rules"].items()},
        **{m: {k: agg(v) for k, v in d.items()} for m, d in res.items() if m != "GA-rules"},
        "rf_catchup_labeled": {k: agg(v) for k, v in catchup.items()},
        "iban_recall": {k: agg(v) for k, v in iban_recall.items()},
    }
    with open("results/e1_patch.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nПідсумок:")
    print(f"  GA до втручання  F1={out['GA-rules']['before']['mean']:.3f}")
    print(f"  GA після правила F1={out['GA-rules']['patched']['mean']:.3f}")
    for m in MODELS:
        print(f"  {m:18s} F1={out[m]['before']['mean']:.3f} (втручання неможливе)")
    for k in ('10', '50', '200'):
        print(f"  RF + {k} розмічених: F1={out['rf_catchup_labeled'][k]['mean']:.3f}")
    print(f"  Повнота на IBAN-документах: GA-патч={out['iban_recall']['GA-patched']['mean']:.3f} "
          f"RF={out['iban_recall']['RF']['mean']:.3f}")


if __name__ == "__main__":
    run()
