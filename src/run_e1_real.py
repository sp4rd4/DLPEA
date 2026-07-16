"""E1 (реальні дані): класифікація документів за рівнем конфіденційності
на двох реальних корпусах:

  ai4privacy PII-Masking-200k - конфіденційність за наявністю критичних
      ідентифікаторів (картка, рахунок/IBAN, SSN, пароль, криптогаманець);
  20 Newsgroups - тематична конфіденційність (криптографія, медицина,
      зброя проти нейтральних тем).

Запропонований ГА-класифікатор на продукційних правилах проти п'яти поширених
методів машинного навчання. 5-блокова стратифікована перехресна валідація.
"""
from __future__ import annotations

import json
from pathlib import Path

from .datasets import load_20ng_binary
from .ai4privacy_corpus import load_ai4privacy
from .run_e1 import run_dataset, lambda_sensitivity, GA_20NG, RESULTS


def main():
    results = {}

    pii_texts, pii_y = load_ai4privacy(n=4000, seed=42)
    results["ai4privacy_pii"] = run_dataset(
        "ai4privacy PII", pii_texts, pii_y,
        ga_params={"lam": 0.05, "pop_size": 100, "generations": 100,
                   "patience": 25})

    ng_texts, ng_y = load_20ng_binary(seed=42, max_docs=4000)
    results["20ng_binary"] = run_dataset("20NG binary", ng_texts, ng_y,
                                         ga_params=GA_20NG)

    print("== lambda sensitivity (20NG)")
    results["lambda_sensitivity_20ng"] = lambda_sensitivity(
        ng_texts, ng_y, ga_params={k: v for k, v in GA_20NG.items()})

    with open(RESULTS / "e1_real.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved -> e1_real.json")


if __name__ == "__main__":
    main()
