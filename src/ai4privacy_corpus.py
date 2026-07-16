"""Завантаження реального корпусу ai4privacy PII-Masking-200k і формування
бінарної задачі класифікації за рівнем конфіденційності.

Документ вважаємо конфіденційним, якщо він містить принаймні один критичний
ідентифікатор (номер платіжної картки, банківський рахунок чи IBAN, номер
соціального страхування, пароль, криптогаманець). Інші документи, що містять
лише малочутливі персональні згадки (ім'я, місто, посада, дата), належать до
неконфіденційних. Такий поділ відтворює реальне розрізнення у політиках ЗВД:
конфіденційність визначає не будь-яка згадка особи, а наявність критичних
ідентифікаторів.

Джерело:
Ai4Privacy. PII-Masking-200k. Hugging Face, 2024.
URL: https://huggingface.co/datasets/ai4privacy/pii-masking-200k
"""
from __future__ import annotations

import os

CRITICAL = {
    "CREDITCARDNUMBER", "CREDITCARDCVV", "SSN", "IBAN", "ACCOUNTNUMBER",
    "PASSWORD", "BITCOINADDRESS", "ETHEREUMADDRESS", "MASKEDNUMBER",
    "CREDITCARDISSUER", "PIN",
}


def load_ai4privacy(n=4000, seed=42):
    """Повертає (texts, y): y=1 якщо документ містить критичний ідентифікатор."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from datasets import load_dataset
    import numpy as np

    ds = load_dataset("ai4privacy/pii-masking-200k", split="train")
    ds = ds.filter(lambda r: r["language"] == "en")

    texts, labels = [], []
    for r in ds:
        labs = set(m["label"] for m in r["privacy_mask"])
        y = 1 if labs & CRITICAL else 0
        texts.append(r["source_text"])
        labels.append(y)
    texts = np.array(texts, dtype=object)
    labels = np.array(labels, dtype=int)

    # збалансований підвибір близько 30% позитивного класу, як у типовому потоці
    rng = np.random.default_rng(seed)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    n_pos = min(len(pos), int(n * 0.30))
    n_neg = min(len(neg), n - n_pos)
    idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                          rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(idx)
    return list(texts[idx]), labels[idx]
