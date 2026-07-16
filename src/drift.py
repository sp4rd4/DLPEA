"""Виявлення дрейфу концепції (п. 3.2): багатоознаковий детектор на основі
тесту Колмогорова-Смирнова з корекцією Бонферроні та правилом підтвердження
(k послідовних сигналів), проти базлайнів ADWIN / DDM / EDDM.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def ks_confirm_detector(X, window=300, step=50, alpha=0.05, k_confirm=3):
    """Повертає позиції тривог. Референсне вікно скидається після тривоги."""
    n, m = X.shape
    alarms = []
    ref_start, ref_end = 0, window
    consec = 0
    pos = ref_end + window
    while pos <= n:
        cur = X[pos - window:pos]
        ref = X[ref_start:ref_end]
        pvals = [ks_2samp(ref[:, j], cur[:, j]).pvalue for j in range(m)]
        signal = min(pvals) < alpha / m            # корекція Бонферроні
        consec = consec + 1 if signal else 0
        if consec >= k_confirm:
            alarms.append(pos)
            ref_start, ref_end = pos, pos + window  # новий референс
            pos = ref_end + window
            consec = 0
            continue
        pos += step
    return alarms


def river_detector_alarms(X, y, kind, seed=0):
    """Базлайни з бібліотеки river: детектори на потоці помилок
    інкрементального класифікатора (логістична регресія SGD)."""
    from sklearn.linear_model import SGDClassifier
    from river import drift
    if kind == "adwin":
        det = drift.ADWIN()
    elif kind == "ddm":
        det = drift.binary.DDM()
    elif kind == "eddm":
        det = drift.binary.EDDM()
    elif kind == "kswin":
        det = drift.KSWIN(alpha=0.005, seed=seed)
    else:
        raise ValueError(kind)
    clf = SGDClassifier(loss="log_loss", random_state=seed)
    n = len(y)
    warm = 200
    all_classes = np.unique(y)
    clf.partial_fit(X[:warm], y[:warm], classes=all_classes)
    alarms = []
    for i in range(warm, n):
        pred = clf.predict(X[i:i + 1])[0]
        err = int(pred != y[i])
        det.update(err)
        if det.drift_detected:
            alarms.append(i)
        clf.partial_fit(X[i:i + 1], y[i:i + 1])
    return alarms


TOLERANCE = {"sudden": 700, "gradual": 1800}


def score_alarms(alarms, drifts, n, tolerance=None):
    """Precision/recall/затримка: тривога зарахована, якщо потрапляє у вікно
    [позиція дрейфу; позиція + tolerance(тип)]; кожен дрейф зараховується
    один раз. Для поступового дрейфу вікно ширше (тривалість перехідного
    періоду)."""
    matched = {}
    tp_alarms = set()
    for a in alarms:
        for pos, typ in drifts:
            tol = tolerance if tolerance else TOLERANCE[typ]
            if pos <= a <= pos + tol:
                # повторна тривога в межах того самого перехідного періоду
                # не вважається хибною, але дрейф зараховується один раз
                if pos not in matched:
                    matched[pos] = a - pos
                tp_alarms.add(a)
                break
    precision = len(tp_alarms) / len(alarms) if alarms else 0.0
    recall = len(matched) / len(drifts) if drifts else 0.0
    delay = float(np.mean(list(matched.values()))) if matched else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "mean_delay": delay, "n_alarms": len(alarms),
            "false_alarms": len(alarms) - len(tp_alarms)}
