"""E3 (реальні дані): виявлення дрейфу концепції на потоках INSECTS
(Souza et al., 2020) з відомими позиціями зміни концепції. Запропонований
детектор KS+підтвердження проти ADWIN / DDM / EDDM / KSWIN.

Тривога зарахована як істинна, якщо потрапляє у вікно [точка; точка+tol].
Показники усереднено за варіантами потоку.
"""
from __future__ import annotations

import json
import numpy as np

from .drift import ks_confirm_detector, river_detector_alarms
from .real_streams import load_insects, standardize_prefix

VARIANTS = [
    "abrupt_balanced",
    "gradual_balanced",
    "incremental_reoccurring_balanced",
    "incremental_abrupt_balanced",
]

TOL = 2000  # вікно допуску (зразків) для зарахування тривоги на реальному потоці
WARM = 1000


def score(alarms, cps, tol=TOL):
    matched, tp = {}, set()
    for a in alarms:
        for p in cps:
            if p <= a <= p + tol:
                matched.setdefault(p, a - p)
                tp.add(a)
                break
    precision = len(tp) / len(alarms) if alarms else 0.0
    recall = len(matched) / len(cps) if cps else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    delay = float(np.mean(list(matched.values()))) if matched else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1,
            "mean_delay": delay, "n_alarms": len(alarms),
            "false_alarms": len(alarms) - len(tp)}


def run():
    methods = ["KS+confirm (proposed)", "KS no-confirm", "adwin", "ddm", "eddm", "kswin"]
    acc = {m: [] for m in methods}
    per_variant = {}
    for v in VARIANTS:
        X, y, cps = load_insects(variant=v)
        Xs = standardize_prefix(X, WARM)
        n = len(Xs)
        # масштабування вікна KS до довжини потоку
        win = max(300, n // 120)
        res_v = {}
        a_ks = ks_confirm_detector(Xs, window=win, step=win // 4, alpha=0.05, k_confirm=3)
        res_v["KS+confirm (proposed)"] = score(a_ks, cps)
        a_ks0 = ks_confirm_detector(Xs, window=win, step=win // 4, alpha=0.05, k_confirm=1)
        res_v["KS no-confirm"] = score(a_ks0, cps)
        for kind in ["adwin", "ddm", "eddm", "kswin"]:
            a = river_detector_alarms(Xs, y, kind, seed=0)
            res_v[kind] = score(a, cps)
        for m in methods:
            acc[m].append(res_v[m])
        per_variant[v] = {"n": n, "change_points": cps, "window": win, "results": res_v}
        print(f"[{v}] n={n} cps={cps} KS+conf f1={res_v['KS+confirm (proposed)']['f1']:.3f} "
              f"adwin f1={res_v['adwin']['f1']:.3f}")

    def agg(key):
        out = {}
        for m in methods:
            vals = {k: [r[k] for r in acc[m] if not (isinstance(r[k], float) and np.isnan(r[k]))]
                    for k in ["precision", "recall", "f1", "mean_delay", "n_alarms", "false_alarms"]}
            out[m] = {k: (float(np.mean(v)) if v else float("nan")) for k, v in vals.items()}
        return out

    summary = {"per_variant": per_variant, "mean_over_variants": agg("all"),
               "tolerance": TOL, "n_variants": len(VARIANTS)}
    with open("results/e3_real.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nСереднє за варіантами:")
    for m, d in summary["mean_over_variants"].items():
        print(f"  {m:26s} P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} "
              f"delay={d['mean_delay']:.0f} FA={d['false_alarms']:.1f}")


if __name__ == "__main__":
    run()
