"""E3 (стандартні генератори): виявлення коваріантного дрейфу концепції на
потоках RandomRBF (river) з точно відомими позиціями зсуву. Запропонований
детектор (багатовимірний KS + корекція Бонферроні + правило підтвердження з
k послідовних сигналів) проти власної абляції без підтвердження та класичних
детекторів ADWIN / DDM / EDDM / KSWIN.

Головна теза: правило підтвердження істотно зменшує частоту хибних тривог за
прийнятного зростання затримки, а на коваріантному дрейфі багатовимірний
KS-детектор перевершує детектори на потоці помилок за точністю виявлення.
"""
from __future__ import annotations

import json
import numpy as np

from .drift import ks_confirm_detector, river_detector_alarms
from .synth_streams import make_abrupt, make_gradual
from .real_streams import standardize_prefix

N_STREAMS = 10
SEG = 3000
N_CONC = 4
TOL = {"abrupt": 500, "gradual": 1200}
WARM = 500


def score(alarms, cps, tol):
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
    by_kind = {"abrupt": {m: [] for m in methods},
               "gradual": {m: [] for m in methods}}

    for kind in ["abrupt", "gradual"]:
        gen = make_abrupt if kind == "abrupt" else make_gradual
        for s in range(N_STREAMS):
            X, y, cps = gen(n_concepts=N_CONC, seg_len=SEG, seed=s)
            Xs = standardize_prefix(X, WARM)
            win = 300
            a = ks_confirm_detector(Xs, window=win, step=win // 4, alpha=0.05, k_confirm=3)
            by_kind[kind]["KS+confirm (proposed)"].append(score(a, cps, TOL[kind]))
            a0 = ks_confirm_detector(Xs, window=win, step=win // 4, alpha=0.05, k_confirm=1)
            by_kind[kind]["KS no-confirm"].append(score(a0, cps, TOL[kind]))
            for det in ["adwin", "ddm", "eddm", "kswin"]:
                al = river_detector_alarms(Xs, y, det, seed=s)
                by_kind[kind][det].append(score(al, cps, TOL[kind]))
        print(f"[{kind}] завершено {N_STREAMS} потоків")

    def agg(d):
        out = {}
        for m, runs in d.items():
            keys = ["precision", "recall", "f1", "mean_delay", "n_alarms", "false_alarms"]
            out[m] = {}
            for k in keys:
                vals = [r[k] for r in runs if not (isinstance(r[k], float) and np.isnan(r[k]))]
                out[m][k + "_mean"] = float(np.mean(vals)) if vals else float("nan")
                out[m][k + "_std"] = float(np.std(vals)) if vals else float("nan")
        return out

    summary = {"abrupt": agg(by_kind["abrupt"]), "gradual": agg(by_kind["gradual"]),
               "n_streams": N_STREAMS, "n_concepts": N_CONC, "seg_len": SEG,
               "tolerance": TOL}
    with open("results/e3_synth.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for kind in ["abrupt", "gradual"]:
        print(f"\n=== {kind} (середнє за {N_STREAMS} потоками):")
        for m in methods:
            d = summary[kind][m]
            print(f"  {m:26s} P={d['precision_mean']:.3f} R={d['recall_mean']:.3f} "
                  f"F1={d['f1_mean']:.3f} delay={d['mean_delay_mean']:.0f} "
                  f"FA={d['false_alarms_mean']:.1f}")


if __name__ == "__main__":
    run()
