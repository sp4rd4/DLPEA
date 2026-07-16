# -*- coding: utf-8 -*-
"""Зведений дайджест результатів розділу 4 (реальні дані) для перевірки тексту."""
import json

def L(p): return json.load(open(f"results/{p}", encoding="utf-8"))

e1 = L("e1_real.json"); e2 = L("e2_cert.json")
e3s = L("e3_synth.json"); e3r = L("e3_real.json"); e4 = L("e4_real.json")

out = []
def w(s=""): out.append(s)

w("# ДАЙДЖЕСТ РЕЗУЛЬТАТІВ РОЗДІЛУ 4 (реальні дані)\n")

w("## E1. Класифікація документів (ai4privacy PII + 20 Newsgroups, реальні корпуси)")
for ds, title in [("ai4privacy_pii", "ai4privacy PII (критичні ідентифікатори)"),
                  ("20ng_binary", "20 Newsgroups (тематична конфіденційність)")]:
    w(f"### {title}")
    for m in ["RandomForest", "GradientBoosting", "LinearSVM", "LogisticRegression", "NaiveBayes", "GA-rules"]:
        r = e1[ds][m]
        w(f"  {m}: F1={r['f1']['mean']:.3f}±{r['f1']['std']:.3f} "
          f"P={r['precision']['mean']:.3f} R={r['recall']['mean']:.3f} "
          f"fit={r['fit_time']['mean']:.2f}s pred/1k={r['predict_time_per_1k']['mean']:.3f}s")
    it = e1[ds]["GA-rules"]["interpretability"]
    w(f"  GA інтерпретованість: правил={it['n_rules']['mean']:.1f} умов/правило={it['avg_conds']['mean']:.1f}")
    w("  Приклади правил GA-rules:")
    for rule in e1[ds]["GA-rules"]["example_rules"]:
        w(f"    - {rule}")
w("### lambda-чутливість (20NG)")
for lam, v in e1["lambda_sensitivity_20ng"].items():
    w(f"  lambda={lam}: F1={v['f1_mean']:.3f}±{v['f1_std']:.3f} правил={v['rules_mean']:.1f}")

w("\n## E2. Поведінкове профілювання (CERT Insider Threat r4.2, реальний)")
w(f"  користувачів={e2['n_users']}, днів={e2['n_days']}, інсайдерів={e2['n_insiders']}, "
  f"шкідливих user-days={e2['n_malicious_userdays']}, ознак={e2['n_features']}, split={e2['split_day']}")
w(f"  активні ознаки ГА-профілю: {e2['active_features']}")
for k, nm in [("GA-profile", "ГА-профіль (запропонований)"), ("isolation_forest", "Isolation Forest"),
              ("lof", "LOF"), ("ocsvm", "One-Class SVM")]:
    r = e2["results"][k]
    w(f"  {nm}: AP={r['ap']:.3f} P@20={r['precision@20']:.3f} P@50={r['precision@50']:.3f} "
      f"R@50={r['recall@50']:.3f} R@100={r['recall@100']:.3f} AUC={r['auc']:.3f}")

w("\n## E3. Виявлення дрейфу концепції")
w("### 3а. Стандартні потоки RandomRBF (генератор river), коваріантний дрейф, "
  f"{e3s['n_streams']} потоків, {e3s['n_concepts']} концепти")
for kind, nm in [("abrupt", "раптовий"), ("gradual", "поступовий")]:
    w(f"  [{nm} дрейф, tolerance={e3s['tolerance'][kind]}]")
    for m in ["KS+confirm (proposed)", "KS no-confirm", "adwin", "ddm", "eddm", "kswin"]:
        r = e3s[kind][m]
        w(f"    {m}: P={r['precision_mean']:.3f} R={r['recall_mean']:.3f} "
          f"F1={r['f1_mean']:.3f} затримка={r['mean_delay_mean']:.0f} "
          f"хибних={r['false_alarms_mean']:.2f}")
w(f"### 3б. Реальні потоки INSECTS (Souza 2020), {e3r['n_variants']} варіанти, tolerance={e3r['tolerance']}")
for m in ["KS+confirm (proposed)", "KS no-confirm", "adwin", "ddm", "eddm", "kswin"]:
    r = e3r["mean_over_variants"][m]
    w(f"    {m}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
      f"затримка={r['mean_delay']:.0f} хибних={r['false_alarms']:.2f}")

w("\n## E4. Еволюційна адаптація політик (Elec2 + INSECTS-Abr, реальні)")
for key in ["elec2", "insects_abrupt"]:
    v = e4[key]
    w(f"  {v['dataset']} (n={v['n_samples']}): "
      f"статична={v['static']['prequential_acc']:.3f} "
      f"адаптивна={v['adaptive']['prequential_acc']:.3f} "
      f"(перенавчань={v['adaptive']['n_retrains']}) "
      f"інкрементальна={v['incremental']['prequential_acc']:.3f} | "
      f"адаптивна проти статичної {v['improvement_vs_static_pct']:+.1f}%")

text = "\n".join(out)
open("results/DIGEST_ch4.txt", "w", encoding="utf-8").write(text)
print(text)
