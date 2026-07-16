"""ГА-класифікатор на продукційних правилах IF-THEN (метод п. 3.1).

Хромосома = набір правил (до K). Правило = кон'юнкція до C умов виду
(ознака, знак, поріг). Документ позитивний, якщо спрацювало хоча б одне правило.
Пристосованість: F1 - lambda * складність (регуляризація).
"""
from __future__ import annotations

import time
import numpy as np
from sklearn.metrics import f1_score


class Rule:
    __slots__ = ("conds",)

    def __init__(self, conds):
        self.conds = conds                     # list[(feat, op, thr)], op: +1 '>', -1 '<'

    def fire(self, X):
        m = np.ones(X.shape[0], dtype=bool)
        for f, op, t in self.conds:
            m &= (X[:, f] > t) if op > 0 else (X[:, f] < t)
        return m

    def describe(self, names):
        parts = [f"{names[f]} {'>' if op > 0 else '<'} {t:.3g}"
                 for f, op, t in self.conds]
        return "IF " + " AND ".join(parts) + " THEN confidential"


class GARuleClassifier:
    def __init__(self, max_rules=8, max_conds=3, pop_size=60, generations=80,
                 lam=0.05, tournament=3, p_mut=0.35, elite=2, patience=15,
                 n_restarts=2, seed=0):
        self.K, self.C = max_rules, max_conds
        self.pop_size, self.generations = pop_size, generations
        self.lam, self.tournament = lam, tournament
        self.p_mut, self.elite, self.patience = p_mut, elite, patience
        self.n_restarts = n_restarts
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.rules_: list[Rule] = []
        self.history_: list[float] = []

    # -- ініціалізація: евристика на статистиках класів + випадкові правила --
    def _heuristic_rule(self, X, y, informative):
        """Кон'юнкція 1-2 умов на найінформативніших ознаках; поріг — середина
        між середніми класів (специфічні AND-правила в дусі політик СЗВ)."""
        k = 2 if self.rng.random() < 0.7 else 1
        feats = self.rng.choice(informative[-12:], size=min(k, len(informative)),
                                replace=False)
        conds = []
        for f in feats:
            f = int(f)
            mu1, mu0 = X[y == 1, f].mean(), X[y == 0, f].mean()
            conds.append((f, 1 if mu1 > mu0 else -1, (mu1 + mu0) / 2))
        return Rule(conds)

    def _random_rule(self, X):
        n_conds = int(self.rng.integers(1, self.C + 1))
        conds = []
        for _ in range(n_conds):
            f = int(self.rng.integers(X.shape[1]))
            lo, hi = X[:, f].min(), X[:, f].max()
            thr = float(self.rng.uniform(lo, hi))
            conds.append((f, int(self.rng.choice([-1, 1])), thr))
        return Rule(conds)

    def _covering_seed(self, X, y, informative):
        """Жадібне послідовне покриття (в дусі CN2/RIPPER): правило
        нарощується умовами, що максимізують F1 правила відносно ще не
        покритих позитивів (баланс точності та покриття); прийняте правило
        виключає покриті позитиви, процес повторюється."""
        remaining = (y == 1).copy()
        neg = (y == 0)
        rules = []
        cand_feats = [int(f) for f in informative]

        def rule_score(m):
            tp = (m & remaining).sum()
            if tp == 0:
                return 0.0
            prec = tp / (tp + (m & neg).sum())
            rec = tp / max(remaining.sum(), 1)
            return 2 * prec * rec / (prec + rec)

        while remaining.sum() > 0.02 * (y == 1).sum() and len(rules) < self.K:
            conds, fire = [], np.ones(len(y), dtype=bool)
            best_score = 0.0
            while len(conds) < self.C:
                best = None
                for f in cand_feats:
                    if any(f == c[0] for c in conds):
                        continue
                    mu1 = X[remaining, f].mean() if remaining.any() else 0.0
                    mu0 = X[neg, f].mean()
                    op = 1 if mu1 > mu0 else -1
                    # кандидати порогів: середина між класами + квантилі
                    # розподілу непокритих позитивів
                    pos_vals = X[remaining, f] if remaining.any() else X[:, f]
                    thrs = {(mu1 + mu0) / 2}
                    for q in (10, 25, 50):
                        qv = float(np.percentile(pos_vals, q))
                        thrs.add(qv - 1e-9 if op > 0 else qv + 1e-9)
                    for thr in thrs:
                        m = fire & ((X[:, f] > thr) if op > 0
                                    else (X[:, f] < thr))
                        s = rule_score(m)
                        if s > best_score + 1e-6:
                            best_score, best = s, ((f, op, thr), m)
                if best is None:
                    break
                conds.append(best[0])
                fire = best[1]
            if not conds or not (fire & remaining).any():
                break
            rules.append(Rule(conds))
            remaining &= ~fire
        return rules or [self._heuristic_rule(X, y, informative)]

    def _init_pop(self, X, y):
        diff = np.abs(X[y == 1].mean(0) - X[y == 0].mean(0)) / (X.std(0) + 1e-9)
        informative = np.argsort(diff)[-30:]
        self._informative = informative
        seed_rules = self._covering_seed(X, y, informative)
        pop = []
        for i in range(self.pop_size):
            if i < self.pop_size * 0.1:        # жадібне покриття + варіації
                rules = [Rule(list(r.conds)) for r in seed_rules]
                if i > 0:
                    rules = self._mutate(rules, X, y)
                pop.append(rules)
            elif i < self.pop_size * 0.4:      # евристична частка (rho=0.4)
                n_rules = int(self.rng.integers(3, 7))
                pop.append([self._heuristic_rule(X, y, informative)
                            for _ in range(n_rules)])
            else:
                n_rules = int(self.rng.integers(2, 6))
                pop.append([self._random_rule(X) for _ in range(n_rules)])
        return pop

    # -- оцінювання --
    def _predict_rules(self, rules, X):
        out = np.zeros(X.shape[0], dtype=bool)
        for r in rules:
            out |= r.fire(X)
        return out.astype(int)

    def _complexity(self, rules):
        n_conds = sum(len(r.conds) for r in rules)
        return 0.5 * len(rules) / self.K + 0.5 * n_conds / (self.K * self.C)

    def _fitness(self, rules, X, y):
        pred = self._predict_rules(rules, X)
        return f1_score(y, pred, zero_division=0) - self.lam * self._complexity(rules)

    # -- оператори --
    def _specialize(self, rules, X, y):
        """Спрямована спеціалізація: до правила з найбільшою кількістю хибних
        спрацювань додається умова з найбільшим приростом пристосованості."""
        if not rules:
            return rules
        fp_counts = [int((r.fire(X) & (y == 0)).sum()) for r in rules]
        ri = int(np.argmax(fp_counts))
        target = rules[ri]
        if fp_counts[ri] == 0 or len(target.conds) >= self.C:
            return rules
        base_fit = self._fitness(rules, X, y)
        used = {f for f, _, _ in target.conds}
        best_gain, best_cond = 0.0, None
        for f in self._informative[-12:]:
            f = int(f)
            if f in used:
                continue
            mu1, mu0 = X[y == 1, f].mean(), X[y == 0, f].mean()
            cond = (f, 1 if mu1 > mu0 else -1, (mu1 + mu0) / 2)
            target.conds.append(cond)
            gain = self._fitness(rules, X, y) - base_fit
            target.conds.pop()
            if gain > best_gain:
                best_gain, best_cond = gain, cond
        if best_cond is not None:
            target.conds.append(best_cond)
        return rules

    def _mutate(self, rules, X, y=None):
        rules = [Rule(list(r.conds)) for r in rules]
        roll = self.rng.random()
        if roll < 0.15 and y is not None:
            return self._specialize(rules, X, y)
        if roll < 0.45 and rules:                       # зсув порога
            r = rules[self.rng.integers(len(rules))]
            i = self.rng.integers(len(r.conds))
            f, op, t = r.conds[i]
            t += float(self.rng.normal(0, 0.2 * (X[:, f].std() + 1e-9)))
            r.conds[i] = (f, op, t)
        elif roll < 0.6 and rules:                      # заміна ознаки
            r = rules[self.rng.integers(len(rules))]
            i = self.rng.integers(len(r.conds))
            _, op, _ = r.conds[i]
            f = int(self.rng.integers(X.shape[1]))
            r.conds[i] = (f, op, float(np.median(X[:, f])))
        elif roll < 0.75:                               # додати/прибрати умову
            r = rules[self.rng.integers(len(rules))] if rules else None
            if r and len(r.conds) > 1 and self.rng.random() < 0.5:
                r.conds.pop(self.rng.integers(len(r.conds)))
            elif r and len(r.conds) < self.C:
                f = int(self.rng.integers(X.shape[1]))
                r.conds.append((f, int(self.rng.choice([-1, 1])),
                                float(np.median(X[:, f]))))
        elif roll < 0.9 and len(rules) < self.K:        # додати правило
            rules.append(self._random_rule(X))
        elif len(rules) > 1:                            # прибрати правило
            rules.pop(self.rng.integers(len(rules)))
        return rules

    def _crossover(self, a, b):
        cut_a = self.rng.integers(0, len(a) + 1)
        cut_b = self.rng.integers(0, len(b) + 1)
        child = [Rule(list(r.conds)) for r in (a[:cut_a] + b[cut_b:])]
        return child[:self.K] if child else [Rule(list(a[0].conds))]

    def _select(self, pop, fits):
        idx = self.rng.integers(0, len(pop), size=self.tournament)
        return pop[idx[np.argmax(fits[idx])]]

    # -- основний цикл --
    def fit(self, X, y):
        """Мультистарт: незалежні запуски еволюції, краща пристосованість."""
        t0 = time.perf_counter()
        best_rules, best_fit, best_hist = None, -np.inf, []
        for r in range(max(1, self.n_restarts)):
            self.rng = np.random.default_rng(self.seed + 1000 * r)
            rules, fit, hist = self._evolve(X, y)
            if fit > best_fit:
                best_rules, best_fit, best_hist = rules, fit, hist
        self.rules_, self.history_ = best_rules, best_hist
        self.fit_time_ = time.perf_counter() - t0
        return self

    def _evolve(self, X, y):
        pop = self._init_pop(X, y)
        fits = np.array([self._fitness(p, X, y) for p in pop])
        best, best_fit, stall = pop[int(np.argmax(fits))], fits.max(), 0
        history = [best_fit]
        for _ in range(self.generations):
            order = np.argsort(fits)[::-1]
            new_pop = [pop[i] for i in order[:self.elite]]
            while len(new_pop) < self.pop_size:
                child = self._crossover(self._select(pop, fits),
                                        self._select(pop, fits))
                if self.rng.random() < self.p_mut:
                    child = self._mutate(child, X, y)
                new_pop.append(child)
            # ін'єкція нових особин для підтримки різноманіття (10% найгірших)
            n_inj = max(1, self.pop_size // 10)
            worst = np.argsort([self._fitness(p, X, y) for p in pop])[:n_inj]
            for wi in worst:
                if self.rng.random() < 0.5:
                    pop[wi] = [self._heuristic_rule(X, y, self._informative)
                               for _ in range(int(self.rng.integers(3, 7)))]
                else:
                    pop[wi] = [self._random_rule(X)
                               for _ in range(int(self.rng.integers(2, 6)))]
            fits = np.array([self._fitness(p, X, y) for p in pop])
            gen_best = fits.max()
            if gen_best > best_fit + 1e-6:
                best, best_fit, stall = pop[int(np.argmax(fits))], gen_best, 0
            else:
                stall += 1
            history.append(best_fit)
            if stall >= self.patience:
                break
        return best, best_fit, history

    def predict(self, X):
        return self._predict_rules(self.rules_, X)

    # -- метрики інтерпретованості --
    def interpretability(self, X):
        n_rules = len(self.rules_)
        n_conds = sum(len(r.conds) for r in self.rules_)
        fired = np.zeros(X.shape[0])
        for r in self.rules_:
            fired += r.fire(X)
        pos = fired > 0
        share_le2 = float((fired[pos] <= 2).mean()) if pos.any() else 1.0
        return {"n_rules": n_rules,
                "avg_conds": n_conds / max(n_rules, 1),
                "share_pos_explained_le2_rules": share_le2}

    def describe(self, names):
        return [r.describe(names) for r in self.rules_]
