"""Генерація та завантаження датасетів для експериментів розділу 4.

Набори:
  1) Synthetic PII corpus  - контрольований корпус ділових документів з PII.
  2) 20 Newsgroups (binary) - проксі тематичної конфіденційності (реальні тексти).
  3) Behavior logs          - синтетичні журнали поведінки користувачів з
                              ін'єкованими епізодами ексфільтрації.
  4) Drift stream           - потік з раптовим та поступовим дрейфом концепції.
"""
from __future__ import annotations

import re
import numpy as np

# ---------------------------------------------------------------------------
# 1. Synthetic PII corpus
# ---------------------------------------------------------------------------

FIRST = ["john", "maria", "petro", "olena", "andrew", "kate", "ivan", "olga",
         "mark", "anna", "serhii", "nina", "paul", "iryna", "viktor", "daria",
         "james", "lucy", "robert", "emma", "david", "sofia", "michael",
         "laura", "thomas", "alice", "peter", "julia", "steven", "diana",
         "oleh", "tetiana", "roman", "oksana", "yurii", "natalia", "denys",
         "alina", "vadym", "marta", "taras", "lesia", "bohdan", "halyna",
         "george", "helen", "frank", "clara", "simon", "vera", "adrian",
         "nadia", "leon", "zoria", "artem", "lidia", "marko", "ruslana"]
LAST = ["smith", "kovalenko", "brown", "shevchenko", "miller", "tkachenko",
        "wilson", "bondar", "taylor", "kravets", "davis", "melnyk",
        "johnson", "boyko", "white", "savchenko", "harris", "rudenko",
        "martin", "lysenko", "clark", "moroz", "lewis", "khomenko",
        "walker", "polishchuk", "hall", "marchenko", "young", "vasylenko",
        "allen", "romanenko", "king", "pavlenko", "wright", "tymoshenko",
        "scott", "kushnir", "green", "havryliuk", "adams", "zinchenko",
        "baker", "horbach", "nelson", "drozd", "carter", "yakovenko",
        "mitchell", "kucher", "perez", "shvets", "roberts", "honchar",
        "turner", "datsenko", "phillips", "verbytskyi"]
DEPTS = ["finance", "sales", "engineering", "legal", "hr", "support",
         "marketing", "operations"]
PRODUCTS = ["analytics platform", "billing module", "mobile app", "api gateway",
            "report builder", "data warehouse", "crm integration"]


def _luhn_checksum(digits):
    total = 0
    for i, d in enumerate(reversed(digits)):
        d = int(d)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10


def _make_card(rng):
    digits = [str(rng.integers(0, 10)) for _ in range(15)]
    digits = ["4"] + digits[1:]
    for last in "0123456789":
        if _luhn_checksum(digits + [last]) == 0:
            digits.append(last)
            break
    s = "".join(digits)
    return f"{s[:4]} {s[4:8]} {s[8:12]} {s[12:]}"


def _make_phone(rng):
    return f"+380{rng.integers(50, 99)}{rng.integers(1000000, 9999999)}"


def _make_email(rng):
    return (f"{rng.choice(FIRST)}.{rng.choice(LAST)}"
            f"@{rng.choice(['corp.example.com', 'gmail.com', 'company.ua'])}")


def _make_ipn(rng):
    return str(rng.integers(2000000000, 3999999999))


def _person(rng):
    return f"{rng.choice(FIRST).capitalize()} {rng.choice(LAST).capitalize()}"


def _serial16(rng):
    """16-значний серійник у форматі картки (НЕ платіжні дані), важкий негатив."""
    return " ".join(str(rng.integers(1000, 9999)) for _ in range(4))


def make_pii_corpus(n=4000, pos_share=0.30, seed=42, label_noise=0.03,
                    obfuscate_share=0.08, domain="fin"):
    """Корпус ділових текстів з композиційною генерацією: обидва класи
    збирають з ОДНОГО пулу ділових речень, тож лексика класів збігається.
    Відмінність лише у вставлених сутностях:
      позитив: справжні критичні PII (картка; особа + ІПН/зарплата/діагноз;
               особа + >=2 ідентифікатори);
      негатив: приманки у тих самих формулюваннях (серійник у форматі картки,
               публічний телефон, командна скринька, ставки без особи,
               ім'я особи без ідентифікаторів).

    Реалістичні недосконалості:
      label_noise     - частка інвертованих міток (неоднозначність ручного
                        маркування реальних корпусів);
      obfuscate_share - частка позитивів, де номер картки обфусковано
                        (остання цифра замінена літерою O, регулярний вираз
                        його не бачить, класифікатор мусить спиратися на
                        контекст).
    """
    rng = np.random.default_rng(seed)

    def body():
        if domain == "fin":
            pool = [
                f"meeting notes for {rng.choice(PRODUCTS)} release plan q{rng.integers(1, 5)}",
                f"invoice {rng.integers(10000, 99999)} total {rng.integers(100, 50000)} usd net {rng.integers(10, 60)} days",
                f"please process the payment request before the close of the quarter",
                f"payroll export for {rng.choice(DEPTS)} is scheduled for friday",
                f"the salary discussion is moved to the next review cycle",
                f"card processing metrics look stable after the gateway update",
                f"support ticket {rng.integers(100000, 999999)} timeout on {rng.choice(PRODUCTS)} severity {rng.integers(1, 4)}",
                f"access request approved for the {rng.choice(DEPTS)} shared folder",
                f"employee onboarding checklist updated per the new process",
                f"refund workflow now requires a second approval step",
                f"build {rng.integers(1000, 9999)} passed coverage {rng.integers(60, 99)} percent",
                f"reminder to update the contact record in the crm",
                f"medical insurance enrollment window closes next month",
                f"draft do not distribute yet, summary follows",
                f"forwarded from the shared inbox, see thread below",
                f"tax reporting deadline confirmed by the {rng.choice(DEPTS)} team",
            ]
        else:  # domain == "med": інша лексика документообігу, та сама політика
            pool = [
                f"clinical intake schedule updated for ward {rng.integers(1, 9)} rotations",
                f"discharge summary template revised by the review board",
                f"lab order {rng.integers(100000, 999999)} results pending from the external facility",
                f"appointment backlog for the outpatient clinic reduced this week",
                f"nursing shift handover checklist published on the portal",
                f"equipment maintenance visit planned for the imaging unit",
                f"consent form wording aligned with the updated regulation",
                f"referral workflow now requires the attending physician sign off",
                f"pharmacy stock reconciliation completed for controlled items",
                f"patient satisfaction survey window closes at month end",
                f"incident report {rng.integers(1000, 9999)} filed for the night shift",
                f"telehealth session quality metrics reviewed by the it group",
                f"vaccination campaign figures reported to the regional office",
                f"draft protocol amendment, circulation restricted to the committee",
                f"archived charts migration to the new records system continues",
                f"insurance claim batch submitted for adjudication",
            ]
        k = rng.integers(2, 5)
        return [str(pool[i]) for i in rng.choice(len(pool), size=k, replace=False)]

    # «близнюкові» вставки: однакові формулювання, різні сутності;
    # у домені med формулювання-синоніми, але ті самі патерни політики
    def pos_insert():
        kind = rng.integers(5)
        if domain == "fin":
            if kind == 0:
                return (f"charge approved for {_person(rng)} card {_make_card(rng)} "
                        f"amount {rng.integers(100, 9000)} usd")
            if kind == 1:
                return (f"record update {_person(rng)} tax id {_make_ipn(rng)} "
                        f"phone {_make_phone(rng)}")
            if kind == 2:
                return (f"approved rate for {_person(rng)} is "
                        f"{rng.integers(900, 8000)} usd monthly")
            if kind == 3:
                return (f"refund to card {_make_card(rng)} customer {_person(rng)}")
            return (f"leave request {_person(rng)} diagnosis "
                    f"{rng.choice(['J06', 'M54', 'K29'])}.{rng.integers(0, 9)} "
                    f"contact {_make_email(rng)}")
        else:
            if kind == 0:
                return (f"billing settled for {_person(rng)} card {_make_card(rng)} "
                        f"copay {rng.integers(100, 9000)} usd")
            if kind == 1:
                return (f"registry entry {_person(rng)} tax id {_make_ipn(rng)} "
                        f"phone {_make_phone(rng)}")
            if kind == 2:
                return (f"compensation set for {_person(rng)} at "
                        f"{rng.integers(900, 8000)} usd monthly")
            if kind == 3:
                return (f"reimbursement to card {_make_card(rng)} patient {_person(rng)}")
            return (f"sick leave note {_person(rng)} diagnosis "
                    f"{rng.choice(['J06', 'M54', 'K29'])}.{rng.integers(0, 9)} "
                    f"contact {_make_email(rng)}")

    def neg_insert():
        kind = rng.integers(7)
        if domain == "fin":
            if kind == 0:
                return (f"charge approved for internal test card {_serial16(rng)} "
                        f"amount {rng.integers(100, 9000)} usd")
            if kind == 1:
                return (f"record update vendor profile tax id "
                        f"{rng.integers(10000000, 99999999)} phone {_make_phone(rng)}")
            if kind == 2:
                return (f"approved rate for the {rng.choice(DEPTS)} band is "
                        f"{rng.integers(900, 8000)} usd monthly")
            if kind == 3:
                return (f"refund to card batch {_serial16(rng)} customer portal "
                        f"case {rng.integers(1000, 9999)}")
            if kind == 4:
                return (f"leave policy reference diagnosis "
                        f"{rng.choice(['J06', 'M54', 'K29'])}.{rng.integers(0, 9)} "
                        f"contact {rng.choice(DEPTS)}@corp.example.com")
            if kind == 5:
                return f"sync with {_person(rng)} about the roadmap"
            return f"hotline {_make_phone(rng)} works 9 to 18"
        else:
            if kind == 0:
                return (f"billing settled for training sandbox card {_serial16(rng)} "
                        f"copay {rng.integers(100, 9000)} usd")
            if kind == 1:
                return (f"registry entry supplier profile tax id "
                        f"{rng.integers(10000000, 99999999)} phone {_make_phone(rng)}")
            if kind == 2:
                return (f"compensation set for the nursing band at "
                        f"{rng.integers(900, 8000)} usd monthly")
            if kind == 3:
                return (f"reimbursement to card batch {_serial16(rng)} claims portal "
                        f"case {rng.integers(1000, 9999)}")
            if kind == 4:
                return (f"clinical guideline reference diagnosis "
                        f"{rng.choice(['J06', 'M54', 'K29'])}.{rng.integers(0, 9)} "
                        f"contact frontdesk@clinic.example.com")
            if kind == 5:
                return f"sync with {_person(rng)} about the ward schedule"
            return f"helpdesk {_make_phone(rng)} works 9 to 18"

    texts, labels = [], []
    n_pos = int(n * pos_share)
    for i in range(n):
        sents = body()
        if i < n_pos:
            ins = [pos_insert()]
            if rng.random() < 0.25:
                ins.append(pos_insert())
            if rng.random() < obfuscate_share:
                ins = [RE_CARDLIKE.sub(lambda m: m.group()[:-1] + "O", s)
                       for s in ins]
            labels.append(1)
        else:
            ins = [neg_insert()] if rng.random() < 0.8 else []
            labels.append(0)
        for s in ins:
            sents.insert(rng.integers(0, len(sents) + 1), s)
        texts.append(". ".join(sents))
    labels = np.array(labels)
    flip = rng.random(n) < label_noise
    labels[flip] = 1 - labels[flip]
    idx = rng.permutation(n)
    return [texts[i] for i in idx], labels[idx]


# ---------------------------------------------------------------------------
# 2. 20 Newsgroups binary (тематична конфіденційність)
# ---------------------------------------------------------------------------

def load_20ng_binary(seed=42, max_docs=4000):
    """Бінаризація 20NG: «чутливі» теми (криптографія, медицина, політика/зброя)
    проти решти. Заголовки/підписи/цитати видалено."""
    from sklearn.datasets import fetch_20newsgroups
    sensitive = ["sci.crypt", "sci.med", "talk.politics.guns"]
    rest = ["comp.graphics", "comp.sys.mac.hardware", "misc.forsale",
            "rec.autos", "rec.sport.hockey", "sci.space", "soc.religion.christian"]
    data = fetch_20newsgroups(subset="all", categories=sensitive + rest,
                              remove=("headers", "footers", "quotes"),
                              shuffle=True, random_state=seed)
    names = [data.target_names[t] for t in data.target]
    y = np.array([1 if nm in sensitive else 0 for nm in names])
    texts = [t.strip() for t in data.data]
    keep = [i for i, t in enumerate(texts) if len(t) > 80]
    rng = np.random.default_rng(seed)
    keep = rng.permutation(keep)[:max_docs]
    return [texts[i] for i in keep], y[keep]


# ---------------------------------------------------------------------------
# Ознаковий простір (спільний для всіх класифікаторів, чесне порівняння)
# ---------------------------------------------------------------------------

RE_CARDLIKE = re.compile(r"\b(?:\d[ -]?){15}\d\b")
RE_EMAIL = re.compile(r"\b[\w.]+@[\w.]+\.\w{2,}\b")
RE_PHONE = re.compile(r"\+\d{11,12}\b")
RE_IPN = re.compile(r"\b[23]\d{9}\b")
RE_MONEY = re.compile(r"\b\d{2,6} ?(?:usd|uah|eur)\b")
RE_PERSON = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
RE_DIAG = re.compile(r"\b[A-Z]\d{2}\.\d\b")
# Патерни критичних ідентифікаторів (номер соцстрахування, IBAN, крипто),
# які промислові СЗВ детектують окремими правилами.
RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
RE_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
RE_CRYPTO = re.compile(r"\b(?:0x[0-9a-fA-F]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b")
KW_SENS = ["confidential", "salary", "diagnosis", "tax id", "credentials",
           "personal", "card", "rate", "monthly"]
KW_CRED = ["password", "passwd", "pin", "iban", "ssn", "account number",
           "routing", "bitcoin", "wallet", "credential", "cvv"]


def _luhn_valid(num_str):
    digits = [c for c in num_str if c.isdigit()]
    return len(digits) == 16 and _luhn_checksum(digits) == 0


def pattern_features(texts):
    """Патернові та метаданні ознаки згідно з моделлю даних (п. 2.1).
    Карткові номери валідуються контрольною сумою Луна, як у промислових СЗВ:
    pat_card_valid: лише валідні; pat_card_like: будь-які 16-цифрові."""
    rows = []
    for t in texts:
        tl = t.lower()
        digits = sum(c.isdigit() for c in t)
        cardlike = RE_CARDLIKE.findall(t)
        rows.append([
            sum(_luhn_valid(c) for c in cardlike),
            len(cardlike),
            len(RE_EMAIL.findall(t)),
            len(RE_PHONE.findall(t)),
            len(RE_IPN.findall(t)),
            len(RE_MONEY.findall(tl)),
            len(RE_PERSON.findall(t)),
            len(RE_DIAG.findall(t)),
            len(RE_SSN.findall(t)),
            len(RE_IBAN.findall(t)),
            len(RE_CRYPTO.findall(t)),
            sum(tl.count(k) for k in KW_SENS),
            sum(tl.count(k) for k in KW_CRED),
            len(t) / 1000.0,
            digits / max(len(t), 1),
        ])
    return np.asarray(rows, dtype=np.float64)


PATTERN_NAMES = ["pat_card_valid", "pat_card_like", "pat_email", "pat_phone",
                 "pat_ipn", "pat_money", "pat_person", "pat_diag",
                 "pat_ssn", "pat_iban", "pat_crypto",
                 "kw_sensitive", "kw_credential", "meta_len", "meta_digit_ratio"]


def build_features(train_texts, test_texts, max_tfidf=300):
    """TF-IDF (top-N) + патерни + метадані -> щільні матриці."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=max_tfidf, sublinear_tf=True,
                          stop_words="english")
    Xtr_t = vec.fit_transform(train_texts).toarray()
    Xte_t = vec.transform(test_texts).toarray()
    Xtr = np.hstack([Xtr_t, pattern_features(train_texts)])
    Xte = np.hstack([Xte_t, pattern_features(test_texts)])
    names = list(vec.get_feature_names_out()) + PATTERN_NAMES
    return Xtr, Xte, names


# ---------------------------------------------------------------------------
# 3. Журнали поведінки користувачів
# ---------------------------------------------------------------------------

BEHAVIOR_FEATURES = [
    "logins", "failed_logins", "files_read", "files_written", "files_copied",
    "net_mb_out", "net_connections", "usb_writes", "prints",
    "after_hours_ratio", "unique_resources", "cloud_uploads",
]


def make_behavior_logs(n_users=40, n_days=120, anomaly_user_share=0.3,
                       seed=42):
    """Синтетичні журнали: на користувача-день вектор з 12 ознак.

    Нормальна поведінка: індивідуальні базові рівні + тижнева сезонність + шум.
    Аномалії: епізоди ексфільтрації (1-3 дні): сплески копіювань, USB,
    хмарних вивантажень, роботи поза годинами.
    Повертає X (users x days x feat), y (users x days), та межу train/test.
    """
    rng = np.random.default_rng(seed)
    F = len(BEHAVIOR_FEATURES)
    base = rng.lognormal(mean=2.0, sigma=0.5, size=(n_users, F))
    base[:, 9] = rng.uniform(0.02, 0.12, n_users)          # after_hours_ratio
    X = np.zeros((n_users, n_days, F))
    y = np.zeros((n_users, n_days), dtype=int)
    for d in range(n_days):
        weekday = d % 7
        season = 0.35 if weekday >= 5 else 1.0
        noise = rng.lognormal(0.0, 0.25, size=(n_users, F))
        X[:, d, :] = base * season * noise
        X[:, d, 9] = np.clip(base[:, 9] * rng.lognormal(0, 0.4, n_users), 0, 1)

    # епізоди ексфільтрації: «низькопрофільний» інсайдер: активність
    # піднімається до рівня, типового для активних користувачів (глобальний
    # 85-95 перцентиль), тож відносно ПОПУЛЯЦІЇ день виглядає звичайним,
    # а відносно ОСОБИСТОГО базового рівня аномальним
    exfil_feats = [4, 5, 7, 11]                # copied, net_out, usb, cloud
    glob_p90 = {j: np.percentile(X[:, :, j], 90) for j in exfil_feats}
    bad_users = rng.choice(n_users, size=int(n_users * anomaly_user_share),
                           replace=False)
    for u in bad_users:
        n_ep = rng.integers(1, 4)
        for _ in range(n_ep):
            start = rng.integers(10, n_days - 3)
            length = rng.integers(1, 4)
            for d in range(start, min(start + length, n_days)):
                for j in exfil_feats:
                    target = glob_p90[j] * rng.uniform(0.75, 1.15)
                    X[u, d, j] = max(X[u, d, j], target)
                X[u, d, 9] = min(1.0, X[u, d, 9] + rng.uniform(0.10, 0.25))
                y[u, d] = 1
    split_day = n_days // 2
    return X, y, split_day


# ---------------------------------------------------------------------------
# 4. Потік з дрейфом концепції
# ---------------------------------------------------------------------------

def make_drift_stream(n=12000, dim=10, seed=42):
    """Потік бінарної класифікації з відомими точками дрейфу.

    Дрейф реального документообігу змінює і концепцію (правила віднесення
    подій до ризикованих), і розподіл ознак (з'являються нові типи
    документів). Концепція має структуру політики СЗВ: диз'юнкцію
    кон'юнктивних умов над ознаками ризику. У точках дрейфу змінюється
    водночас набір правил концепції та середні значення частини ознак:
      - раптовий у точках n*0.25 та n*0.583 (нова політика + стрибок середніх),
      - поступовий на інтервалі [n*0.75, n*0.817] (імовірнісне змішування
        старої та нової концепцій + лінійний зсув середніх).
    Повертає X, y, список (позиція, тип).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, dim))

    def random_concept():
        """Диз'юнкція 2-3 кон'юнкцій по 2 умови на центрованих ознаках."""
        rules = []
        for _ in range(rng.integers(2, 4)):
            feats = rng.choice(dim, size=2, replace=False)
            ops = rng.choice([-1, 1], size=2)
            thrs = rng.uniform(-0.4, 0.7, size=2)
            rules.append(list(zip(feats.tolist(), ops.tolist(), thrs.tolist())))
        return rules

    def concept_apply(z, rules):
        for conds in rules:
            ok = True
            for f, op, t in conds:
                if (z[f] <= t) if op > 0 else (z[f] >= -t):
                    ok = False
                    break
            if ok:
                return 1
        return 0

    def mean_shift():
        mu = np.zeros(dim)
        feats = rng.choice(dim, size=3, replace=False)
        mu[feats] = rng.uniform(0.6, 1.2, size=3) * rng.choice([-1, 1], 3)
        return mu

    c1, c2, c3, c4 = (random_concept() for _ in range(4))
    mu1 = np.zeros(dim)
    mu2 = mu1 + mean_shift()
    mu3 = mu2 + mean_shift()
    mu4 = mu3 + mean_shift()
    p_s1, p_s2 = int(n * 0.25), int(n * 0.583)
    g_a, g_b = int(n * 0.75), int(n * 0.817)
    y = np.zeros(n, dtype=int)
    noise = rng.random(n) < 0.05                    # 5% шуму міток
    for i in range(n):
        if i < p_s1:
            c, mu = c1, mu1
        elif i < p_s2:
            c, mu = c2, mu2
        elif i < g_a:
            c, mu = c3, mu3
        elif i < g_b:
            a = (i - g_a) / (g_b - g_a)
            mu = (1 - a) * mu3 + a * mu4
            c = c4 if rng.random() < a else c3      # імовірнісне змішування
        else:
            c, mu = c4, mu4
        X[i] += mu
        y[i] = concept_apply(X[i] - mu, c)
    y[noise] = 1 - y[noise]
    drifts = [(p_s1, "sudden"), (p_s2, "sudden"), (g_a, "gradual")]
    return X, y, drifts
