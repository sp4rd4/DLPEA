"""Побудова ознак «користувач-день» з логів CERT Insider Threat r4.2
(Lindauer et al., 2020) потоковим проходом (низька пам'ять). Мітки днів
формуються з answers/insiders.csv: день користувача вважається шкідливим,
якщо потрапляє у вікно [start; end] відомого інсайдерського сценарію.

Джерело даних:
Lindauer B. Insider Threat Test Dataset. Carnegie Mellon University, 2020.
DOI: 10.1184/R1/12841247.v1.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
from collections import defaultdict

import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "cert")
R42 = os.path.join(BASE, "r4.2")
NEED = os.path.join(BASE, "need")
ANS = os.path.join(BASE, "answers_dir", "answers")

FMT = "%m/%d/%Y %H:%M:%S"


def _parse(ts):
    return dt.datetime.strptime(ts, FMT)


def _find(name):
    for d in (R42, NEED):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def malicious_user_days():
    """Множина (user, date) шкідливих днів з insiders.csv для r4.2."""
    bad = set()
    with open(os.path.join(ANS, "insiders.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] != "4.2":
                continue
            u = row["user"]
            s = _parse(row["start"]).date()
            e = _parse(row["end"]).date()
            d = s
            while d <= e:
                bad.add((u, d))
                d += dt.timedelta(days=1)
    return bad


def build(max_rows=None):
    feats = defaultdict(lambda: defaultdict(float))

    def add(key, field, v=1.0):
        feats[key][field] += v

    # logon.csv: id,date,user,pc,activity(Logon/Logoff)
    p = _find("logon.csv")
    if p:
        with open(p, encoding="utf-8") as f:
            r = csv.reader(f)
            next(r, None)
            for i, row in enumerate(r):
                if max_rows and i > max_rows:
                    break
                _, date, user, pc, act = row[:5]
                t = _parse(date)
                key = (user, t.date())
                if act == "Logon":
                    add(key, "logons")
                    if t.hour < 7 or t.hour >= 19:
                        add(key, "logon_afterhours")
                    if t.weekday() >= 5:
                        add(key, "logon_weekend")
                feats[key].setdefault("pc:" + pc, 0.0)
                feats[key]["pc:" + pc] = 1.0

    # device.csv: id,date,user,pc,activity(Connect/Disconnect)
    p = _find("device.csv")
    if p:
        with open(p, encoding="utf-8") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                _, date, user, pc, act = row[:5]
                t = _parse(date)
                key = (user, t.date())
                if act == "Connect":
                    add(key, "usb_connects")
                    if t.hour < 7 or t.hour >= 19:
                        add(key, "usb_afterhours")

    # file.csv: id,date,user,pc,filename,content  (копіювання на знімний носій)
    p = _find("file.csv")
    if p:
        with open(p, encoding="utf-8") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) < 5:
                    continue
                date, user = row[1], row[2]
                t = _parse(date)
                key = (user, t.date())
                add(key, "file_copies")
                fn = row[4].lower()
                if fn.endswith((".exe", ".zip", ".rar", ".7z")):
                    add(key, "file_exe_zip")

    # email.csv: id,date,user,pc,to,cc,bcc,from,size,attachments,content
    p = _find("email.csv")
    if p:
        with open(p, encoding="utf-8") as f:
            r = csv.reader(f)
            hdr = next(r, None)
            for row in r:
                if len(row) < 9:
                    continue
                date, user = row[1], row[2]
                t = _parse(date)
                key = (user, t.date())
                add(key, "emails")
                recips = (row[4] or "") + ";" + (row[5] or "") + ";" + (row[6] or "")
                ext = sum(1 for a in recips.split(";")
                          if a and "dtaa.com" not in a.lower())
                if ext:
                    add(key, "email_external")
                try:
                    add(key, "email_bytes", float(row[8]))
                except (ValueError, IndexError):
                    pass

    # згорнути pc:* у кількість унікальних ПК
    rows = []
    bad = malicious_user_days()
    for (user, date), d in feats.items():
        n_pc = sum(1 for k in d if k.startswith("pc:"))
        rec = {
            "user": user, "date": date.isoformat(),
            "logons": d.get("logons", 0.0),
            "logon_afterhours": d.get("logon_afterhours", 0.0),
            "logon_weekend": d.get("logon_weekend", 0.0),
            "n_pc": float(n_pc),
            "usb_connects": d.get("usb_connects", 0.0),
            "usb_afterhours": d.get("usb_afterhours", 0.0),
            "file_copies": d.get("file_copies", 0.0),
            "file_exe_zip": d.get("file_exe_zip", 0.0),
            "emails": d.get("emails", 0.0),
            "email_external": d.get("email_external", 0.0),
            "email_bytes": d.get("email_bytes", 0.0),
            "label": 1 if (user, date) in bad else 0,
        }
        rows.append(rec)
    return rows


FEATURE_COLS = ["logons", "logon_afterhours", "logon_weekend", "n_pc",
                "usb_connects", "usb_afterhours", "file_copies", "file_exe_zip",
                "emails", "email_external", "email_bytes"]


def to_arrays(rows):
    users = sorted(set(r["user"] for r in rows))
    rows = sorted(rows, key=lambda r: (r["user"], r["date"]))
    X = np.array([[r[c] for c in FEATURE_COLS] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows], dtype=int)
    u = np.array([r["user"] for r in rows])
    return X, y, u, rows


if __name__ == "__main__":
    rows = build()
    X, y, u, _ = to_arrays(rows)
    os.makedirs("results", exist_ok=True)
    with open("results/cert_userdays.json", "w", encoding="utf-8") as f:
        json.dump({"n_userdays": len(rows), "n_malicious": int(y.sum()),
                   "n_users": len(set(u)), "features": FEATURE_COLS}, f,
                  ensure_ascii=False, indent=2)
    print(f"user-days: {len(rows)}, malicious: {int(y.sum())}, users: {len(set(u))}")
