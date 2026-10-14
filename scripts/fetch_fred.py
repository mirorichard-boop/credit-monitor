#!/usr/bin/env python3
"""Pull the 15 credit-monitor series from FRED, merge them into data/history/*.csv,
and write data/latest.json for the dashboard.

The FRED key comes from the FRED_API_KEY environment variable (or a local, gitignored
.env file). It is never written to disk or printed.

FRED's ICE BofA series only return the trailing ~3 years, so data/history/ is the real
archive: every run merges new observations in and never drops older rows.
"""
import calendar
import csv
import datetime as dt
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HIST_DIR = ROOT / "data" / "history"
LATEST = ROOT / "data" / "latest.json"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

# id, name, group, sensitivity weight (1-10), display unit, multiplier to display unit, inverse-scored
SERIES = [
    ("BAMLC0A1CAAA",    "AAA OAS",         "ig",     3,  "bps", 100, False),
    ("BAMLC0A2CAA",     "AA OAS",          "ig",     4,  "bps", 100, False),
    ("BAMLC0A3CA",      "A OAS",           "ig",     5,  "bps", 100, False),
    ("BAMLC0A4CBBB",    "BBB OAS",         "ig",     7,  "bps", 100, False),
    ("BAMLH0A0HYM2",    "HY Master II",    "hy",     7,  "bps", 100, False),
    ("BAMLH0A1HYBB",    "BB OAS",          "hy",     6,  "bps", 100, False),
    ("BAMLH0A2HYB",     "B OAS",           "hy",     8,  "bps", 100, False),
    ("BAMLH0A3HYC",     "CCC & Below",     "hy",     10, "bps", 100, False),
    ("T10Y2Y",          "10Y – 2Y",        "curve",  6,  "bps", 100, True),
    ("T10Y3M",          "10Y – 3M",        "curve",  8,  "bps", 100, True),
    ("T10YFF",          "10Y – FedFunds",  "curve",  5,  "bps", 100, True),
    ("BAMLEMCBPIOAS",   "EM Corp OAS",     "global", 6,  "bps", 100, False),
    ("BAMLHE00EHYIOAS", "Euro HY OAS",     "global", 5,  "bps", 100, False),
    ("MORTGAGE30US",    "30Y Mortgage",    "global", 5,  "%",   1,   False),
    ("BAMLC0A0CM",      "US Corp Master",  "global", 6,  "bps", 100, False),
]

# Upper bound (inclusive) of each zone on the 0-100 stress scale.
ZONES = [(25, "CLEAR"), (50, "WATCH"), (75, "CAUTION"), (100, "DANGER")]


def load_key():
    key = os.environ.get("FRED_API_KEY", "").strip()
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            name, _, val = line.partition("=")
            if name.strip() == "FRED_API_KEY":
                key = val.strip().strip("'\"")
    if not key:
        sys.exit("FRED_API_KEY is not set (export it, or put FRED_API_KEY=... in .env)")
    return key


def ssl_context():
    # python.org macOS builds ship without a CA bundle; certifi fixes that when present.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch(series_id, key, ctx):
    """Return {date: value_string} for every non-missing observation FRED serves."""
    qs = urllib.parse.urlencode({"series_id": series_id, "api_key": key, "file_type": "json",
                                 "sort_order": "asc", "limit": 100000})
    last_err = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{FRED_URL}?{qs}", timeout=60, context=ctx) as r:
                body = json.load(r)
            return {o["date"]: o["value"] for o in body["observations"] if o["value"] not in (".", "")}
        except urllib.error.HTTPError as e:
            # 4xx other than rate limiting won't fix itself on retry (bad key, bad id).
            last_err = f"HTTP {e.code}"
            if e.code < 500 and e.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as e:
            last_err = type(e).__name__
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(last_err)  # deliberately carries no URL, so the key can't leak into logs


def read_history(series_id):
    path = HIST_DIR / f"{series_id}.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["date"]: row["value"] for row in csv.DictReader(f) if row["value"] not in (".", "")}


def write_history(series_id, rows):
    with (HIST_DIR / f"{series_id}.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["date", "value"])
        for d in sorted(rows):
            w.writerow([d, rows[d]])


def value_on_or_before(obs, target):
    """obs: ascending list of (date, float). Latest observation dated <= target."""
    best = None
    for d, v in obs:
        if d > target:
            break
        best = v
    return best


def zone(score):
    return next(label for top, label in ZONES if score <= top)


def summarize(sid, name, group, weight, unit, mult, invert, rows):
    obs = [(dt.date.fromisoformat(d), float(v)) for d, v in sorted(rows.items())]
    last_date, cur = obs[-1]
    vals = [v for _, v in obs]
    # Same definition as the original dashboard: share of observations <= today's value.
    pct = round(100 * sum(1 for v in vals if v <= cur) / len(vals))
    stress = 100 - pct if invert else pct

    def change(days=0, months=0):
        if months:
            y, m = divmod(last_date.month - 1 - months, 12)
            year, month = last_date.year + y, m + 1
            target = dt.date(year, month, min(last_date.day, calendar.monthrange(year, month)[1]))
        else:
            target = last_date - dt.timedelta(days=days)
        prior = value_on_or_before(obs, target)
        return None if prior is None else round((cur - prior) * mult, 2)

    first = obs[0][0]
    return {
        "id": sid, "name": name, "group": group, "weight": weight, "unit": unit, "invert": invert,
        "value": round(cur * mult, 2), "date": last_date.isoformat(),
        "chg_1w": change(days=7), "chg_1m": change(months=1), "chg_1y": change(months=12),
        "percentile": pct, "stress": stress,
        "range_start": first.isoformat(), "range_end": last_date.isoformat(), "n_obs": len(obs),
        "range_years": round((last_date - first).days / 365.25, 1),
    }


def main():
    key = load_key()
    ctx = ssl_context()
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    out, failed = [], []

    for sid, *meta in SERIES:
        rows = read_history(sid)
        before = len(rows)
        try:
            rows.update(fetch(sid, key, ctx))  # FRED revisions overwrite archived values
            write_history(sid, rows)
            print(f"{sid:16} {len(rows):6} rows (+{len(rows) - before})")
        except RuntimeError as e:
            failed.append(sid)
            print(f"::warning::{sid} fetch failed ({e}); using archive only")
        if rows:
            out.append(summarize(sid, *meta, rows))
        time.sleep(0.6)  # stay well under FRED's 120 requests/minute

    if not out:
        sys.exit("no data for any series")

    total_w = sum(s["weight"] for s in out)
    composite = round(sum(s["stress"] * s["weight"] for s in out) / total_w)
    groups = {}
    for g in ("ig", "hy", "curve", "global"):
        members = [s["stress"] for s in out if s["group"] == g]
        if members:
            avg = round(sum(members) / len(members))
            groups[g] = {"stress": avg, "zone": zone(avg)}

    latest = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_through": max(s["date"] for s in out),
        "composite": composite, "zone": zone(composite),
        "zones": [{"max": top, "label": label} for top, label in ZONES],
        "groups": groups, "failed": failed, "series": out,
    }
    LATEST.write_text(json.dumps(latest, indent=1, ensure_ascii=False) + "\n")
    print(f"composite {composite} {zone(composite)} | {len(out) - len(failed)}/{len(SERIES)} fetched")
    sys.exit(1 if len(failed) == len(SERIES) else 0)


if __name__ == "__main__":
    main()
