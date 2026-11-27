#!/usr/bin/env python3
"""Pull the credit-monitor series from FRED, merge them into data/history/*.csv,
and write data/latest.json for the dashboard.

Scoring is anchored to long history. FRED serves only ~3 years of the ICE BofA OAS
indices, and those 3 years were unusually tight, so ranking ICE spreads against their own
window overstates stress. Each ICE series is instead mapped onto a Moody's spread that FRED
serves back to the 1980s (Aaa - 10Y or Baa - 10Y): take the ICE value's rank inside its own
window, read the Moody's value at that same rank over the same dates, then rank that
Moody's value against its full history. See long_run_percentile().

CCC & Below and single-B are the exception: the Moody's bridge cannot rank them (CCC moves
almost independently of Baa, and a series at its 3-year high is capped at Baa's 3-year
high). They are ranked against their own ICE history since Dec 1996 instead: a frozen
quantile table of the pre-archive years (data/longrun/, built by build_longrun.py) combined
with the live archive. See own_history_percentile().

CCC always counts in the composite at a floor weight, which rises when stress has spread
to single-B and BB (breadth_state()). A separate ladder shows how far up the rating tiers
the widening has climbed (stress_ladder()).

The FRED key comes from the FRED_API_KEY environment variable (or a local, gitignored
.env file). It is never written to disk or printed.

FRED's ICE BofA series only return the trailing ~3 years, so data/history/ is the real
archive: every run merges new observations in and never drops older rows.
"""
import bisect
import calendar
import csv
import math
import statistics
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

# id, name, group, composite weight (0-10), display unit, multiplier to display unit,
# inverse-scored, long-history bridge (None = series has its own long history)
SERIES = [
    ("BAMLC0A1CAAA",    "AAA OAS",           "ig",     2,  "bps", 100, False, "AAA10Y"),
    ("BAMLC0A2CAA",     "AA OAS",            "ig",     3,  "bps", 100, False, "AAA10Y"),
    ("BAMLC0A3CA",      "A OAS",             "ig",     4,  "bps", 100, False, "AAA10Y"),
    ("BAMLC0A4CBBB",    "BBB OAS",           "ig",     7,  "bps", 100, False, "BAA10Y"),
    ("AAA10Y",          "Moody's Aaa – 10Y", "ig",     4,  "bps", 100, False, None),
    ("BAA10Y",          "Moody's Baa – 10Y", "ig",     7,  "bps", 100, False, None),
    ("BAMLH0A0HYM2",    "HY Master II",      "hy",     6,  "bps", 100, False, "BAA10Y"),
    ("BAMLH0A1HYBB",    "BB OAS",            "hy",     7,  "bps", 100, False, "BAA10Y"),
    ("BAMLH0A2HYB",     "B OAS",             "hy",     7,  "bps", 100, False, None),
    ("BAMLH0A3HYC",     "CCC & Below",       "hy",     3,  "bps", 100, False, None),
    ("T10Y2Y",          "10Y – 2Y",          "curve",  6,  "bps", 100, True,  None),
    ("T10Y3M",          "10Y – 3M",          "curve",  8,  "bps", 100, True,  None),
    ("T10YFF",          "10Y – FedFunds",    "curve",  5,  "bps", 100, True,  None),
    ("BAMLEMCBPIOAS",   "EM Corp OAS",       "global", 6,  "bps", 100, False, "BAA10Y"),
    ("BAMLHE00EHYIOAS", "Euro HY OAS",       "global", 5,  "bps", 100, False, "BAA10Y"),
    ("MORTGAGE30US",    "30Y Mortgage",      "global", 5,  "%",   1,   False, None),
    ("BAMLC0A0CM",      "US Corp Master",    "global", 6,  "bps", 100, False, "BAA10Y"),
]

LONGRUN_DIR = ROOT / "data" / "longrun"

# CCC distress gauge. CCC counts at its floor weight (SERIES above) and steps up once the
# widening has spread to single-B and BB.
CCC, SINGLE_B, BB, HY = "BAMLH0A3HYC", "BAMLH0A2HYB", "BAMLH0A1HYBB", "BAMLH0A0HYM2"
BBB, IG = "BAMLC0A4CBBB", "BAMLC0A0CM"
CCC_STRESSED_PCT = 75       # CCC long-run percentile at/above this = CCC is stressed
CCC_GAP_ALERT_PCT = 80      # or CCC-B gap long-run percentile at/above this
CCC_CONFIRMED_WEIGHT = 6    # weight CCC gets once breadth confirms
# Breadth: spreads scale with their level, so compare tiers in percent, not bps.
BREADTH_DAYS = 91           # look-back for the widening
BREADTH_ENTER = 0.25        # B and BB both up >= 25% (log) ...
BREADTH_ENTER_HOLD = 5      # ... for 5 straight trading days confirms
BREADTH_EXIT = 0.15         # confirmation ends once B and BB are both under +15% ...
BREADTH_EXIT_HOLD = 10      # ... for 10 straight trading days
# Stress ladder: a rung has turned when its 5-day average sits this many bps above its
# low of the past year. Thresholds scale roughly with each tier's typical spread.
LADDER = [(CCC, "CCC", 200), (SINGLE_B, "B", 100), (BB, "BB", 75), (BBB, "BBB", 50), (IG, "IG Corp", 40)]
LADDER_LOOKBACK_DAYS = 365

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


def fetch(series_id, key, ctx, start=None):
    """Return {date: value_string} for every non-missing observation FRED serves."""
    params = {"series_id": series_id, "api_key": key, "file_type": "json",
              "sort_order": "asc", "limit": 100000}
    if start:
        params["observation_start"] = start
    qs = urllib.parse.urlencode(params)
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


def share_at_or_below(sorted_vals, x):
    return bisect.bisect_right(sorted_vals, x) / len(sorted_vals)


def long_run_percentile(cur, own_vals, bridge_rows, start):
    """Translate a short-window value into a long-history percentile via a bridge series.

    1. q = share of the series' own archive at or below today's value.
    2. v = the bridge's value at quantile q over the same dates as the archive.
       If today's value is outside the archive's range, extend linearly past the edge,
       scaling by the ratio of the two series' standard deviations in the window.
    3. Return the share of the bridge's full history at or below v.
    """
    own = sorted(own_vals)
    window = sorted(v for d, v in bridge_rows.items() if d >= start)
    full = sorted(bridge_rows.values())
    if len(window) < 30:
        return None
    if cur > own[-1] or cur < own[0]:
        scale = statistics.pstdev(window) / (statistics.pstdev(own) or 1)
        edge_own, edge_b = (own[-1], window[-1]) if cur > own[-1] else (own[0], window[0])
        v = edge_b + (cur - edge_own) * scale
    else:
        q = share_at_or_below(own, cur)
        v = window[min(len(window) - 1, max(0, math.ceil(q * len(window)) - 1))]
    return round(100 * share_at_or_below(full, v)), round(v, 4)


def load_longrun(measure):
    path = LONGRUN_DIR / f"{measure}.json"
    return json.loads(path.read_text()) if path.exists() else None


def own_history_percentile(cur, archive_vals, table):
    """Rank cur against frozen pre-archive quantiles plus the live archive, weighted by
    observation count. Archive values dated before table['archive_from'] are not double
    counted because the table stops where the archive starts."""
    q = table["quantiles"]
    frozen_share = bisect.bisect_right(q, cur) / len(q)
    n_old, n_new = table["n"], len(archive_vals)
    at_or_below = n_old * frozen_share + sum(1 for v in archive_vals if v <= cur)
    return round(100 * at_or_below / (n_old + n_new))


def summarize(sid, name, group, weight, unit, mult, invert, bridge, rows, bridges):
    obs = [(dt.date.fromisoformat(d), float(v)) for d, v in sorted(rows.items())]
    last_date, cur = obs[-1]
    vals = [v for _, v in obs]
    # Window percentile: share of this series' own archive <= today's value.
    pct = round(100 * sum(1 for v in vals if v <= cur) / len(vals))
    lr, bridge_value = pct, None
    own = load_longrun(sid)
    if own:
        lr, bridge = own_history_percentile(cur, vals, own), None
    elif bridge and bridges.get(bridge):
        res = long_run_percentile(cur, vals, bridges[bridge], obs[0][0].isoformat())
        if res:
            lr, bridge_value = res
    else:
        bridge = None
    stress = 100 - lr if invert else lr

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
        "percentile": pct, "long_run_percentile": lr, "stress": stress,
        "bridge": bridge, "bridge_value": None if bridge_value is None else round(bridge_value * mult, 1),
        "bridge_range_start": min(bridges[bridge]) if bridge else None,
        "own_history_start": own["start"] if own else None,
        "own_history_n": own["n"] + len(vals) if own else None,
        "range_start": first.isoformat(), "range_end": last_date.isoformat(), "n_obs": len(obs),
        "range_years": round((last_date - first).days / 365.25, 1),
    }


def series_obs(hist, sid):
    return [(dt.date.fromisoformat(d), float(v) * 100) for d, v in sorted(hist[sid].items())]


def log_change(dates, vals, i, days):
    """log(v_i / v_j), j = last observation dated <= dates[i] - days (None if absent/stale)."""
    target = dates[i] - dt.timedelta(days=days)
    j = bisect.bisect_right(dates, target) - 1
    if j < 0 or (dates[i] - dates[j]).days > days + 10:
        return None
    return math.log(vals[i] / vals[j])


def breadth_state(hist):
    """Replay the breadth confirmation day by day over the archive, with hysteresis.

    Enters when single-B and BB have both widened >= BREADTH_ENTER over BREADTH_DAYS for
    BREADTH_ENTER_HOLD straight days; exits after BREADTH_EXIT_HOLD straight days with both
    under BREADTH_EXIT. Replaying from the start keeps the state deterministic without a
    state file.
    """
    b, bb = dict(series_obs(hist, SINGLE_B)), dict(series_obs(hist, BB))
    dates = sorted(set(b) & set(bb))
    bv, bbv = [b[d] for d in dates], [bb[d] for d in dates]
    confirmed, enter_run, exit_run, since = False, 0, 0, None
    b3 = bb3 = None
    for i in range(len(dates)):
        b3, bb3 = log_change(dates, bv, i, BREADTH_DAYS), log_change(dates, bbv, i, BREADTH_DAYS)
        if b3 is None or bb3 is None:
            continue
        if not confirmed:
            enter_run = enter_run + 1 if b3 >= BREADTH_ENTER and bb3 >= BREADTH_ENTER else 0
            if enter_run >= BREADTH_ENTER_HOLD:
                confirmed, since, exit_run = True, dates[i].isoformat(), 0
        else:
            exit_run = exit_run + 1 if b3 < BREADTH_EXIT and bb3 < BREADTH_EXIT else 0
            if exit_run >= BREADTH_EXIT_HOLD:
                confirmed, since, enter_run = False, None, 0
    pct = lambda x: None if x is None else round(100 * x, 1)
    return {"confirmed": confirmed, "since": since, "b_3m_pct": pct(b3), "bb_3m_pct": pct(bb3),
            "enter_run": 0 if confirmed else enter_run, "exit_run": exit_run if confirmed else 0,
            "date": dates[-1].isoformat()}


def stress_ladder(hist):
    """How far up the rating tiers the widening has climbed."""
    rungs = []
    for sid, label, threshold in LADDER:
        if sid not in hist:
            continue
        obs = series_obs(hist, sid)
        dates, vals = [d for d, _ in obs], [v for _, v in obs]
        ma = [sum(vals[max(0, i - 4):i + 1]) / (i + 1 - max(0, i - 4)) for i in range(len(vals))]
        cutoff = dates[-1] - dt.timedelta(days=LADDER_LOOKBACK_DAYS)
        start = bisect.bisect_left(dates, cutoff)
        t = min(range(start, len(ma)), key=lambda i: ma[i])
        rise = ma[-1] - ma[t]
        turned_on = next((dates[i].isoformat() for i in range(t, len(ma)) if ma[i] - ma[t] >= threshold), None)
        rungs.append({"id": sid, "label": label, "value": round(vals[-1]), "low": round(ma[t]),
                      "low_date": dates[t].isoformat(), "rise_bps": round(rise), "threshold_bps": threshold,
                      "turned": rise >= threshold, "turned_on": turned_on if rise >= threshold else None})
    height = 0
    for r in rungs:
        if not r["turned"]:
            break
        height += 1
    return {"rungs": rungs, "height": height, "lookback_days": LADDER_LOOKBACK_DAYS}


def ccc_distress(out, hist):
    """CCC level and gap against 1996+ history, breadth confirmation, composite weight."""
    by = {s["id"]: s for s in out}
    if any(sid not in hist or sid not in by for sid in (CCC, SINGLE_B, BB)):
        return None
    c, b, bb = hist[CCC], hist[SINGLE_B], hist[BB]
    common = sorted(set(c) & set(b) & set(bb))
    gaps = [(float(c[d]) - float(b[d])) * 100 for d in common]
    ratios = [float(c[d]) / float(bb[d]) for d in common]
    gap, ratio = gaps[-1], ratios[-1]
    gap_tab, ratio_tab = load_longrun("CCC_B_GAP"), load_longrun("CCC_BB_RATIO")
    gap_pct = own_history_percentile(gap, gaps, gap_tab) if gap_tab else round(100 * sum(g <= gap for g in gaps) / len(gaps))
    ratio_pct = own_history_percentile(ratio, ratios, ratio_tab) if ratio_tab else None
    prior = [g for d, g in zip(common, gaps) if d <= (dt.date.fromisoformat(common[-1]) - dt.timedelta(days=30)).isoformat()]
    gap_1m = round(gap - prior[-1]) if prior else None

    br = breadth_state(hist)
    ccc_lr = by[CCC]["long_run_percentile"]
    stressed = ccc_lr >= CCC_STRESSED_PCT or gap_pct >= CCC_GAP_ALERT_PCT
    base_w = by[CCC]["weight"]
    if br["confirmed"]:
        state, weight = "CONFIRMED", CCC_CONFIRMED_WEIGHT
        note = (f"Single-B and BB have widened as fast as CCC (B {br['b_3m_pct']:+.0f}%, BB {br['bb_3m_pct']:+.0f}% "
                f"over 3 months) since {br['since']}. CCC counts at weight {CCC_CONFIRMED_WEIGHT}.")
    elif br["enter_run"] > 0:
        state, weight = "BUILDING", base_w
        note = (f"B and BB are both up at least {BREADTH_ENTER:.0%} over 3 months, day {br['enter_run']} of "
                f"{BREADTH_ENTER_HOLD} needed to confirm. CCC counts at its floor weight {base_w} until then.")
    elif stressed:
        state, weight = "ISOLATED", base_w
        note = (f"CCC is wide against its 1996+ history but single-B and BB have not widened at the same pace. "
                f"CCC counts at its floor weight {base_w}.")
    else:
        state, weight = "QUIET", base_w
        note = f"CCC is not unusually wide against its 1996+ history. It counts at its floor weight {base_w}."

    fmt = lambda x: None if x is None else round(x, 1)
    confirms = [
        {"label": "Single-B 3-month widening", "value": fmt(br["b_3m_pct"]), "threshold": round(100 * BREADTH_ENTER), "unit": "%",
         "hit": br["b_3m_pct"] is not None and br["b_3m_pct"] >= 100 * BREADTH_ENTER},
        {"label": "BB 3-month widening", "value": fmt(br["bb_3m_pct"]), "threshold": round(100 * BREADTH_ENTER), "unit": "%",
         "hit": br["bb_3m_pct"] is not None and br["bb_3m_pct"] >= 100 * BREADTH_ENTER},
    ]
    return {"state": state, "note": note, "weight": weight, "base_weight": base_w, "confirmed_weight": CCC_CONFIRMED_WEIGHT,
            "ccc_long_run_percentile": ccc_lr, "ccc_stressed_pct": CCC_STRESSED_PCT,
            "gap_bps": round(gap), "gap_1m_bps": gap_1m, "gap_percentile": gap_pct, "gap_alert_pct": CCC_GAP_ALERT_PCT,
            "ccc_bb_ratio": round(ratio, 2), "ccc_bb_ratio_percentile": ratio_pct,
            "history_start": gap_tab["start"] if gap_tab else common[0],
            "breadth": {**br, "enter_pct": round(100 * BREADTH_ENTER), "enter_hold": BREADTH_ENTER_HOLD,
                        "exit_pct": round(100 * BREADTH_EXIT), "exit_hold": BREADTH_EXIT_HOLD},
            "confirms": confirms, "date": common[-1]}


def main():
    key = load_key()
    ctx = ssl_context()
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    failed, hist = [], {}

    for sid, *_ in SERIES:
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
            hist[sid] = rows
        time.sleep(0.6)  # stay well under FRED's 120 requests/minute

    if not hist:
        sys.exit("no data for any series")

    bridges = {sid: {d: float(v) for d, v in rows.items()} for sid, rows in hist.items()
               if sid in {m[-1] for _, *m in SERIES if m[-1]}}
    out = [summarize(sid, *meta, hist[sid], bridges) for sid, *meta in SERIES if sid in hist]

    distress = ccc_distress(out, hist)
    if distress:
        next(s for s in out if s["id"] == CCC)["weight"] = distress["weight"]
    ladder = stress_ladder(hist)
    for s in out:
        s["in_composite"] = s["weight"] > 0

    scored = [s for s in out if s["weight"] > 0]
    total_w = sum(s["weight"] for s in scored)
    composite = round(sum(s["stress"] * s["weight"] for s in scored) / total_w)
    # Same weights, scored against each series' own short window: what the old method said.
    window_stress = lambda s: 100 - s["percentile"] if s["invert"] else s["percentile"]
    composite_window = round(sum(window_stress(s) * s["weight"] for s in scored) / total_w)
    groups = {}
    for g in ("ig", "hy", "curve", "global"):
        members = [s for s in scored if s["group"] == g]
        if members:
            avg = round(sum(s["stress"] * s["weight"] for s in members) / sum(s["weight"] for s in members))
            groups[g] = {"stress": avg, "zone": zone(avg)}

    latest = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_through": max(s["date"] for s in out),
        "composite": composite, "zone": zone(composite),
        "composite_window": composite_window, "zone_window": zone(composite_window),
        "zones": [{"max": top, "label": label} for top, label in ZONES],
        "groups": groups, "ccc_distress": distress, "ladder": ladder, "failed": failed, "series": out,
    }
    LATEST.write_text(json.dumps(latest, indent=1, ensure_ascii=False) + "\n")
    print(f"composite {composite} {zone(composite)} (window-ranked {composite_window}) | "
          f"CCC {distress['state'] if distress else 'n/a'} | ladder {ladder['height']}/{len(ladder['rungs'])} | {len(out) - len(failed)}/{len(SERIES)} fetched")
    sys.exit(1 if len(failed) == len(SERIES) else 0)


if __name__ == "__main__":
    main()
