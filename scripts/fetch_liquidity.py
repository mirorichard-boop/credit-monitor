#!/usr/bin/env python3
"""Liquidity regime watch: a public-data proxy for Michael Howell's (CrossBorder Capital)
global liquidity framework. Pulls FRED, Yahoo Finance and Treasury Fiscal Data, builds a
weekly panel, scores a 0-100 Liquidity Cycle Index, classifies the cycle phase, checks
tripwires, backtests forward returns by phase, and writes data/liquidity.json.

Raw downloads are cached in data/liquidity/raw/ and merged on every run, so a source that
fails one day (Yahoo is unofficial) falls back to the archive instead of blanking the page.

The proxy is ours, not CrossBorder's: their GLI covers ~90 central banks and private-sector
balance sheets. What this can see is the Fed, ECB and BoJ balance sheets, collateral
volatility (MOVE, term premium), repo plumbing (SOFR-IORB), the dollar, credit and
Treasury bill issuance. PBoC liquidity is not in any free daily/weekly series.
"""
import csv
import datetime as dt
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_fred import fetch, load_key, ssl_context  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "liquidity" / "raw"
PANEL = ROOT / "data" / "liquidity" / "panel.csv"
OUT = ROOT / "data" / "liquidity.json"
START = "2005-01-01"
Z_WINDOW = 260          # 5 years of weeks for rolling z-scores
Z_MIN = 104             # need 2 years before a component counts
BTC_LINE = 70_000       # Howell, Pinnacle Digest 2026-10-23: "bets are off" below 70k
SMOOTH = 13             # weeks of smoothing on the cycle index
MOM = 26                # weeks over which the smoothed index's direction is judged
CONFIRM = 6             # a new phase must persist this many weeks before it is adopted

FRED = ["WALCL", "WTREGEN", "RRPONTSYD", "WRESBAL", "ECBASSETSW", "JPNASSETS", "DEXUSEU",
        "DEXJPUS", "DEXCHUS", "DTWEXBGS", "THREEFYTP10", "NFCI", "VIXCLS", "DGS10", "DGS2",
        "SOFR", "IORB", "CBBTCUSD", "BAA10Y", "GFDEBTN"]
YAHOO = {"MOVE": "^MOVE", "GOLD": "GC=F", "COPPER": "HG=F", "SPX": "^GSPC"}

# Howell's model is quantity-based: MOVE and spreads are outputs, not inputs. So the cycle
# index uses balance-sheet quantities only, and the price-based gauges form a separate
# plumbing-stress score.
# key, label, weight, sign (+1 = higher reading means easier liquidity), pillar
LCI_COMPONENTS = [
    ("fed_mom",  "Fed net liquidity, 26w %",         2.0, +1, "Central bank"),
    ("g3_mom",   "G3 central bank liquidity, 52w %", 2.0, +1, "Central bank"),
    ("res_mom",  "Bank reserves, 26w %",             1.0, +1, "Central bank"),
    ("bills",    "Bill share of debt, 52w Δ",        1.0, +1, "Treasury QE"),
    ("usd_mom",  "Broad dollar, 26w %",              1.0, -1, "Cross-border"),
]
# sign +1 = higher reading means more stress
STRESS_COMPONENTS = [
    ("move",     "MOVE (Treasury vol)",              1.5, +1, "Collateral"),
    ("repo",     "SOFR − IORB",                      1.0, +1, "Plumbing"),
    ("credit",   "Baa − 10Y, 13w Δ",                 1.0, +1, "Credit"),
    ("nfci",     "Chicago Fed NFCI",                 1.0, +1, "Conditions"),
    ("vix",      "VIX",                              0.5, +1, "Equity vol"),
]

# Howell's four-phase cycle (Monetary Matters, Mar 2026): level vs. trend and direction.
PHASES = {
    "REBOUND":     "Below trend, rising. Own equities and credit; cyclical growth (tech, discretionary, financials).",
    "CALM":        "Above trend, rising. Own equities and commodities.",
    "SPECULATION": "Above trend, rolling over. Trim equities, exit credit, rotate to commodities and cyclical value; bonds bear-flatten.",
    "TURBULENCE":  "Below trend, falling. Cash on the way down, long-duration Treasuries into the trough; defensives lead.",
}


def read_raw(name):
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {r["date"]: r["value"] for r in csv.DictReader(f)}


def write_raw(name, rows):
    with (RAW_DIR / f"{name}.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["date", "value"])
        for d in sorted(rows):
            w.writerow([d, rows[d]])


def fetch_yahoo(symbol, ctx):
    p1 = int(dt.datetime.fromisoformat(START).replace(tzinfo=dt.timezone.utc).timestamp())
    qs = urllib.parse.urlencode({"period1": p1, "period2": int(time.time()), "interval": "1d"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        res = json.load(r)["chart"]["result"][0]
    closes = res["indicators"]["quote"][0]["close"]
    return {dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat(): f"{c:.4f}"
            for t, c in zip(res["timestamp"], closes) if c is not None}


def fetch_bill_share(ctx):
    """Monthly bills / total marketable debt from the Monthly Statement of the Public Debt."""
    base = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/debt/mspd/mspd_table_1"
    out, page = {}, 1
    totals, bills = {}, {}
    while True:
        qs = urllib.parse.urlencode({
            "fields": "record_date,security_type_desc,security_class_desc,total_mil_amt",
            "filter": f"record_date:gte:{START}", "page[size]": 10000, "page[number]": page})
        with urllib.request.urlopen(f"{base}?{qs}", timeout=60, context=ctx) as r:
            body = json.load(r)
        for row in body["data"]:
            d, amt = row["record_date"], row["total_mil_amt"]
            if amt in (None, "", "null"):
                continue
            if row["security_type_desc"] == "Total Marketable":
                totals[d] = float(amt)
            elif row["security_type_desc"] == "Marketable" and row["security_class_desc"] == "Bills":
                bills[d] = float(amt)
        if page >= body["meta"]["total-pages"]:
            break
        page += 1
    for d in totals:
        if d in bills and totals[d]:
            out[d] = f"{100 * bills[d] / totals[d]:.3f}"
    return out


def gather():
    key, ctx = load_key(), ssl_context()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(s, lambda s=s: fetch(s, key, ctx, start=START)) for s in FRED]
    jobs += [(k, lambda v=v: fetch_yahoo(v, ctx)) for k, v in YAHOO.items()]
    jobs += [("BILLSHARE", lambda: fetch_bill_share(ctx))]
    series, failed = {}, []
    for name, job in jobs:
        rows = read_raw(name)
        try:
            rows.update(job())
            write_raw(name, rows)
        except Exception as e:  # noqa: BLE001 - any source failure degrades to the archive
            failed.append(name)
            print(f"::warning::{name} fetch failed ({type(e).__name__}); using archive only")
        if rows:
            s = pd.Series({pd.Timestamp(d): float(v) for d, v in rows.items() if v not in (".", "")})
            series[name] = s.sort_index()
        time.sleep(0.4)
    return series, failed


def weekly(s):
    """Last observation on or before each Wednesday (H.4.1 date), forward-filled."""
    return s.resample("W-WED").last().ffill()


def rolling_z(s):
    m = s.rolling(Z_WINDOW, min_periods=Z_MIN).mean()
    sd = s.rolling(Z_WINDOW, min_periods=Z_MIN).std()
    return ((s - m) / sd).clip(-3, 3)


def build_panel(raw):
    w = {k: weekly(v) for k, v in raw.items()}
    idx = pd.date_range(START, w["WALCL"].index[-1], freq="W-WED")
    w = {k: v.reindex(idx).ffill() for k, v in w.items()}
    p = pd.DataFrame(index=idx)

    # Fed net liquidity: balance sheet less the Treasury's cash account and the RRP facility.
    p["fed_net"] = (w["WALCL"] - w["WTREGEN"] - w["RRPONTSYD"].fillna(0) * 1000) / 1e6   # $T
    p["reserves"] = w["WRESBAL"] / 1e6
    p["ecb_usd"] = w["ECBASSETSW"] * w["DEXUSEU"] / 1e6
    p["boj_usd"] = w["JPNASSETS"] / w["DEXJPUS"] / 1e4
    p["g3"] = p["fed_net"] + p["ecb_usd"] + p["boj_usd"]
    # Lookbacks chosen for a cycle of ~65 months: 13-week versions flip phase every few weeks.
    p["fed_mom"] = 100 * p["fed_net"].pct_change(26)
    p["g3_mom"] = 100 * p["g3"].pct_change(52)

    p["bill_share"] = w["BILLSHARE"]
    p["bills"] = p["bill_share"].diff(52)
    p["move"] = w["MOVE"]
    p["term_premium"] = w["THREEFYTP10"] * 100                    # bps
    p["tp_chg"] = p["term_premium"].diff(13)
    p["repo"] = (w["SOFR"] - w["IORB"]) * 100                     # bps
    p["usd"] = w["DTWEXBGS"]
    p["usd_mom"] = 100 * p["usd"].pct_change(26)
    p["usd_13"] = 100 * p["usd"].pct_change(13)
    p["cny"] = w["DEXCHUS"]
    p["baa10y"] = w["BAA10Y"] * 100
    p["credit"] = p["baa10y"].diff(13)
    p["nfci"] = w["NFCI"]
    p["vix"] = w["VIXCLS"]
    p["dgs10"] = w["DGS10"]
    p["curve"] = (w["DGS10"] - w["DGS2"]) * 100
    p["policy"] = (w["DGS2"] - w["SOFR"]) * 100
    p["btc"] = w["CBBTCUSD"]
    p["gold"] = w["GOLD"]
    p["spx"] = w["SPX"]
    p["cu_au"] = 1000 * w["COPPER"] / w["GOLD"]
    # Howell's debt-to-liquidity idea, US-only proxy: federal debt over G3 CB liquidity.
    p["debt_liq"] = (w["GFDEBTN"] / 1e6) / p["g3"]

    p["res_mom"] = 100 * p["reserves"].pct_change(26)
    p["fed_13"] = 100 * p["fed_net"].pct_change(13)
    p["lci"] = score(p, LCI_COMPONENTS)
    p["stress"] = score(p, STRESS_COMPONENTS)
    p["lci_smooth"] = p["lci"].rolling(SMOOTH, min_periods=2).mean()
    p["lci_mom"] = p["lci_smooth"].diff(MOM)
    p["signal"] = [phase_of(l, m) for l, m in zip(p["lci_smooth"], p["lci_mom"])]
    p["phase"] = confirm(p["signal"], CONFIRM)
    return p


def score(p, components):
    """Weighted mean of signed rolling z-scores, mapped to 0-100 through the normal CDF.
    The composite is divided by its own full-sample std so it spans the range; that is the
    only full-sample (look-ahead) step, and it changes scale, not ranking."""
    total_w = sum(c[2] for c in components)
    zsum = pd.Series(0.0, index=p.index)
    wsum = pd.Series(0.0, index=p.index)
    for key, _, weight, sign, _ in components:
        z = sign * rolling_z(p[key])
        p[f"z_{key}"] = z
        zsum = zsum.add(z.fillna(0) * weight)
        wsum = wsum.add(z.notna() * weight)
    comp = (zsum / wsum).where(wsum >= 0.5 * total_w)
    comp = comp / comp.std()
    return comp.apply(lambda x: 100 * 0.5 * (1 + math.erf(x / math.sqrt(2))) if pd.notna(x) else np.nan)


def confirm(signal, n):
    """Adopt a new phase only after the raw signal has shown it n weeks running."""
    out, cur, cand, cnt = [], None, None, 0
    for x in signal:
        if x is None:
            out.append(None)
            continue
        if cur is None:
            cur = x
        if x == cur:
            cand, cnt = None, 0
        else:
            cnt = cnt + 1 if x == cand else 1
            cand = x
            if cnt >= n:
                cur, cand, cnt = x, None, 0
        out.append(cur)
    return out


def phase_of(level, mom):
    if pd.isna(level) or pd.isna(mom):
        return None
    if level < 50:
        return "REBOUND" if mom > 0 else "TURBULENCE"
    return "CALM" if mom > 0 else "SPECULATION"


def curve_regime(p, weeks=13):
    d10 = (p["dgs10"].iloc[-1] - p["dgs10"].iloc[-1 - weeks]) * 100
    dc = p["curve"].iloc[-1] - p["curve"].iloc[-1 - weeks]
    kind = ("Bull" if d10 < 0 else "Bear") + (" steepener" if dc > 0 else " flattener")
    note = {
        "Bear flattener": "Long yields up, curve flatter. Howell's sequence step 1: late-cycle tightening (Speculation).",
        "Bull flattener": "Long yields down, curve flatter. Step 2: bonds rally as liquidity bottoms (Turbulence).",
        "Bull steepener": "Front end falling fastest. Step 3: steepening is his risk-on signal (Rebound).",
        "Bear steepener": "Long end selling off faster than the front: supply or term-premium pressure.",
    }[kind]
    return {"kind": kind, "d10_bps": round(d10), "dcurve_bps": round(dc), "note": note}


def tripwires(p):
    last, ago13, ago4 = p.iloc[-1], p.iloc[-14], p.iloc[-5]

    def tw(name, value, fmt, status, detail):
        return {"name": name, "value": fmt.format(value) if pd.notna(value) else "n/a",
                "status": status, "detail": detail}

    def lvl(v, watch, alert, higher_bad=True):
        if pd.isna(v):
            return "na"
        if not higher_bad:
            v, watch, alert = -v, -watch, -alert
        return "alert" if v >= alert else "watch" if v >= watch else "ok"

    fed4 = 100 * (last.fed_net / ago4.fed_net - 1)
    hi52 = p["lci_smooth"].iloc[-52:].max()
    return [
        tw("Fed net liquidity, 13w", last.fed_13, "{:+.1f}%", lvl(last.fed_13, 0, -3, higher_bad=False),
           f"${last.fed_net:.2f}T now; 4-week change {fed4:+.1f}%. Falling level is Howell's first warning."),
        tw("G3 CB liquidity, 52w", last.g3_mom, "{:+.1f}%", lvl(last.g3_mom, 0, -4, higher_bad=False),
           f"Fed + ECB + BoJ in USD: ${last.g3:.1f}T."),
        tw("MOVE index", last.move, "{:.0f}", lvl(last.move, 100, 120),
           "Treasury vol drives collateral haircuts. >100 shrinks balance-sheet capacity; >120 is stress."),
        tw("SOFR − IORB", last.repo, "{:+.0f} bp", lvl(last.repo, 0, 5),
           "Repo trading above the Fed's floor means reserves are scarce (Sept 2019 pattern)."),
        tw("2Y − SOFR", last.policy, "{:+.0f} bp", lvl(last.policy, 25, 60),
           "Howell's policy-tightness gauge: 2Y above overnight funding = market pricing more hikes."),
        tw("Baa − 10Y, 13w Δ", last.credit, "{:+.0f} bp", lvl(last.credit, 15, 40),
           f"Spread {last.baa10y:.0f} bp. Credit widens before equities in refinancing squeezes."),
        tw("Broad dollar, 13w", last.usd_13, "{:+.1f}%", lvl(last.usd_13, 2, 4),
           "A rising dollar drains offshore dollar liquidity."),
        tw("Bitcoin vs $70k", last.btc, "${:,.0f}", "alert" if last.btc < BTC_LINE else "watch" if last.btc < BTC_LINE * 1.1 else "ok",
           "Howell's liquidity bellwether; he says bets are off below $70k."),
        tw("Copper / gold, 13w", 100 * (last.cu_au / ago13.cu_au - 1), "{:+.1f}%",
           lvl(100 * (last.cu_au / ago13.cu_au - 1), -5, -12, higher_bad=False),
           "Falling ratio = real-economy demand cooling relative to monetary hedges."),
        tw("LCI off 52w high", last.lci_smooth - hi52, "{:+.0f} pts", lvl(last.lci_smooth - hi52, -10, -25, higher_bad=False),
           f"52-week high {hi52:.0f}. A peak-to-now drop is the cycle turning."),
        tw("Chicago Fed NFCI", last.nfci, "{:+.2f}", lvl(last.nfci, -0.2, 0),
           "Above zero = tighter than average financial conditions."),
    ]


def nexus(p):
    """Howell's debt-liquidity nexus (BTC Sessions, Oct 2026). Two ways it breaks:
    debt can't be turned into liquidity -> MOVE spikes, SOFR spreads widen;
    liquidity can't be turned into debt -> term premia compress, credit spreads widen."""
    last, ago = p.iloc[-1], p.iloc[-14]
    d_move, d_repo = last.move - ago.move, last.repo - ago.repo
    d_tp, d_cr = last.term_premium - ago.term_premium, last.baa10y - ago.baa10y
    a = int(d_move > 10) + int(d_repo > 3)
    b = int(d_tp < -10) + int(d_cr > 15)
    return {
        "debt_to_liq": {"score": a, "signals": [
            {"name": "MOVE, 13w Δ", "value": round(d_move), "on": bool(d_move > 10)},
            {"name": "SOFR − IORB, 13w Δ (bp)", "value": round(d_repo), "on": bool(d_repo > 3)}]},
        "liq_to_debt": {"score": b, "signals": [
            {"name": "Term premium, 13w Δ (bp)", "value": round(d_tp), "on": bool(d_tp < -10)},
            {"name": "Baa − 10Y, 13w Δ (bp)", "value": round(d_cr), "on": bool(d_cr > 15)}]},
    }


def backtest(p):
    """Forward returns by phase, using only the phase known at each week's close."""
    fwd = pd.DataFrame(index=p.index)
    for k in ("spx", "gold", "btc"):
        fwd[f"{k}_13"] = 100 * (p[k].shift(-13) / p[k] - 1)
        fwd[f"{k}_26"] = 100 * (p[k].shift(-26) / p[k] - 1)
        fwd[f"{k}_39"] = 100 * (p[k].shift(-39) / p[k] - 1)   # Howell: liquidity leads ~9 months
    fwd["ust10_13"] = -(p["dgs10"].shift(-13) - p["dgs10"]) * 100     # yield fall = bond gain
    fwd["credit_13"] = p["baa10y"].shift(-13) - p["baa10y"]
    fwd["phase"] = p["phase"]
    rows = []
    for ph in PHASES:
        g = fwd[fwd["phase"] == ph]
        r = {"phase": ph, "weeks": int(len(g)), "share": round(100 * len(g) / max(1, fwd["phase"].notna().sum()))}
        for col in ("spx_13", "spx_26", "spx_39", "gold_13", "gold_39", "btc_13", "btc_39", "ust10_13", "credit_13"):
            v = g[col].dropna()
            r[col] = None if v.empty else round(float(v.mean()), 1)
            if col in ("spx_13", "btc_13"):
                r[col.replace("_13", "_hit")] = None if v.empty else round(100 * float((v > 0).mean()))
        dd = g["spx_13"].dropna()
        r["spx_13_p10"] = None if dd.empty else round(float(dd.quantile(0.1)), 1)
        rows.append(r)
    return rows


def episodes(p):
    """Contiguous phase runs, for the timeline strip."""
    out, cur, start = [], None, None
    for d, ph in p["phase"].dropna().items():
        if ph != cur:
            if cur:
                out.append({"phase": cur, "start": start.date().isoformat(), "end": prev.date().isoformat()})
            cur, start = ph, d
        prev = d
    if cur:
        out.append({"phase": cur, "start": start.date().isoformat(), "end": prev.date().isoformat()})
    return out


def projection(p, horizon=26):
    """If the raw index stayed at this week's reading, when would the smoothed index cross 50
    and its 26-week direction flip? Gives a 'weeks until the phase changes' figure."""
    raw = p["lci"].dropna().tolist()
    sm = p["lci_smooth"].dropna().tolist()
    hold = raw[-1]
    cross = flip = None
    for k in range(1, horizon + 1):
        raw.append(hold)
        sm.append(sum(raw[-SMOOTH:]) / SMOOTH)
        level, mom = sm[-1], sm[-1] - sm[-1 - MOM]
        if cross is None and (level < 50) != (p["lci_smooth"].iloc[-1] < 50):
            cross = k
        if flip is None and (mom > 0) != (p["lci_mom"].iloc[-1] > 0):
            flip = k
    return {"hold_at": round(hold, 1), "weeks_to_cross_50": cross, "weeks_to_momentum_flip": flip,
            "signal_after": phase_of(sm[-1], sm[-1] - sm[-1 - MOM])}


def r(x, n=2):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), n)


def main():
    raw, failed = gather()
    p = build_panel(raw)
    p.round(4).to_csv(PANEL, index_label="date")
    last = p.iloc[-1]

    def comps(spec):
        return [{"key": k, "label": lab, "pillar": pil, "weight": wt, "value": r(last[k]),
                 "z": r(last[f"z_{k}"]), "sign": sg} for k, lab, wt, sg, pil in spec]

    chart_cols = ["lci", "lci_smooth", "stress", "fed_net", "g3", "reserves", "move", "term_premium", "repo",
                  "usd", "baa10y", "nfci", "bill_share", "debt_liq", "btc", "gold", "spx", "cu_au",
                  "dgs10", "curve", "cny"]
    ch = p.loc["2007-01-01":, chart_cols]
    dl = p["debt_liq"].dropna()

    out = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_through": p.index[-1].date().isoformat(),
        "lci": r(last.lci_smooth, 1), "lci_raw": r(last.lci, 1), "lci_mom": r(last.lci_mom, 1),
        "lci_4w_ago": r(p["lci_smooth"].iloc[-5], 1), "lci_13w_ago": r(p["lci_smooth"].iloc[-14], 1),
        "lci_52w_high": r(p["lci_smooth"].iloc[-52:].max(), 1),
        "phase": last.phase, "phase_desc": PHASES.get(last.phase), "phases": PHASES,
        "signal": last.signal, "signal_weeks": int(next((i for i, x in enumerate(reversed(p["signal"].tolist())) if x != last.signal), len(p))),
        "confirm_weeks": CONFIRM, "projection": projection(p),
        "phase_since": next((e["start"] for e in reversed(episodes(p)) if e["phase"] == last.phase), None),
        "headline": {
            "fed_net": r(last.fed_net), "fed_mom": r(last.fed_13, 1), "g3": r(last.g3, 1),
            "g3_mom": r(last.g3_mom, 1), "move": r(last.move, 0), "debt_liq": r(last.debt_liq),
            "debt_liq_pct": r(100 * (dl <= dl.iloc[-1]).mean(), 0), "btc": r(last.btc, 0),
        },
        "stress": r(last.stress, 1), "stress_13w_ago": r(p["stress"].iloc[-14], 1),
        "lci_components": comps(LCI_COMPONENTS), "stress_components": comps(STRESS_COMPONENTS),
        "tripwires": tripwires(p), "curve": curve_regime(p), "nexus": nexus(p),
        "backtest": backtest(p), "episodes": episodes(p), "failed": failed,
        "series": {"dates": [d.date().isoformat() for d in ch.index],
                   **{c: [r(v, 3) for v in ch[c]] for c in chart_cols}},
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"LCI {out['lci']} ({out['phase']}) through {out['data_through']} | failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
