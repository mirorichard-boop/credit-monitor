#!/usr/bin/env python3
"""One-off: freeze the pre-archive ICE BofA history into quantile tables.

FRED now serves only ~3 years of ICE BofA OAS data, so the dashboard's archive
(data/history/) starts in Sep 2023. Older daily history (Dec 1996 onward) survives only
in downloads made before FRED cut it back. This script reads such a file and writes,
per measure, 1,001 quantile breakpoints of every observation dated before the archive
begins. fetch_fred.py combines that frozen distribution with the live archive to rank
today's value against the full 1996+ history.

Only the derived breakpoints are committed. The raw file stays outside the repo: ICE's
terms prohibit reproducing the underlying data.

Usage: python3 scripts/build_longrun.py path/to/merged.csv
The input needs a date column first, then columns named CCC, B, BB, in bps.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "longrun"
N_QUANTILES = 1001

# measure -> (function of a row in bps, archive series that marks where the archive takes over)
MEASURES = {
    "BAMLH0A3HYC": (lambda r: r["CCC"] / 100, "BAMLH0A3HYC"),          # percent, like the archive
    "BAMLH0A2HYB": (lambda r: r["B"] / 100, "BAMLH0A2HYB"),
    "CCC_B_GAP": (lambda r: r["CCC"] - r["B"], "BAMLH0A3HYC"),           # bps
    "CCC_BB_RATIO": (lambda r: r["CCC"] / r["BB"], "BAMLH0A3HYC"),       # x
}


def archive_start(sid):
    with (ROOT / "data" / "history" / f"{sid}.csv").open() as f:
        next(f)
        return next(f).split(",")[0]


def quantiles(vals):
    vals = sorted(vals)
    n = len(vals)
    return [round(vals[min(n - 1, round(i * (n - 1) / (N_QUANTILES - 1)))], 4) for i in range(N_QUANTILES)]


def main():
    src = Path(sys.argv[1])
    with src.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for raw in reader:
            rec = {"date": raw[0]}
            for name, cell in zip(header[1:], raw[1:]):
                try:
                    rec[name] = float(cell)
                except ValueError:
                    rec[name] = None
            rows.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    for measure, (fn, anchor) in MEASURES.items():
        cutoff = archive_start(anchor)
        pts = []
        for r in rows:
            if r["date"] >= cutoff:
                continue
            try:
                pts.append((r["date"], fn(r)))
            except (TypeError, ZeroDivisionError):
                continue
        table = {
            "measure": measure,
            "source": "ICE BofA US HY OAS indices via pre-2023 FRED downloads; derived quantiles only",
            "start": pts[0][0], "end": pts[-1][0], "n": len(pts),
            "archive_from": cutoff, "quantiles": quantiles([v for _, v in pts]),
        }
        (OUT / f"{measure}.json").write_text(json.dumps(table, separators=(",", ":")) + "\n")
        print(f"{measure:13} {table['start']} → {table['end']}  n={table['n']}  median={table['quantiles'][500]}")


if __name__ == "__main__":
    main()
