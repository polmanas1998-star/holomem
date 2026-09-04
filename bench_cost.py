# -*- coding: utf-8 -*-
"""
bench_cost.py : what the trace costs in time and bytes as N grows.

    python bench_cost.py                    # 3 passes, ~5 minutes
    python bench_cost.py --counts 250,1000  # skip the slow point

Section 6 of the README is produced by this file. It used to be a table with
no code behind it, which was the wrong way round for a repository whose whole
argument is that the numbers are checkable. It is also the table that was
published wrong once, so it is the one that most needed a script.

WHY THIS FILE EXISTS AT ALL, AND WHAT IT MEASURES
─────────────────────────────────────────────────
`learn()` scans the whole fact list for a duplicate before appending, so
inserting N facts is quadratic. `_build()` recomputes every weight from the
current time, so the trace is rebuilt rather than patched. Neither is
amortised. Both are deliberate (a weight that depends on `now` cannot be
cached), and both are the reason "fixed size" describes the trace and never
the system.

Four columns:

  insert total   building the store from nothing, N calls to learn().
  rebuild        one _build() over an existing fact list.
  query          one query(), averaged over several, including the cleanup
                 scan over the candidate pool.
  bytes          the fact list, recursively, against the trace, exactly.

HOW IT REPORTS, AND WHY IT REPORTS THAT WAY
───────────────────────────────────────────
Three passes, median printed, min and max printed next to it. Never one pass.

This is not statistical decoration. The first version of this table was
measured while another benchmark was still running on the same machine, and
the contention hit the shortest point hardest: only the N=250 point was badly
wrong, and because that point is the denominator, the published ratio came out
at 619x. It was corrected to 194x by hand, and that correction was itself
taken from a script that was never committed and does not reproduce here.

Which is the lesson, and it is worth more than either number. A ratio between
two measured times is a fact about one laptop on one afternoon. Quote the
LOCAL EXPONENT instead: it is scale free, so a machine twice as fast leaves it
untouched, and it says the thing you actually wanted to say about the
algorithm. This file prints the exponent ladder for that reason, and prints
both ends of any ratio so a reader can recompute it.

A structurally quadratic loop that measures below exponent 2.0 is a warning,
not a triumph: it means the small-N end was inflated. That is precisely the
signature the 619x had, and the 194x still had.

Before you trust a number out of here, check that nothing else is running on
the machine. The script cannot check that for you, which is exactly how it
went wrong the first time.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

from holomem import HolographicMemory

DIM = 1024
COUNTS = (250, 500, 1000, 2000, 4000)
QUERIES_PER_PASS = 10


def deep_sizeof(obj, seen: set[int] | None = None) -> int:
    """Recursive sys.getsizeof, counting each object once.

    Implementation-dependent by construction: a Python object header is not a
    fact about FHRR. The number is here to be compared against the trace on
    the same interpreter, not quoted as an absolute. The run stamps its
    interpreter and platform into the JSON for that reason.
    """
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_sizeof(k, seen) + deep_sizeof(v, seen)
                    for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_sizeof(i, seen) for i in obj)
    elif hasattr(obj, "__dict__"):
        size += deep_sizeof(vars(obj), seen)
    elif hasattr(obj, "__slots__"):
        size += sum(deep_sizeof(getattr(obj, s), seen)
                    for s in obj.__slots__ if hasattr(obj, s))
    return size


def triples(n: int) -> list[tuple[str, str, str]]:
    """N triples, every symbol distinct: the same hard case as the capacity sweep."""
    return [(f"s{i}", f"r{i}", f"o{i}") for i in range(n)]


def one_pass(n: int, dim: int = DIM) -> dict:
    """One full measurement of one N. Returns seconds and bytes."""
    facts = triples(n)

    m = HolographicMemory(dim=dim)
    t0 = time.perf_counter()
    for s, r, o in facts:
        m.learn(s, r, o)
    insert_s = time.perf_counter() - t0

    m.trace                      # materialise once, outside the timing
    m._invalidate()
    t0 = time.perf_counter()
    m.trace
    rebuild_s = time.perf_counter() - t0

    step = max(1, n // QUERIES_PER_PASS)
    probes = facts[::step][:QUERIES_PER_PASS]
    t0 = time.perf_counter()
    for s, r, _ in probes:
        m.query(s, r)
    query_s = (time.perf_counter() - t0) / len(probes)

    return {
        "insert_s": insert_s,
        "rebuild_s": rebuild_s,
        "query_s": query_s,
        "factlist_bytes": deep_sizeof(m.facts()),
        "trace_bytes": int(m.trace.nbytes),
    }


def _band(values: list[float]) -> tuple[float, float, float]:
    return statistics.median(values), min(values), max(values)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=",".join(str(c) for c in COUNTS),
                    help="comma-separated N values")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--out", default="results/cost.json")
    a = ap.parse_args()
    counts = [int(c) for c in a.counts.split(",") if c.strip()]

    if a.repeats < 3:
        print("WARNING: fewer than 3 passes. The band is the point of this "
              "file; a single pass is how the 619x error got published.\n")

    print(f"holomem cost, d={a.dim}, {a.repeats} passes per cell")
    print(f"{platform.python_implementation()} {platform.python_version()} "
          f"on {platform.system()} {platform.machine()}\n")
    print(f"{'N':>6} {'insert total':>22} {'rebuild':>20} {'query':>10} "
          f"{'fact list':>11} {'trace':>9}")
    print("-" * 84)

    rows = []
    for n in counts:
        passes = [one_pass(n, a.dim) for _ in range(a.repeats)]
        ins = _band([p["insert_s"] for p in passes])
        reb = _band([p["rebuild_s"] for p in passes])
        qry = _band([p["query_s"] for p in passes])
        row = {
            "n": n, "dim": a.dim, "passes": a.repeats,
            "insert_s": {"median": ins[0], "min": ins[1], "max": ins[2]},
            "rebuild_ms": {"median": reb[0] * 1e3, "min": reb[1] * 1e3,
                           "max": reb[2] * 1e3},
            "query_ms": {"median": qry[0] * 1e3, "min": qry[1] * 1e3,
                         "max": qry[2] * 1e3},
            "factlist_bytes": passes[0]["factlist_bytes"],
            "trace_bytes": passes[0]["trace_bytes"],
        }
        rows.append(row)
        print(f"{n:>6} {ins[0]:>9.2f} s [{ins[1]:.2f}-{ins[2]:.2f}] "
              f"{reb[0] * 1e3:>8.0f} ms [{reb[1] * 1e3:.0f}-{reb[2] * 1e3:.0f}] "
              f"{qry[0] * 1e3:>7.1f} ms "
              f"{row['factlist_bytes'] / 1024:>8.0f} KB "
              f"{row['trace_bytes'] / 1024:>6.0f} KB")

    # The local exponent, and why it is the number to quote.
    #
    # A speed ratio between two N is a fact about this laptop. An exponent is a
    # fact about the algorithm: it survives a machine twice as fast, because
    # both ends move together. That is the same move as the z-gate in section 4
    # of the README, measuring a quantity in units that cancel the scale, and
    # it is the honest repair for having once published a ratio that a slow
    # denominator had inflated.
    ladder = []
    for lo, hi in zip(rows, rows[1:]):
        t_lo = lo["insert_s"]["median"]
        t_hi = hi["insert_s"]["median"]
        exponent = math.log(t_hi / t_lo) / math.log(hi["n"] / lo["n"])
        ladder.append({"from_n": lo["n"], "to_n": hi["n"],
                       "from_s": t_lo, "to_s": t_hi, "exponent": exponent})

    summary = None
    if len(rows) >= 2:
        lo, hi = rows[0], rows[-1]
        n_ratio = hi["n"] / lo["n"]
        t_ratio = hi["insert_s"]["median"] / lo["insert_s"]["median"]
        summary = {
            "n_ratio": n_ratio,
            "insert_time_ratio": t_ratio,
            "pure_quadratic_would_be": n_ratio ** 2,
            "from_median_s": lo["insert_s"]["median"],
            "to_median_s": hi["insert_s"]["median"],
            "overall_exponent": math.log(t_ratio) / math.log(n_ratio),
            "ladder": ladder,
        }
        print("\nlocal exponent of the insert cost, step by step:")
        for step in ladder:
            print(f"  {step['from_n']:>5} -> {step['to_n']:<5} "
                  f"{step['from_s']:>8.3f} s -> {step['to_s']:<8.3f} s   "
                  f"exponent {step['exponent']:.2f}")
        # Both ends printed on purpose. A ratio is far more fragile than the
        # values it is built from, so quote it only next to its two bounds.
        print(f"\n{n_ratio:.0f}x the facts costs {t_ratio:.0f}x the insert time "
              f"({lo['insert_s']['median']:.3g} s -> "
              f"{hi['insert_s']['median']:.3g} s), overall exponent "
              f"{summary['overall_exponent']:.2f}. "
              f"Pure quadratic would be {n_ratio ** 2:.0f}x.")
        print("Quote the exponent, not the ratio: the ratio is a fact about "
              "this laptop.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "dim": a.dim, "passes": a.repeats,
        "rows": rows, "summary": summary,
    }, indent=1), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
