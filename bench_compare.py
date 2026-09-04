# -*- coding: utf-8 -*-
"""
bench_compare.py : the trace against an exact dict, on the same facts.

    python bench_compare.py             # ~2 minutes

WHY THIS FILE EXISTS
────────────────────
On r/LLMDevs, u/carefactor3zero put the objection plainly: this is an
approximate associative key/value structure, and a B-tree is an exact indexed
one, roughly similar in form. That deserved a measurement rather than an
argument, so this is the measurement.

The baseline is a plain Python dict, which is the honest stand-in for the
B-tree: exact, ordinary, and the thing any sane person reaches for first. It
gets every advantage. It is exact by construction, so its accuracy is 1.000 at
every N and there is no draw where it does worse.

WHAT IS ACTUALLY BEING COMPARED, AND WHY IT IS NOT THE OBVIOUS THING
────────────────────────────────────────────────────────────────────
Comparing total memory would be dishonest in this repository's favour and
against it at the same time, because both stores keep the same ground truth:
holomem keeps its fact list, the dict keeps its forward mapping. Neither can
throw those away. Comparing them measures Python, not FHRR.

The question with an actual answer is narrower, and it is the one the thread
was really about: **what does it cost to answer backwards?**

  · the dict answers (subject, relation) -> object from its forward mapping.
    To answer (relation, object) -> subject it needs a SECOND mapping, and
    that second mapping grows with N. That is the price, and it is measured
    here as `reverse_bytes`.
  · holomem answers both directions by unbinding the same trace with different
    arguments. The backward direction costs zero extra bytes, at every N,
    because there is no second structure to build. What it costs instead is
    exactness.

So the comparison is: constant bytes and approximate, against growing bytes
and exact. Both columns are reported at every N, plus the crossover where the
dict's reverse index first exceeds the whole trace.

WHAT THE READER SHOULD DO WITH THE ANSWER
─────────────────────────────────────────
Read the crossover next to the capacity curve, not on its own. There is a band
of N where the trace is both smaller than the reverse index and still accurate,
and outside that band the dict simply wins. The width of that band is the
honest case for this repository, and it is narrower than the pitch in this
space usually admits.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from holomem import HolographicMemory, fold

COUNTS = (50, 100, 150, 250, 500)
FIXED_DIM = 1024
TRIALS = 5


def deep_sizeof(obj, seen: set[int] | None = None) -> int:
    """Recursive sys.getsizeof, counting each object once. See bench_cost.py."""
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
    return size


class ExactStore:
    """The baseline: two dicts, exact, no approximation anywhere.

    Deliberately not clever. A B-tree would order the keys and pay a log
    factor for it; a dict is strictly kinder to the baseline, so if the trace
    cannot beat a dict on the one axis it claims, it beats nothing.
    """

    def __init__(self) -> None:
        self.forward: dict[tuple[str, str], str] = {}
        self.reverse: dict[str, tuple[str, str]] = {}

    def learn(self, s: str, r: str, o: str) -> None:
        self.forward[(fold(s), fold(r))] = o
        self.reverse[fold(o)] = (s, r)

    def query(self, s: str, r: str) -> str | None:
        return self.forward.get((fold(s), fold(r)))

    def query_subject(self, r: str, o: str) -> str | None:
        hit = self.reverse.get(fold(o))
        return hit[0] if hit else None


def _names(n: int, rng: np.random.Generator, tag: str) -> list[str]:
    base = rng.integers(0, 2**40)
    return [f"{tag}{base}_{i}" for i in range(n)]


def one_trial(n: int, dim: int, rng: np.random.Generator) -> dict:
    """One draw of N facts, measured through both stores."""
    subjects, relations, objects = (_names(n, rng, t) for t in ("s", "r", "o"))
    facts = list(zip(subjects, relations, objects))

    exact = ExactStore()
    t0 = time.perf_counter()
    for s, r, o in facts:
        exact.learn(s, r, o)
    exact_build = time.perf_counter() - t0

    mem = HolographicMemory(dim=dim)
    t0 = time.perf_counter()
    for s, r, o in facts:
        mem.learn(s, r, o)
    mem_build = time.perf_counter() - t0
    mem.trace                                  # materialise before timing reads

    # forward, both stores
    t0 = time.perf_counter()
    exact_fwd = sum(exact.query(s, r) == o for s, r, o in facts)
    exact_fwd_s = (time.perf_counter() - t0) / n

    t0 = time.perf_counter()
    mem_fwd = sum(fold(mem.query(s, r)[0]) == fold(o) for s, r, o in facts)
    mem_fwd_s = (time.perf_counter() - t0) / n

    # backward, both stores
    t0 = time.perf_counter()
    exact_bwd = sum(exact.query_subject(r, o) == s for s, r, o in facts)
    exact_bwd_s = (time.perf_counter() - t0) / n

    t0 = time.perf_counter()
    mem_bwd = sum(fold(mem.query_subject(r, o)[0]) == fold(s) for s, r, o in facts)
    mem_bwd_s = (time.perf_counter() - t0) / n

    return {
        "exact_build_s": exact_build,
        "mem_build_s": mem_build,
        "exact_fwd_acc": exact_fwd / n,
        "mem_fwd_acc": mem_fwd / n,
        "exact_bwd_acc": exact_bwd / n,
        "mem_bwd_acc": mem_bwd / n,
        "exact_fwd_s": exact_fwd_s,
        "mem_fwd_s": mem_fwd_s,
        "exact_bwd_s": exact_bwd_s,
        "mem_bwd_s": mem_bwd_s,
        # the two numbers the whole file is about
        "reverse_bytes": deep_sizeof(exact.reverse),
        "trace_bytes": int(mem.trace.nbytes),
        "forward_bytes": deep_sizeof(exact.forward),
        "factlist_bytes": deep_sizeof(mem.facts()),
    }


def _med(trials: list[dict], key: str) -> float:
    return statistics.median(t[key] for t in trials)


def _mean(trials: list[dict], key: str) -> float:
    return statistics.fmean(t[key] for t in trials)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=",".join(str(c) for c in COUNTS))
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--dim", type=int, default=FIXED_DIM)
    ap.add_argument("--out", default="results/compare.json")
    a = ap.parse_args()
    counts = [int(c) for c in a.counts.split(",") if c.strip()]

    print(f"holomem against an exact dict, {a.trials} draws per cell\n")
    print(f"{'N':>5} {'d':>6} {'fwd acc':>9} {'bwd acc':>9} {'dict acc':>9} "
          f"{'fwd query':>11} {'dict query':>11} {'trace':>9} {'reverse idx':>12}")
    print("-" * 92)

    rows = []
    for n in counts:
        # two arms: the dimension the README recommends holding fixed, and the
        # dimension the d/4 rule of thumb would pick, which is the CROSSING.
        for label, dim in (("fixed", a.dim), ("d = 4N", 4 * n)):
            trials = [one_trial(n, dim, np.random.default_rng(9000 + n + k))
                      for k in range(a.trials)]
            row = {
                "n": n, "dim": dim, "arm": label, "trials": a.trials,
                "mem_fwd_acc": _mean(trials, "mem_fwd_acc"),
                "mem_bwd_acc": _mean(trials, "mem_bwd_acc"),
                "exact_fwd_acc": _mean(trials, "exact_fwd_acc"),
                "exact_bwd_acc": _mean(trials, "exact_bwd_acc"),
                "mem_fwd_ms": _med(trials, "mem_fwd_s") * 1e3,
                "mem_bwd_ms": _med(trials, "mem_bwd_s") * 1e3,
                "exact_fwd_ms": _med(trials, "exact_fwd_s") * 1e3,
                "exact_bwd_ms": _med(trials, "exact_bwd_s") * 1e3,
                "mem_build_s": _med(trials, "mem_build_s"),
                "exact_build_s": _med(trials, "exact_build_s"),
                "trace_bytes": trials[0]["trace_bytes"],
                "reverse_bytes": int(_med(trials, "reverse_bytes")),
                "forward_bytes": int(_med(trials, "forward_bytes")),
                "factlist_bytes": int(_med(trials, "factlist_bytes")),
            }
            rows.append(row)
            print(f"{n:>5} {dim:>6} {row['mem_fwd_acc']:>9.3f} "
                  f"{row['mem_bwd_acc']:>9.3f} {row['exact_fwd_acc']:>9.3f} "
                  f"{row['mem_fwd_ms']:>8.2f} ms {row['exact_fwd_ms']:>8.4f} ms "
                  f"{row['trace_bytes'] / 1024:>6.0f} KB "
                  f"{row['reverse_bytes'] / 1024:>9.0f} KB")
        print()

    # Where the dict's reverse index first costs more than the entire trace.
    # Interpolated between the two measured N that straddle it, the same way
    # the capacity curve interpolates its 50 % crossing.
    fixed = [r for r in rows if r["arm"] == "fixed"]
    crossover = None
    for lo, hi in zip(fixed, fixed[1:]):
        if lo["reverse_bytes"] <= lo["trace_bytes"] < hi["reverse_bytes"]:
            span = hi["reverse_bytes"] - lo["reverse_bytes"]
            frac = (lo["trace_bytes"] - lo["reverse_bytes"]) / span
            crossover = lo["n"] + frac * (hi["n"] - lo["n"])
            break

    if crossover is not None:
        print(f"At d={a.dim} the reverse index passes the {fixed[0]['trace_bytes'] // 1024} KB "
              f"trace at about N={crossover:.0f}.")
        print("Below that, the dict is smaller AND exact, and there is no "
              "argument to make.")
        print("Above it, the trace is smaller, and the question becomes how "
              "long it stays accurate.")
    else:
        print("No crossover inside the measured range.")

    slowdown = statistics.median(r["mem_fwd_ms"] / r["exact_fwd_ms"]
                                 for r in fixed)
    print(f"\nCost of that: a forward query is {slowdown:,.0f}x slower than the "
          f"dict lookup, median over the fixed-d rows,")
    print("and approximate where the dict is exact at every N.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dim_fixed": a.dim, "trials": a.trials, "rows": rows,
        "reverse_index_crossover_n": crossover,
        "forward_query_slowdown_median": slowdown,
    }, indent=1), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
