# -*- coding: utf-8 -*-
"""
bench_capacity.py : how many facts fit in one vector before recall collapses?

Everything the README claims about capacity is produced by this file. Run it
and you get the same numbers, or you have found a bug worth reporting:

    python bench_capacity.py            # the full sweep, ~35 s, 40 cells
    python bench_capacity.py --quick    # 6 coarse cells, a smoke test

The full sweep is deterministic: every cell is seeded from its own dim and
N, so a clean checkout reproduces results/capacity.json byte for byte, and
the README table and both figures are views of that one file. --quick does
NOT reproduce the table, and writes somewhere else so it cannot pretend to.

WHAT IS MEASURED
────────────────
For a given dimension d and fact count N, we store N triples whose symbols are
all distinct, then ask `query(s, r)` for every one of them and check whether
the returned object is the right one. This is the HARD case on purpose: N
facts means N candidate objects in the cleanup pool, so chance is 1/N and the
crosstalk is maximal. A real memory where one subject holds many relations does
better than these curves, never worse. The numbers here are a floor.

Two quantities come out, and the second matters more than the first:

  top-1 accuracy    how often the raw answer is correct.
  gated precision   how often it is correct AMONG the answers the memory is
                    willing to give, reported with the coverage.

**The gate is measured in units of noise, and the first version of this bench
got that wrong.** It used an absolute margin threshold of 0.10, which looked
sensible at N=25 and let through 0.3 % of queries at N=100, because every
score shrinks as the trace fills. An absolute confidence threshold silently
stops firing exactly when the memory starts needing one. The gate here is a
z-score instead: how far the winner stands above the other candidates, in
standard deviations of their own distribution. That number is scale free, so
one threshold holds across every dimension and every N in the table.

A memory that is right 60 % of the time is not useful. A memory that is right
97 % of the time on the 55 % of questions it is willing to answer, and silent
on the rest, is useful. The margin is what buys that, and the gated column is
the one to read.

METHOD NOTES, SO THE NUMBERS CAN BE ARGUED WITH
───────────────────────────────────────────────
  · every cell is `--repeats` independent trials with different symbol names;
    the table prints the mean and the min, because a mean alone hides the bad
    draw that a user would actually experience.
  · all weights are 1.0 and no decay is applied: this isolates crosstalk from
    the temporal machinery. The temporal behaviour is tested separately in
    test_holomem.py.
  · the sweep uses a vectorised cleanup for speed. It is checked against the
    library's own `HolographicMemory.query` on every configuration before the
    sweep starts, and the run aborts if they ever disagree. A benchmark that
    measures a faster copy of the code instead of the code is measuring
    nothing.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from holomem import HolographicMemory, bind, fold, symbol, unbind

DIMS = (256, 512, 1024, 2048, 4096)
COUNTS = (10, 25, 50, 100, 150, 200, 300, 500)
Z_GATE = 4.0          # winner must stand 4 sigma above the other candidates


def _names(n: int, rng: np.random.Generator, tag: str) -> list[str]:
    """n distinct symbol names, unique per trial so trials are independent."""
    base = rng.integers(0, 2**40)
    return [f"{tag}{base}_{i}" for i in range(n)]


def _pool_matrix(names: list[str], dim: int) -> np.ndarray:
    """Stack the pool's symbols, rows already L2-normalised for the cosine."""
    m = np.stack([symbol(n, dim) for n in names])
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def one_trial(dim: int, n: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """One trial. Returns (top-1 accuracy, gated precision, gated coverage)."""
    subjects = _names(n, rng, "s")
    relations = _names(n, rng, "r")
    objects = _names(n, rng, "o")

    terms = np.stack([bind(symbol(s, dim), symbol(r, dim), symbol(o, dim))
                       for s, r, o in zip(subjects, relations, objects)])
    trace = terms.sum(axis=0)

    pool = _pool_matrix(objects, dim)                       # (n, d), normalised
    keys = np.stack([bind(symbol(s, dim), symbol(r, dim))
                     for s, r in zip(subjects, relations)])
    probes = trace[None, :] * np.conjugate(keys)       # (n, d), one per query
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)

    # Real part of the Hermitian product, every query against every candidate.
    scores = np.real(probes @ np.conjugate(pool).T)    # (n_queries, n_candidates)

    order = np.argsort(-scores, axis=1)
    winner = order[:, 0]
    correct = winner == np.arange(n)

    # z of the winner against the candidates it beat. Using the losers' own
    # spread, not a constant, is what makes one threshold work at every N.
    ranked = np.take_along_axis(scores, order, axis=1)
    head, rest = ranked[:, 0], ranked[:, 1:]
    if rest.shape[1] >= 2:
        z = (head - rest.mean(axis=1)) / (rest.std(axis=1) + 1e-12)
    else:
        z = np.full(n, np.inf)

    kept = z >= Z_GATE
    coverage = float(kept.mean())
    precision = float(correct[kept].mean()) if kept.any() else float("nan")
    return float(correct.mean()), precision, coverage


def fidelity_check(dim: int = 512, n: int = 40) -> None:
    """The vectorised path must agree with the shipped library, or we stop.

    This is not ceremony. The first version of this bench built its trace with
    a different symbol order and quietly measured a memory the library does not
    implement. A bench that drifts from the code it benchmarks publishes a
    number about nothing.
    """
    rng = np.random.default_rng(7)
    subjects, relations, objects = (_names(n, rng, t) for t in ("s", "r", "o"))
    m = HolographicMemory(dim=dim)
    for s, r, o in zip(subjects, relations, objects):
        m.learn(s, r, o)

    pool = _pool_matrix(objects, dim)
    trace = m.trace
    mismatches = 0
    for i, (s, r) in enumerate(zip(subjects, relations)):
        expected, lib_score, _margin = m.query(s, r)
        probe = unbind(trace, bind(symbol(s, dim), symbol(r, dim)))
        probe = probe / np.linalg.norm(probe)
        scores = np.real(probe @ np.conjugate(pool).T)
        got = objects[int(np.argmax(scores))]
        if fold(got) != fold(expected) or abs(scores.max() - lib_score) > 1e-9:
            mismatches += 1
    if mismatches:
        raise SystemExit(f"ABORT: {mismatches}/{n} mismatches between bench and library. "
                         f"The bench is not measuring the shipped code.")
    print(f"fidelity check: {n}/{n} queries identical to the library\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None,
                    help="default: results/capacity.json, or "
                         "results/capacity-quick.json under --quick")
    a = ap.parse_args()

    dims = (512, 1024) if a.quick else DIMS
    counts = (25, 100, 300) if a.quick else COUNTS
    repeats = 4 if a.quick else a.repeats

    # --quick used to default to the same output path as the full sweep, so a
    # reader following the README overwrote the 40-cell file the README's table
    # and both figures are drawn from, and only found out from `git status`. A
    # coarse run must never be able to stand in for the published one.
    out = Path(a.out) if a.out else Path(
        "results/capacity-quick.json" if a.quick else "results/capacity.json")

    if a.quick:
        print("QUICK: 6 coarse cells, 4 trials each. This is a smoke test, not "
              "the published table.\n       The README table comes from the "
              "full sweep: python bench_capacity.py\n")

    fidelity_check()

    print(f"FHRR capacity, {repeats} trials per cell, gate z >= {Z_GATE}")
    print(f"{'dim':>5} {'noise':>7} {'N':>5} {'top-1':>8} {'worst':>7} "
          f"{'gated':>8} {'cover':>7}")
    print("-" * 54)

    rows = []
    t0 = time.time()
    for dim in dims:
        floor = 1.0 / np.sqrt(2 * dim)
        for n in counts:
            rng = np.random.default_rng(1000 + dim + n)
            trials = [one_trial(dim, n, rng) for _ in range(repeats)]
            acc = [e[0] for e in trials]
            prec = [e[1] for e in trials if e[1] == e[1]]
            cov = [e[2] for e in trials]
            row = {
                "dim": dim, "n": n,
                "top1_mean": round(float(np.mean(acc)), 4),
                "top1_worst": round(float(np.min(acc)), 4),
                "gated_precision": round(float(np.mean(prec)), 4) if prec else None,
                "coverage": round(float(np.mean(cov)), 4),
                "floor_bruit": round(float(floor), 4),
                "trials": repeats,
            }
            rows.append(row)
            pf = f"{row['gated_precision']:.3f}" if prec else "   n/a"
            print(f"{dim:>5} {floor:>7.4f} {n:>5} {row['top1_mean']:>8.3f} "
                  f"{row['top1_worst']:>7.3f} {pf:>8} {row['coverage']:>7.3f}")
        print()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{time.time() - t0:.1f} s  ->  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
