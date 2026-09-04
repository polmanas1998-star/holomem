# -*- coding: utf-8 -*-
"""
plot_results.py : draw the two README figures from results/capacity.json.

    python plot_results.py

Both figures in the README used to be checked in without the code that made
them, which meant the reader had to take them on trust while the rest of the
repository was reproducible. This file closes that hole: the figures are a
*view* of results/capacity.json and nothing else. Re-run the sweep and the
curves move with it.

Nothing here computes a result. Every number drawn is read from the JSON, and
the only derived quantity is the 50 % crossing, interpolated linearly between
the two measured cells that straddle it. That is the same interpolation the
README quotes, done here in five lines so it can be checked.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results/capacity.json")
INK = "#1a1a1a"
MUTED = "#5f5f5f"
PAPER = "#fafafa"
# light to dark, so the eye reads dimension as depth without the legend
BLUES = {256: "#9ecae1", 512: "#6baed6", 1024: "#3182bd",
         2048: "#1c6cab", 4096: "#0b3d68"}
GATE_COLORS = ("#1f77b4", "#f4642a", "#17a866")


def load(path: Path = RESULTS) -> dict[int, list[tuple[int, dict]]]:
    """{dim: [(n, cell), ...]} sorted by n. Fails loudly on a missing file."""
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python bench_capacity.py")
    rows = json.loads(path.read_text(encoding="utf-8"))
    by: dict[int, list[tuple[int, dict]]] = {}
    for c in rows:
        by.setdefault(c["dim"], []).append((c["n"], c))
    for d in by:
        by[d].sort()
    return by


def crossing_50(points: list[tuple[int, float]]) -> float | None:
    """N where top-1 falls through 0.50, linearly interpolated. None if never.

    Returning None rather than extrapolating is deliberate: d=4096 does not
    cross inside the sweep, and inventing a crossing for it would be the one
    number on the figure that no measurement supports.
    """
    for (n1, a1), (n2, a2) in zip(points, points[1:]):
        if a1 >= 0.5 > a2:
            return n1 + (a1 - 0.5) * (n2 - n1) / (a1 - a2)
    return None


def _log_n_axis(ax, counts: list[int]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.minorticks_off()
    ax.set_xlabel("N facts stored", color=MUTED)


def _frame(ax) -> None:
    ax.set_facecolor(PAPER)
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_ylim(-0.04, 1.08)


def figure_capacity(by, out: Path) -> None:
    """Left: recall against N. Right: the same curves against load N/d."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.0, 7.8), dpi=100)
    fig.patch.set_facecolor(PAPER)

    counts = [n for n, _ in by[min(by)]]
    loads_at_50 = []

    for dim in sorted(by):
        pts = [(n, c["top1_mean"]) for n, c in by[dim]]
        xs = [n for n, _ in pts]
        ys = [a for _, a in pts]
        ax1.plot(xs, ys, "-o", color=BLUES[dim], markersize=4, linewidth=2)
        ax1.annotate(f"d={dim}", (xs[-1], ys[-1]), xytext=(9, 0),
                     textcoords="offset points", va="center",
                     fontsize=11, fontweight="bold", color=INK)

        n50 = crossing_50(pts)
        if n50 is not None:
            ax1.plot([n50], [0.5], "o", markersize=11, markerfacecolor="none",
                     markeredgecolor=BLUES[dim], markeredgewidth=2.2)
            ax1.annotate(f"{round(n50)}", (n50, 0.5), xytext=(0, -22),
                         textcoords="offset points", ha="center",
                         fontsize=11, fontweight="bold", color=INK)
            loads_at_50.append(n50 / dim)
            ax2.plot([n50 / dim], [0.5], "o", markersize=11,
                     markerfacecolor="none", markeredgecolor=BLUES[dim],
                     markeredgewidth=2.2)

        ax2.plot([n / dim for n in xs], ys, "-o", color=BLUES[dim],
                 markersize=4, linewidth=2, label=f"d = {dim}")

    for ax in (ax1, ax2):
        _frame(ax)
        ax.axhline(0.5, linestyle="--", color="#444444", linewidth=1.3)
        ax.set_ylabel("top-1 recall", color=MUTED)
    _log_n_axis(ax1, counts)
    ax1.annotate("50 % top-1", (10, 0.5), xytext=(2, 8),
                 textcoords="offset points", fontsize=10, color=MUTED)

    ax2.set_xscale("log")
    ax2.set_xticks([0.0025, 0.01, 0.05, 0.25, 1, 2])
    ax2.set_xticklabels(["0.0025", "0.01", "0.05", "0.25", "1", "2"])
    ax2.minorticks_off()
    ax2.set_xlabel("load   N / d", color=MUTED)
    ax2.axvline(0.25, linestyle=":", color="#333333", linewidth=1.3)
    ax2.annotate("N = d/4", (0.25, 0.93), xytext=(9, 0),
                 textcoords="offset points", fontsize=11,
                 fontweight="bold", color=INK)
    ax2.legend(loc="lower left", frameon=False, fontsize=11, labelcolor=INK)

    ax1.set_title("Capacity: where an FHRR trace stops answering",
                  fontsize=16, fontweight="bold", color=INK, loc="left", pad=26)
    ax1.annotate("12 trials per cell. Every symbol distinct, so chance is 1/N "
                 "and crosstalk is maximal.",
                 xy=(0, 1), xycoords="axes fraction", xytext=(0, 12),
                 textcoords="offset points", fontsize=11, color=MUTED)
    spread = ", ".join(f"{l:.2f}" for l in loads_at_50)
    ax2.set_title("The same five curves, against load",
                  fontsize=16, fontweight="bold", color=INK, loc="left", pad=26)
    ax2.annotate(f"Crossings at load {spread}. d/4 is a planning number, "
                 "not a law.",
                 xy=(0, 1), xycoords="axes fraction", xytext=(0, 12),
                 textcoords="offset points", fontsize=11, color=MUTED)

    fig.text(0.058, 0.028, "holomem  -  FHRR (Plate 1995)  -  reproduce: "
             "python bench_capacity.py && python plot_results.py",
             fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 1.0))
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    print(f"  {out}")


def figure_gate(by, out: Path, dims=(1024, 2048)) -> None:
    """What the gate buys: precision up, coverage down, at two dimensions."""
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.6), dpi=100)
    fig.patch.set_facecolor(PAPER)
    labels = ("top-1, no gate", "precision when it answers",
              "share of questions answered")

    for ax, dim in zip(axes, dims):
        cells = by[dim]
        counts = [n for n, _ in cells]
        series = (
            [c["top1_mean"] for _, c in cells],
            [c["gated_precision"] for _, c in cells],
            [c["coverage"] for _, c in cells],
        )
        for ys, colour, label in zip(series, GATE_COLORS, labels):
            xs = [n for n, y in zip(counts, ys) if y is not None]
            yy = [y for y in ys if y is not None]
            ax.plot(xs, yy, "-o", color=colour, markersize=4,
                    linewidth=2, label=label)
        _frame(ax)
        _log_n_axis(ax, counts)
        ax.set_ylabel("proportion", color=MUTED)
        ax.legend(loc="lower left", frameon=False, fontsize=11, labelcolor=INK)

    axes[0].set_title("d = 1024: the gate trades coverage for precision",
                      fontsize=16, fontweight="bold", color=INK,
                      loc="left", pad=26)
    axes[0].annotate("Answer only when the winner stands 4 sigma above the "
                     "candidates it beat.",
                     xy=(0, 1), xycoords="axes fraction", xytext=(0, 12),
                     textcoords="offset points", fontsize=11, color=MUTED)

    ref = dict(by[dims[1]])[300]
    axes[1].set_title(f"d = {dims[1]}: the same trade, one octave later",
                      fontsize=16, fontweight="bold", color=INK,
                      loc="left", pad=26)
    axes[1].annotate(f"300 facts: {ref['top1_mean'] * 100:.0f} % raw, or "
                     f"{ref['gated_precision'] * 100:.0f} % precision on the "
                     f"{ref['coverage'] * 100:.0f} % it is willing to answer.",
                     xy=(0, 1), xycoords="axes fraction", xytext=(0, 12),
                     textcoords="offset points", fontsize=11, color=MUTED)

    fig.text(0.058, 0.028, "holomem  -  gate z >= 4  -  12 trials per cell",
             fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 1.0))
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    print(f"  {out}")


def figure_compare(data: dict, out: Path) -> None:
    """The trace against an exact dict: bytes on the left, accuracy on the right.

    Drawn from results/compare.json. Both panels share an x axis on purpose:
    the whole argument is that the byte crossover and the accuracy collapse
    happen at different N, and the gap between them is the operating band.
    """
    rows = [r for r in data["rows"] if r["arm"] == "fixed"]
    ns = [r["n"] for r in rows]
    dim = data["dim_fixed"]
    dict_total = [(r["forward_bytes"] + r["reverse_bytes"]) / 1024 for r in rows]
    holo_total = [(r["factlist_bytes"] + r["trace_bytes"]) / 1024 for r in rows]
    trace_kb = [r["trace_bytes"] / 1024 for r in rows]
    reverse_kb = [r["reverse_bytes"] / 1024 for r in rows]
    acc = [r["mem_fwd_acc"] for r in rows]
    crossover = data.get("reverse_index_crossover_n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.0, 7.6), dpi=100)
    fig.patch.set_facecolor(PAPER)

    ax1.plot(ns, dict_total, "-o", color="#f4642a", linewidth=2, markersize=5,
             label="exact dict: forward + reverse index")
    ax1.plot(ns, holo_total, "-o", color="#3182bd", linewidth=2, markersize=5,
             label="holomem: fact list + trace")
    ax1.plot(ns, reverse_kb, "--o", color="#f4a26a", linewidth=1.8, markersize=4,
             label="the dict's reverse index alone")
    ax1.plot(ns, trace_kb, "--o", color="#7fb3d8", linewidth=1.8, markersize=4,
             label=f"the trace alone (d={dim})")
    _frame(ax1)
    ax1.set_ylim(0, max(dict_total) * 1.12)
    _log_n_axis(ax1, ns)
    ax1.set_ylabel("kilobytes", color=MUTED)
    ax1.legend(loc="upper left", frameon=False, fontsize=11, labelcolor=INK)
    ax1.set_title("What each store costs to hold N facts",
                  fontsize=16, fontweight="bold", color=INK, loc="left", pad=26)
    ax1.annotate("Both keep the same ground truth. The dict pays a second "
                 "index to answer backwards; the trace does not.",
                 xy=(0, 1), xycoords="axes fraction", xytext=(0, 12),
                 textcoords="offset points", fontsize=11, color=MUTED)

    ax2.axhline(1.0, color="#f4642a", linewidth=2)
    ax2.annotate("exact dict, 1.000 at every N", (ns[0], 1.0), xytext=(6, 8),
                 textcoords="offset points", fontsize=11, color="#c04a16")
    ax2.plot(ns, acc, "-o", color="#3182bd", linewidth=2, markersize=5,
             label=f"holomem top-1, d={dim}")
    if crossover:
        ax2.axvspan(crossover, 150, color="#3182bd", alpha=0.10)
        ax2.annotate(f"trace smaller than the\nreverse index from N={crossover:.0f}",
                     (crossover, 0.12), xytext=(6, 0), textcoords="offset points",
                     fontsize=11, color=INK)
        ax1.axvline(crossover, linestyle=":", color="#333333", linewidth=1.3)
    _frame(ax2)
    _log_n_axis(ax2, ns)
    ax2.set_ylabel("top-1 accuracy", color=MUTED)
    ax2.legend(loc="lower left", frameon=False, fontsize=11, labelcolor=INK)
    ax2.set_title("What that saving costs in answers",
                  fontsize=16, fontweight="bold", color=INK, loc="left", pad=26)
    ax2.annotate("The shaded band is where the trace is both smaller and "
                 "still trustworthy. It is narrow.",
                 xy=(0, 1), xycoords="axes fraction", xytext=(0, 12),
                 textcoords="offset points", fontsize=11, color=MUTED)

    slow = data.get("forward_query_slowdown_median")
    fig.text(0.058, 0.028, f"holomem  -  {data['trials']} draws per cell  -  "
             f"a forward query is {slow:,.0f}x slower than the dict lookup  -  "
             "reproduce: python bench_compare.py", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 1.0))
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    print(f"  {out}")


def main() -> int:
    by = load()
    print(f"drawing from {RESULTS} ({sum(len(v) for v in by.values())} cells)")
    figure_capacity(by, Path("results/capacity.png"))
    figure_gate(by, Path("results/gate.png"))

    compare = Path("results/compare.json")
    if compare.exists():
        figure_compare(json.loads(compare.read_text(encoding="utf-8")),
                       Path("results/compare.png"))
    else:
        print("  (no results/compare.json yet: run python bench_compare.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
