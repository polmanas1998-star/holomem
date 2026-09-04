# -*- coding: utf-8 -*-
"""
test_holomem.py : the properties that must hold, and how each one can fail.

Run with `pytest -q`. Every test here is written so that breaking the line it
guards makes it fail. Where that was not obvious, the failure mode is named in
a comment, because a test whose failure mode nobody can state is decoration.
"""
from __future__ import annotations

import numpy as np
import pytest

from holomem import (Fact, HolographicMemory, bind, csim, epoch_of, fold,
                     symbol, unbind)

DAY = 86_400.0


# ── the algebra ────────────────────────────────────────────────────────────

def test_unbind_inverts_bind_exactly():
    """Not approximately. This is what separates FHRR from a lossy code."""
    a, b = symbol("ana", 512), symbol("works_on", 512)
    assert np.allclose(unbind(bind(a, b), b), a, atol=1e-12)


def test_binding_is_commutative_and_associative():
    a, b, c = (symbol(n, 256) for n in ("a", "b", "c"))
    assert np.allclose(bind(a, b, c), bind(c, b, a), atol=1e-12)
    assert np.allclose(bind(bind(a, b), c), bind(a, bind(b, c)), atol=1e-12)


def test_symbols_are_unit_phasors():
    v = symbol("anything", 1024)
    assert np.allclose(np.abs(v), 1.0, atol=1e-12)


def test_unrelated_symbols_sit_near_the_noise_floor():
    """If this drifts, every score in the README means something else."""
    d = 1024
    scores = [abs(csim(symbol(f"x{i}", d), symbol(f"y{i}", d))) for i in range(200)]
    assert np.mean(scores) < 4.0 / np.sqrt(2 * d)


def test_symbols_are_derived_not_assigned():
    """Same name, same vector, forever. This is what makes the trace portable:
    no codebook to persist, no insertion order to preserve."""
    assert np.array_equal(symbol("compiler", 512), symbol("compiler", 512))
    assert not np.array_equal(symbol("compiler", 512), symbol("compilers", 512))


def test_fold_merges_case_and_accents():
    """Skip this and one concept silently becomes two orthogonal vectors."""
    assert fold("Startup") == fold("startup ") == fold("startup")
    assert np.array_equal(symbol("Éléonore", 256), symbol("eleonore", 256))


# ── recall ─────────────────────────────────────────────────────────────────

def test_query_returns_the_bound_object():
    m = HolographicMemory(dim=1024)
    m.learn("ana", "works_on", "compiler")
    m.learn("ana", "lives_in", "lisbon")
    m.learn("bruno", "works_on", "scheduler")
    assert fold(m.query("ana", "works_on")[0]) == "compiler"
    assert fold(m.query("ana", "lives_in")[0]) == "lisbon"
    assert fold(m.query("bruno", "works_on")[0]) == "scheduler"


def test_inverse_query_needs_no_second_index():
    """The property a vector index does not have: walk the relation backwards
    with the same trace, by unbinding the other pair."""
    m = HolographicMemory(dim=1024)
    m.learn("ana", "lives_in", "lisbon")
    m.learn("bruno", "lives_in", "porto")
    assert fold(m.query_subject("lives_in", "lisbon")[0]) == "ana"
    assert fold(m.query_subject("lives_in", "porto")[0]) == "bruno"


def test_trace_size_does_not_depend_on_fact_count():
    small, big = HolographicMemory(dim=512), HolographicMemory(dim=512)
    small.learn("a", "r", "b")
    for i in range(300):
        big.learn(f"s{i}", f"r{i}", f"o{i}")
    assert small.trace.shape == big.trace.shape == (512,)


# ── time ───────────────────────────────────────────────────────────────────

def test_weight_halves_after_one_half_life():
    f = Fact("a", "r", "b", weight=1.0, created_ts=0.0, last_seen_ts=0.0)
    assert f.effective_weight(45 * DAY, 45.0) == pytest.approx(0.5, abs=1e-9)
    assert f.effective_weight(90 * DAY, 45.0) == pytest.approx(0.25, abs=1e-9)


def test_decay_runs_from_last_confirmation_not_creation():
    """Age is not irrelevance. A fact repeated last week is sharp however old.

    Mutation that must fail this: swap `last_seen_ts` for `created_ts` in
    Fact.effective_weight.
    """
    f = Fact("a", "r", "b", weight=1.0, created_ts=0.0, last_seen_ts=100 * DAY)
    assert f.effective_weight(100 * DAY, 45.0) == pytest.approx(1.0, abs=1e-9)


def test_relearning_reinforces_and_saturates():
    now = [0.0]
    m = HolographicMemory(dim=256, now_fn=lambda: now[0])
    assert m.learn("ana", "works_on", "compiler") is True
    for _ in range(20):
        assert m.learn("ana", "works_on", "compiler") is False
    assert len(m) == 1
    assert m.facts()[0].weight == pytest.approx(m.MAX_WEIGHT)


def test_faded_facts_leave_the_trace_before_they_leave_the_list():
    """The weight floor is not cosmetic: a nearly dead fact contributes its
    full share of crosstalk and no signal. Without the floor you pay the noise
    of a memory you no longer have."""
    now = [0.0]
    m = HolographicMemory(dim=512, now_fn=lambda: now[0])
    m.learn("ana", "works_on", "compiler")
    now[0] = 200 * DAY                       # 0.5 ** (200/45) = 0.046 < 0.18
    assert np.allclose(m.trace, 0)
    assert len(m) == 1                       # still in the ledger
    assert m.forget_faded() == 1
    assert len(m) == 0


def test_contradiction_actually_damps_the_old_belief():
    """⚠ The first version of this test asserted `weight == CONTRADICT_FACTOR`,
    which is `x == x`: mutating the constant to 1.0, so that contradiction
    damped nothing at all, left the suite green. Found by mutation, not by
    reading. It now pins the BEHAVIOUR, a strict decrease, instead of the
    symbol.
    """
    m = HolographicMemory(dim=1024)
    m.learn("ana", "lives_in", "lisbon")
    before = m.facts()[0].weight
    assert m.contradict("ana", "lives_in", o="porto") == 1
    after = m.facts()[0].weight
    assert after < before


def test_contradiction_makes_the_new_belief_decisive():
    """Two beliefs at equal weight leave the memory torn; damping the old one
    is what turns a coin flip into an answer. The margin must widen.

    Superposition is linear, so this subtraction is exact: there is no index to
    invalidate and nothing to rebuild.
    """
    m = HolographicMemory(dim=1024)
    m.learn("ana", "lives_in", "lisbon")
    m.learn("ana", "lives_in", "porto")
    torn = m.query("ana", "lives_in")[2]
    m.contradict("ana", "lives_in", o="porto")
    settled_name, _score, settled = m.query("ana", "lives_in")
    assert fold(settled_name) == "porto"
    assert settled > torn


def test_the_old_belief_survives_at_reduced_weight():
    """Contradiction is damping, not deletion. Keeping a little mass is what
    lets a system say "you used to say X" instead of rewriting history."""
    m = HolographicMemory(dim=512)
    m.learn("ana", "lives_in", "lisbon")
    m.contradict("ana", "lives_in", o="porto")
    assert len(m) == 1 and m.facts()[0].weight > 0


def test_epochal_trace_answers_dated_questions():
    """The second trace is the point of the whole temporal layer: ask what was
    true then, without polluting the main trace with date crosstalk."""
    ts_may = 1777593600.0                  # 2026-05-01 UTC
    ts_now = ts_may + 120 * DAY
    m = HolographicMemory(dim=2048, now_fn=lambda: ts_now)
    m.learn("ana", "lives_in", "lisbon", created_ts=ts_may)
    m.learn("ana", "lives_in", "porto", created_ts=ts_now)
    then = epoch_of(ts_may)
    assert fold(m.query_at(then, "ana", "lives_in")[0]) == "lisbon"
    assert fold(m.query_at(epoch_of(ts_now), "ana", "lives_in")[0]) == "porto"


# ── the confidence signal ──────────────────────────────────────────────────

def test_margin_collapses_when_the_memory_is_overloaded():
    """The number that makes this usable. A full trace still returns a winner;
    what tells you not to trust it is that the winner stops standing out.

    This is the property the README's gated column is built on, and it is the
    reason an absolute score threshold is the wrong gate.
    """
    light, heavy = HolographicMemory(dim=256), HolographicMemory(dim=256)
    light.learn("ana", "works_on", "compiler")
    for i in range(5):
        light.learn(f"s{i}", f"r{i}", f"o{i}")
    heavy.learn("ana", "works_on", "compiler")
    for i in range(400):
        heavy.learn(f"s{i}", f"r{i}", f"o{i}")
    assert light.query("ana", "works_on")[2] > heavy.query("ana", "works_on")[2]


def test_noise_floor_matches_the_measured_spread():
    m = HolographicMemory(dim=1024)
    assert m.noise_floor() == pytest.approx(1 / np.sqrt(2048), rel=1e-9)


# ── ground truth ───────────────────────────────────────────────────────────

def test_the_fact_list_is_the_truth_and_the_trace_is_a_view():
    """Rebuild from the same facts, get the same vector. This is what makes the
    memory portable across machines and restarts with no stored codebook."""
    facts = [("ana", "works_on", "compiler"), ("bruno", "lives_in", "porto")]
    a, b = HolographicMemory(dim=512, now_fn=lambda: 0.0), \
           HolographicMemory(dim=512, now_fn=lambda: 0.0)
    for s, r, o in facts:
        a.learn(s, r, o)
    for s, r, o in reversed(facts):          # insertion order must not matter
        b.learn(s, r, o)
    assert np.allclose(a.trace, b.trace, atol=1e-12)


# ── the README ─────────────────────────────────────────────────────────────

def test_readme_example():
    """Section 2 of the README, run verbatim, down to the printed scores.

    The README used to show this snippet WITHOUT `created_ts` on the lisbon
    line, and print an answer of 'lisbon' underneath it. That cannot happen: a
    fact learned without a date is dated today, so the epochal query has no May
    to find and comes back with a near-noise winner. The block was quoting an
    output the code could not produce, and nothing in the suite objected.

    Mutation that must fail this: drop `created_ts=MAY` from the lisbon line.
    The dated query then returns something that is not lisbon.
    """
    now = 1788480000.0                          # 2026-09-04 UTC, pinned
    may = 1778803200.0                          # 2026-05-15 UTC
    assert epoch_of(may) == "epoch:2026-05"

    m = HolographicMemory(dim=1024, now_fn=lambda: now)
    m.learn("ana", "works_on", "compiler")
    m.learn("ana", "lives_in", "lisbon", created_ts=may)
    m.learn("bruno", "works_on", "scheduler")

    assert m.trace.shape == (1024,)
    assert len(m) == 3
    assert m.noise_floor() == pytest.approx(0.0221, abs=5e-5)

    obj, score, margin = m.query("ana", "works_on")
    assert fold(obj) == "compiler"
    assert (score, margin) == (pytest.approx(0.580, abs=5e-4),
                               pytest.approx(0.583, abs=5e-4))

    subj, score, _ = m.query_subject("works_on", "scheduler")
    assert fold(subj) == "bruno" and score == pytest.approx(0.565, abs=5e-4)

    m.contradict("ana", "lives_in", o="porto")
    m.learn("ana", "lives_in", "porto")

    obj, score, margin = m.query("ana", "lives_in")
    assert fold(obj) == "porto"
    assert (score, margin) == (pytest.approx(0.587, abs=5e-4),
                               pytest.approx(0.433, abs=5e-4))

    # The line the old README got wrong, and the reason this test exists.
    obj, score, margin = m.query_at("epoch:2026-05", "ana", "lives_in")
    assert fold(obj) == "lisbon"
    assert (score, margin) == (pytest.approx(0.190, abs=5e-4),
                               pytest.approx(0.175, abs=5e-4))
    assert score > m.noise_floor()
