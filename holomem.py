# -*- coding: utf-8 -*-
"""
holomem.py : a holographic associative memory that lives in time.

One vector holds every fact you ever learned. Its size never changes.

This is a Fourier Holographic Reduced Representation (FHRR), the frequency
domain member of the Vector Symbolic Architecture family described by Plate in
1995. The classical construction is well known and this file does not claim it.
What it adds is the part that classical VSA leaves out: **a memory that ages**.
A fact nobody reconfirms fades, a fact relearned strengthens, a contradicted
fact is subtracted, and a second trace indexes everything by the month it was
learned so you can ask what was true back then.

    pip install numpy
    python -c "import holomem; help(holomem)"

─────────────────────────────────────────────────────────────────────────────
THE ALGEBRA, IN FOUR LINES
─────────────────────────────────────────────────────────────────────────────

Every symbol is a unit phasor in C^d: d complex numbers, each of modulus 1,
with a random phase derived from the hash of the symbol's name.

    bind(a, b)   = a * b            elementwise. Phases ADD.
    unbind(c, b) = c * conj(b)      phases SUBTRACT. Exact inverse of bind.
    T            = sum of w_i * bind(S_i, R_i, O_i)     one vector, always
    unbind(T, bind(S, R)) ~ O + noise                   the query

The last line is the whole trick, and it is worth staring at. Binding is
exactly invertible for unit phasors: unbind(bind(a, b), b) == a to floating
point. Superposition is linear. So when you unbind the *sum* by one key, the
term that carried that key comes back clean and every other term comes back as
a phase-scrambled vector that is nearly orthogonal to everything you care
about. The answer is signal; the rest of the memory is noise that grows like
sqrt(N). That ratio is the entire capacity story, and section 3 measures it
rather than asserting it.

─────────────────────────────────────────────────────────────────────────────
WHAT THIS IS FOR, AND WHAT IT IS NOT FOR
─────────────────────────────────────────────────────────────────────────────

It is for the case a vector database handles badly: **relational recall of a
small, changing set of facts about one person or one project**, where the
question is "what is the object of (subject, relation)?" and the answer must
degrade gracefully as the belief gets old.

Properties you get that cosine search over embeddings does not give you:

  fixed-size trace    the trace is d complex numbers whether it carries 10
                      facts or 10 000. No index to rebuild, no shards, no
                      re-embedding. The rest of the object is NOT fixed: the
                      fact list is retained as ground truth and the symbol
                      pools are rebuilt per query, both O(N). See section 6.
  inverse queries     "who works on X?" is the same operation as "what does A
                      work on?", with the arguments swapped. No second index.
  graceful fading     a half-remembered fact still contributes a little to
                      recall. Forgetting is a slope, not a boolean.
  exact subtraction   superposition is linear, so removing a contradicted fact
                      is exact. There is no SECONDARY index to invalidate,
                      because there is only one structure. The trace itself is
                      dropped and recomputed: see _invalidate().

And the honest other side:

  bounded capacity    crosstalk grows with N. Past a point, recall collapses.
                      Section 3 measures where, for four dimensions.
  cleanup required    the raw query output is approximate. You must snap it to
                      a known symbol, so you need the list of candidates.
  not a text index    it stores triples, not prose. It complements a text
                      retriever, it does not replace one.
  ground truth lives elsewhere
                      keep a plain list of facts. The trace is a computed view
                      of that list, and it is rebuilt, never repaired.

**One disclaimer worth reading before you build on this.** A tempting idea in
this space is to hand the trace vector itself to a language model as a "ghost
vector" of the user's history. With every hosted model API in production today
that is impossible: they consume tokens, not vectors. This memory earns its
keep by deciding *which few facts are still sharp enough to be worth spending
tokens on*, and that is a real job, but it is a smaller job than the one the
vector-native pitch implies. Anyone promising you the other thing on top of a
token API has not tried it.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

__all__ = [
    "bind", "unbind", "csim", "symbol", "fold", "epoch_of",
    "Fact", "HolographicMemory",
]

_EPS = 1e-12
_DAY = 86_400.0


# ═══════════════════════════════════════════════════════════════════════════
# 1. THE ALGEBRA
# ═══════════════════════════════════════════════════════════════════════════

def bind(*vs: np.ndarray) -> np.ndarray:
    """Bind symbols together. Elementwise product, so phases add.

    Binding is commutative and associative, which is why bind(S, R, O) can be
    written in any order and why the query below can peel off S and R together
    in a single operation.
    """
    out = np.asarray(vs[0])
    for v in vs[1:]:
        out = out * np.asarray(v)
    return out


def unbind(c: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Peel `b` back off `c`. Multiply by the conjugate, so phases subtract.

    For unit phasors this is an exact inverse, not an approximation:
    unbind(bind(a, b), b) == a to floating point. The approximation in this
    system comes from *superposition*, never from binding.
    """
    return np.asarray(c) * np.conjugate(np.asarray(b))


def csim(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised complex cosine: the real part of the Hermitian inner product.

    Two independent random phasors of dimension d score around 0 with standard
    deviation about 1/sqrt(2d). That number is the noise floor every capacity
    claim in this file is measured against.
    """
    a, b = np.asarray(a), np.asarray(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < _EPS or nb < _EPS:
        return 0.0
    return float(np.real(np.vdot(a, b)) / (na * nb))


# ═══════════════════════════════════════════════════════════════════════════
# 2. SYMBOLS: DERIVED FROM THE NAME, NEVER STORED
# ═══════════════════════════════════════════════════════════════════════════

def fold(name: str) -> str:
    """Normalise a symbol name: strip accents, lowercase, keep [a-z0-9_].

    "Startup", "startup" and "startup " must be the SAME symbol. Skip this and
    a case difference produces two nearly orthogonal vectors for one concept,
    which is invisible at write time and simply loses the recall later.
    """
    nfkd = unicodedata.normalize("NFKD", str(name or ""))
    flat = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "_", flat).strip("_") or "empty"


_cache: dict[tuple[str, int], np.ndarray] = {}
_CACHE_MAX = 4096


def symbol(name: str, dim: int) -> np.ndarray:
    """The unit phasor for `name`, derived from a hash of the folded name.

    This is the design decision that makes the whole thing portable. Because
    the vector is *derived* rather than assigned, there is no codebook to
    persist, no insertion order to preserve, and no migration when you restart.
    A plain list of facts is enough to rebuild the identical trace on another
    machine, in another process, a year later.
    """
    key = (fold(name), int(dim))
    hit = _cache.get(key)
    if hit is not None:
        return hit
    seed = int.from_bytes(
        hashlib.blake2b(key[0].encode("utf-8"), digest_size=8).digest(), "big")
    theta = np.random.default_rng(seed).uniform(-np.pi, np.pi, size=int(dim))
    v = np.exp(1j * theta)
    if len(_cache) >= _CACHE_MAX:
        _cache.clear()               # bounded, and free to regenerate
    _cache[key] = v
    return v


def epoch_of(ts: float) -> str:
    """The month a fact was learned, as a symbol name: "epoch:2026-09"."""
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    return f"epoch:{d.year:04d}-{d.month:02d}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. FACTS THAT AGE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Fact:
    """One triple, plus everything needed to know how much it still counts."""
    s: str
    r: str
    o: str
    weight: float = 1.0
    created_ts: float = field(default_factory=time.time)
    last_seen_ts: float = field(default_factory=time.time)

    def effective_weight(self, now: float, half_life_days: float) -> float:
        """Weight after exponential decay since the fact was last confirmed.

        Decay runs from `last_seen_ts`, not `created_ts`. A fact you keep
        mentioning stays sharp however old it is, which is the behaviour you
        want: age is not the same thing as irrelevance.
        """
        age_days = max(0.0, (now - self.last_seen_ts) / _DAY)
        return self.weight * (0.5 ** (age_days / half_life_days))

    def key(self) -> tuple[str, str, str]:
        return (fold(self.s), fold(self.r), fold(self.o))


class HolographicMemory:
    """Facts in, one vector out, and the vector remembers when it learned them.

        m = HolographicMemory(dim=1024)
        m.learn("ana", "works_on", "compiler")
        m.learn("ana", "lives_in", "lisbon")
        m.query("ana", "works_on")        -> ("compiler", 0.71, 0.55)
        m.query_subject("lives_in", "lisbon") -> ("ana", 0.70)

    Three parameters carry all the temporal behaviour, and each is a policy
    choice rather than a mathematical constant. The defaults below are the ones
    measured in `bench_capacity.py`; change them to change what your system
    believes about how beliefs age.
    """

    #: Time for an unconfirmed fact to count half as much. Six weeks is a
    #: deliberate middle: long enough that a monthly conversation keeps a fact
    #: alive, short enough that a stale belief stops being asserted within a
    #: quarter.
    HALF_LIFE_DAYS = 45.0

    #: Below this effective weight a fact stops entering the trace at all.
    #: Without a floor, thousands of nearly dead facts still contribute their
    #: full share of crosstalk while contributing no signal: you pay the noise
    #: of a memory you no longer have.
    MIN_WEIGHT = 0.18

    #: What relearning an existing fact adds, and the ceiling it saturates at.
    #: Saturation matters: without it, one chatty week would make a fact
    #: permanently louder than everything learned since.
    REINFORCE_STEP = 0.25
    MAX_WEIGHT = 1.5

    #: What a contradiction multiplies the old fact's weight by. NOT deletion.
    #: A contradicted belief that keeps a little mass is what lets the memory
    #: answer "you used to say X, now you say Y" instead of silently rewriting
    #: history. Superposition is linear, so this subtraction is exact.
    CONTRADICT_FACTOR = 0.35

    def __init__(self, dim: int = 1024, now_fn=time.time) -> None:
        self.dim = int(dim)
        self._now = now_fn                 # injected, so time is testable
        self._facts: list[Fact] = []
        self._trace: np.ndarray | None = None
        self._trace_epochal: np.ndarray | None = None

    # ── writing ────────────────────────────────────────────────────────────

    def learn(self, s: str, r: str, o: str, weight: float = 1.0,
              created_ts: float | None = None) -> bool:
        """Add a fact, or reinforce it if it is already known. True if new."""
        now = self._now()
        k = (fold(s), fold(r), fold(o))
        for f in self._facts:
            if f.key() == k:
                f.weight = min(self.MAX_WEIGHT, f.weight + self.REINFORCE_STEP)
                f.last_seen_ts = now
                self._invalidate()
                return False
        self._facts.append(Fact(s, r, o, weight,
                                created_ts if created_ts is not None else now,
                                now))
        self._invalidate()
        return True

    def contradict(self, s: str, r: str, o: str | None = None) -> int:
        """Damp facts about (s, r) that are not `o`. Returns how many.

        Call this when you learn a new value for a relation that already had
        one. Passing `o=None` damps every object of that relation.
        """
        ks, kr = fold(s), fold(r)
        ko = fold(o) if o is not None else None
        n = 0
        for f in self._facts:
            if f.key()[0] == ks and f.key()[1] == kr and f.key()[2] != ko:
                f.weight *= self.CONTRADICT_FACTOR
                n += 1
        if n:
            self._invalidate()
        return n

    def forget_faded(self) -> int:
        """Drop facts that have decayed below the floor. Returns how many."""
        now = self._now()
        keep = [f for f in self._facts
                if f.effective_weight(now, self.HALF_LIFE_DAYS) >= self.MIN_WEIGHT]
        dropped = len(self._facts) - len(keep)
        if dropped:
            self._facts = keep
            self._invalidate()
        return dropped

    # ── the traces ─────────────────────────────────────────────────────────

    def _invalidate(self) -> None:
        self._trace = self._trace_epochal = None

    def _term(self, f: Fact) -> np.ndarray:
        return bind(symbol(f.s, self.dim),
                    symbol(f.r, self.dim),
                    symbol(f.o, self.dim))

    def _build(self) -> None:
        """Rebuild both traces from the fact list.

        Rebuilt from scratch, never patched in place. Every weight depends on
        `now`, so an incrementally maintained trace would drift away from the
        facts it claims to represent, and the drift would be silent. Rebuilding
        is O(N) vector adds, which is nothing next to one model call.
        """
        now = self._now()
        t = np.zeros(self.dim, dtype=np.complex128)
        te = np.zeros(self.dim, dtype=np.complex128)
        for f in self._facts:
            w = f.effective_weight(now, self.HALF_LIFE_DAYS)
            if w < self.MIN_WEIGHT:
                continue
            term = self._term(f)
            t += w * term
            # The epochal trace binds one extra symbol: the month. It answers
            # dated questions without adding crosstalk to the main trace, which
            # is the whole reason it is a second vector and not a tag.
            te += w * bind(symbol(epoch_of(f.created_ts), self.dim), term)
        self._trace, self._trace_epochal = t, te

    @property
    def trace(self) -> np.ndarray:
        """T(t): one vector in C^d, whatever N is."""
        if self._trace is None:
            self._build()
        return self._trace

    # ── reading ────────────────────────────────────────────────────────────

    def _pools(self) -> tuple[list[str], list[str]]:
        subs, objs = {}, {}
        for f in self._facts:
            subs.setdefault(fold(f.s), f.s)
            objs.setdefault(fold(f.o), f.o)
        return list(subs.values()), list(objs.values())

    def _cleanup(self, probe: np.ndarray, pool: list[str]):
        """Snap an approximate probe to the nearest known symbol.

        Returns (name, score, margin). **The margin is the useful number**, not
        the score. A high score with a low margin means the memory is torn
        between two candidates, which is exactly the situation where you should
        say nothing rather than assert the winner. Sections 3 and 4 of the
        README use the margin as the confidence gate.
        """
        if not pool:
            return None, 0.0, 0.0
        scored = sorted(((csim(probe, symbol(n, self.dim)), n) for n in pool),
                        reverse=True)
        best_score, best = scored[0]
        margin = best_score - scored[1][0] if len(scored) > 1 else best_score
        return best, float(best_score), float(margin)

    def query(self, s: str, r: str):
        """"What is the object of (s, r)?" Returns (object, score, margin)."""
        probe = unbind(self.trace, bind(symbol(s, self.dim), symbol(r, self.dim)))
        return self._cleanup(probe, self._pools()[1])

    def query_subject(self, r: str, o: str):
        """The inverse: "which subject satisfies (?, r, o)?"

        Same operation, different arguments. This is the property that has no
        equivalent in a vector index: you do not build a second index to walk
        the relation backwards, you unbind by the other pair.
        """
        probe = unbind(self.trace, bind(symbol(r, self.dim), symbol(o, self.dim)))
        return self._cleanup(probe, self._pools()[0])

    def query_at(self, epoch: str, s: str, r: str):
        """"What was the object of (s, r) back in `epoch`?" e.g. "epoch:2026-05"."""
        if self._trace_epochal is None:
            self._build()
        key = bind(symbol(epoch, self.dim),
                   symbol(s, self.dim), symbol(r, self.dim))
        probe = unbind(self._trace_epochal, key)
        return self._cleanup(probe, self._pools()[1])

    # ── housekeeping ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._facts)

    def facts(self) -> list[Fact]:
        """The ground truth. The trace is a view of this, never the reverse."""
        return list(self._facts)

    def noise_floor(self) -> float:
        """Expected |csim| between two unrelated symbols: about 1/sqrt(2d).

        Print this next to any score you are about to trust. A score of 0.05 is
        meaningless at d=256 and meaningful at d=8192, and the only way to know
        which you are looking at is to compare against this number.
        """
        return 1.0 / math.sqrt(2 * self.dim)
