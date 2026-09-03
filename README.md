# A 1024-dimensional vector that remembers 150 facts, and knows when it doesn't

One complex vector holds every fact you ever taught it. Its size never changes.
You query it by algebra, not by search, and it answers backwards as easily as
forwards. It also ages: facts nobody reconfirms fade, contradicted facts are
subtracted exactly, and a second trace lets you ask what was true last spring.

This is a Fourier Holographic Reduced Representation, the frequency-domain
member of the Vector Symbolic Architecture family that Tony Plate described in
1995. That part is textbook and this repository does not claim it. What it adds
is the part the textbooks leave out: **a working temporal layer, and a measured
capacity curve you can reproduce in sixty seconds.**

```
pip install numpy
python bench_capacity.py --quick
```

I built this as the memory layer of [Dermioz](https://dermiozai.com), an AI
assistant I am working on. It earns its keep there, and the honest limits are
in section 6.

---

## 1. The algebra, in four lines

Every symbol is a unit phasor in `C^d`: `d` complex numbers of modulus 1, with
phases derived from a hash of the symbol's name.

```
bind(a, b)   = a * b                 elementwise. Phases ADD.
unbind(c, b) = c * conj(b)           phases SUBTRACT. Exact inverse of bind.
T            = sum_i  w_i * bind(S_i, R_i, O_i)      one vector, always
unbind(T, bind(S, R))  ~=  O + noise                 the query
```

The last line is the whole trick. Binding is *exactly* invertible for unit
phasors, so `unbind(bind(a,b), b) == a` to floating point. Superposition is
linear. When you unbind the whole sum by one key, the term that carried that
key comes back clean, and every other term comes back phase-scrambled and
nearly orthogonal to anything you care about.

So the answer is signal and the rest of the memory is noise. The noise grows
like `sqrt(N)`. **That ratio is the entire capacity story**, and section 3
measures it rather than asserting it.

One design choice makes the whole thing portable: symbols are *derived* from
the name, never assigned. There is no codebook to persist, insertion order does
not matter, and a plain list of facts rebuilds the identical vector on another
machine a year later.

---

## 2. Sixty seconds

```python
from holomem import HolographicMemory, epoch_of

m = HolographicMemory(dim=1024)
m.learn("ana",   "works_on", "compiler")
m.learn("ana",   "lives_in", "lisbon")
m.learn("bruno", "works_on", "scheduler")

m.query("ana", "works_on")                 # what does ana work on?
m.query_subject("works_on", "scheduler")   # who works on the scheduler?

m.contradict("ana", "lives_in", o="porto") # she moved
m.learn("ana", "lives_in", "porto")
m.query("ana", "lives_in")                 # porto
m.query_at("epoch:2026-05", "ana", "lives_in")   # but back in May: lisbon
```

Actual output, unedited:

```
trace (1024,) complex128, 3 facts, noise floor 0.0221
query(ana, works_on)                 'compiler'   score 0.580  margin 0.583
query_subject(works_on, scheduler)   'bruno'      score 0.565
query(ana, lives_in)  after move     'porto'      score 0.587  margin 0.433
query_at(epoch:2026-05, ana, lives_in) 'lisbon'   score 0.160  margin 0.136
```

Note the third line. `query_subject` is not a second index, it is the same
trace unbound by the other pair. Walking a relation backwards costs nothing and
stores nothing, which is the property no vector database gives you.

Note also the noise floor, `0.0221`. Two unrelated symbols in `d=1024` score
about that against each other. Every number above should be read against it,
and `m.noise_floor()` prints it for you, because a score of 0.05 is meaningless
at `d=256` and meaningful at `d=8192`.

---

## 3. What it costs: the capacity curve

`bench_capacity.py` stores `N` triples whose symbols are all distinct, then
queries every one of them. This is the hard case on purpose: `N` facts means
`N` candidate objects, so chance is `1/N` and crosstalk is maximal. A real
memory, where one subject carries many relations, does better than this. **The
curve is a floor.**

12 trials per cell, means shown. `top-1` is the raw answer; `gated` and `cover`
are explained in section 4.

| d | N | top-1 | worst trial | gated precision | coverage |
|---:|---:|---:|---:|---:|---:|
| 256 | 50 | 0.800 | 0.640 | 0.994 | 26 % |
| 256 | 100 | 0.404 | 0.350 | 0.928 | 5 % |
| 512 | 50 | 0.980 | 0.940 | 1.000 | 74 % |
| 512 | 150 | 0.485 | 0.427 | 0.952 | 10 % |
| **1024** | **100** | **0.973** | 0.950 | **1.000** | **71 %** |
| **1024** | **150** | **0.835** | 0.813 | **0.993** | **37 %** |
| 1024 | 300 | 0.391 | 0.340 | 0.890 | 10 % |
| 2048 | 300 | 0.782 | 0.753 | 0.984 | 40 % |
| 4096 | 500 | 0.825 | 0.806 | 0.984 | 53 % |

![top-1 recall against N, and against load N/d, for d = 256 to 4096](results/capacity.png)

The full sweep, all 40 cells, is `results/capacity.json`, and the figure is
drawn from that file. The right panel is the same five curves plotted against
load `N/d`: they nearly collapse onto one another, which is what makes a single
rule of thumb possible at all, and the residual spread is what makes it a rule
of thumb rather than a law.

**The rule of thumb, interpolated from the measurements: top-1 recall crosses
50 % at about `N = d/4`.** Measured crossings: `d=256` at N=88, `d=512` at
N=147, `d=1024` at N=261, `d=2048` at N=463. The ratio `d/N50` drifts from 2.9
to 4.4 across that range, so `d/4` is a planning number, not a law. `d=4096`
never crossed 50 % within the sweep's ceiling of 500 facts.

For a personal memory holding a few hundred facts, `d=2048` is 32 KB of
complex128 per trace, 64 KB with the epochal one, and answers with 98 %
precision on 40 % of questions. That is the operating point I would start from.

---

## 4. Knowing when it doesn't know

Raw top-1 accuracy is the wrong headline. A memory that is right 60 % of the
time is not usable. A memory that is right 98 % of the time on the 40 % of
questions it is *willing* to answer, and silent on the rest, is usable.

What buys that is the **margin**: how far the winning candidate stands above
the rest. When the trace is overloaded the query still returns a winner, but
the winner stops standing out, and that collapse is measurable before you act
on the answer.

**I got the gate wrong the first time, and the failure is worth copying.** The
first version thresholded on an absolute margin of 0.10. It looked sensible at
N=25. At N=100 it passed **0.3 %** of queries, because every score shrinks as
the trace fills. An absolute confidence threshold silently stops firing exactly
when the memory starts needing one.

The fix is to measure confidence in units of the noise it competes with: a
z-score of the winner against the distribution of the candidates it beat.

```python
z = (top - others.mean()) / others.std()
answer if z >= 4 else stay silent
```

That number is scale-free, so one threshold holds across every dimension and
every N in the table above. All the `gated` and `cover` columns use `z >= 4`.

![top-1, gated precision and coverage against N, at d = 1024 and d = 2048](results/gate.png)

The blue line is what the memory says; the orange line is what it is right
about when it agrees to speak; the green line is how often it agrees. Past
the knee, the blue line is the one that lies to you.

---

## 5. The temporal layer

Classical VSA is timeless: facts go in and stay at full strength forever. A
memory about a person cannot work that way, because people change and beliefs
go stale. Four mechanisms, each a policy choice rather than a constant of
nature:

| mechanism | default | what it buys |
|---|---|---|
| **decay** | 45-day half-life | an unconfirmed belief fades instead of being asserted forever. Decay runs from *last confirmation*, not creation: age is not the same thing as irrelevance |
| **reinforcement** | +0.25 per mention, capped at 1.5 | repetition sharpens. The cap matters: without it one chatty week makes a fact permanently louder than everything since |
| **contradiction** | weight × 0.35 | superposition is linear, so damping the old belief is *exact*. No index to invalidate, no rebuild. And it is damping, not deletion, which is what lets a system say "you used to say X" instead of quietly rewriting history |
| **epochal trace** | a second vector | every fact is also bound to the month it was learned, so dated questions are answerable without adding date crosstalk to the main trace |

There is also a floor: below an effective weight of 0.18 a fact stops entering
the trace. Without it, thousands of nearly dead facts contribute their full
share of crosstalk and no signal. You would be paying the noise of a memory you
no longer have.

---

## 6. What this is not

Read this section before building on it.

- **It is not a text index.** It stores triples, not prose. It complements a
  retriever, it does not replace one.
- **Capacity is bounded and the collapse is steep.** Past `N ≈ d/4` you are
  reading noise. Section 3 is the map.
- **Cleanup needs a candidate list.** The raw query output is approximate; you
  must snap it to a known symbol, which means holding the symbols.
- **Ground truth lives outside.** Keep the plain fact list. The trace is a
  computed view of it, rebuilt rather than repaired, because every weight
  depends on the current time.
- **You cannot hand the vector to a language model.** This is the one that
  matters, because the pitch in this space keeps promising it. Every hosted
  model API in production consumes tokens, not vectors. There is no "ghost
  vector" of the user's history you can feed to a model behind a token API.
  What this memory actually does is decide *which few facts are still sharp
  enough to be worth spending tokens on*. That is a real job, and a smaller one
  than the vector-native version implies. Anyone selling you the other thing on
  top of a token API has not tried it.

---

## 7. Tests, and the one that was decorative

```
pytest -q        # 20 tests, under two seconds
```

The suite is written so that breaking a guarded line fails it. That claim is
cheap to make and I checked it by mutation: flip a line in the library, confirm
the suite goes red, restore. Seven mutants plus a neutral witness.

Six bit. One did not, and it is the instructive one:

```python
# before, and green even when contradiction was disabled entirely
assert old.weight == pytest.approx(m.CONTRADICT_FACTOR)
```

That asserts `x == x`. Setting `CONTRADICT_FACTOR = 1.0`, which turns
contradiction into a no-op, moved both sides of the comparison together and the
suite stayed green. It now pins the behaviour, a strict decrease, and a second
test asserts the margin *widens* after a contradiction, which is the effect
anyone would actually notice. Both go red under that mutant now.

I found it by mutating, not by reading. Reading had already passed it twice.

---

## 8. Prior art

- Plate, *Holographic Reduced Representations*, 1995. The original.
- Kleyko et al., *A Survey on Hyperdimensional Computing aka Vector Symbolic
  Architectures*, ACM Computing Surveys.
- Alam et al., *Generalized Holographic Reduced Representations*, 2024.

The classical algebra is theirs. The temporal layer, the z-gate and the
capacity numbers in this repository are mine, and the code is short enough to
check.

This came out of [Dermioz](https://dermiozai.com). If you want to know what
else is in there, that is where to look.

## License

MIT.
