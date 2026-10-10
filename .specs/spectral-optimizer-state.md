# Experiment Specification: Dynamic Spectral Compression of Optimizer State

**AdamW and SGD with Momentum using DCT and STFT representations**

| | |
|---|---|
| **Status** | Complete at micro scale (section 37 steps 1–9, DCT only). **Outcome: no-go — see section 43.** |
| **Implementation** | `spectral/`, `train_spectral.py`, `spectral_probe.py`, `spectral_selftest.py`, `spectral_report.py` |
| **Reconciliation** | Sections 40–42 map this document to the code and record every deviation |
| **Testbed** | tinyGPT (this repo): one shared model, TinyStories, bits/char on a held-out file |

Section numbers in this document are load-bearing: code comments cite them
(`# spec section 15`). Renumbering breaks those references — append new sections
instead.

---

## 1. Objective

Determine whether the persistent optimizer state used by **AdamW** and **SGD with
momentum** can be represented by a dynamically sized spectral representation with
substantially lower memory requirements while preserving:

* convergence behavior;
* final model quality;
* numerical stability;
* acceptable training throughput.

The primary hypothesis is:

> Optimizer state contains sufficient spatial/structural redundancy that it can
> be represented using `K << N` spectral coefficients, where `N` is the number of
> optimizer-state values in the dense representation.

The experiment **must not assume that such compression exists**. It measures

```
K_required(t)
```

throughout training and determines whether a configured maximum `K_max` is
sufficient.

The underlying question, stated plainly: *are the `N` numbers in an optimizer's
state actually `N` independent numbers?* A negative answer is a publishable
result too — it tells us how much information AdamW's memory really carries.

---

## 2. Scope

The experiment compares six primary configurations:

```
Dense SGD + Momentum
DCT   SGD + Momentum
STFT  SGD + Momentum
Dense AdamW
DCT   AdamW
STFT  AdamW
```

The dense implementations are the control groups. The spectral implementations
retain **full dense model weights** and replace only the persistent
optimizer-state tensors with compressed spectral representations.

This experiment does **not** compress:

* model weights;
* activations;
* parameter gradients;
* embeddings themselves;
* forward-pass tensors.

It specifically targets:

```
SGD:    momentum state
AdamW:  first moment  m
        second moment v
```

This differentiates the experiment from work that parameterizes the **weights
themselves** in DCT coefficient space and reconstructs them during the forward
pass. Our target is different: **retain ordinary dense weights and compress the
persistent optimizer history**.

Likewise, GaLore and related work reduce optimizer-state memory by projecting
gradients into lower-dimensional spaces, and later methods such as COAP address
projection cost and subspace correlation. These are useful comparison points, but
the present experiment investigates **fixed, zero-basis-storage transforms rather
than learned or SVD-derived subspaces**. A fixed basis has no matrix to store, no
subspace to rediscover, and no periodic decomposition cost.

---

## 3. Core research questions

### Q1. Compressibility

How many coefficients are required to approximate optimizer state — is `K << N`?
And does the answer differ between `g`, `m`, and `v`?

### Q2. Evolution

Does required spectral capacity change during training, `K = K(t)`? Does it grow,
shrink, stabilize, oscillate, or behave differently by layer?

### Q3. Saturation

When `K_required > K_max`, can we detect it reliably? And does saturation predict
subsequent degradation in loss, perplexity, BPC, or update fidelity?

### Q4. Resource trade-off

Does the memory reduction compensate for transform cost, inverse-transform cost,
coefficient selection, metadata, and temporary buffers?

### Q5. Optimization quality

Can spectral versions reach approximately the same quality as their dense
counterparts for the same training tokens, model, dataset, initialization,
learning-rate schedule, batch order, and precision?

---

## 4. Dense reference optimizers

### 4.1 SGD with momentum

One precisely defined implementation throughout:

```
m_t     = beta * m_{t-1} + g_t
theta_t+1 = theta_t - eta * m_t
```

Persistent state: `m ∈ R^N`. FP32 state memory: `M_SGD = 4N` bytes.

### 4.2 AdamW

```
m_t   = b1 * m_{t-1} + (1 - b1) * g_t
v_t   = b2 * v_{t-1} + (1 - b2) * g_t^2

m_hat = m_t / (1 - b1^t)
v_hat = v_t / (1 - b2^t)

theta_t+1 = (1 - eta*lambda) * theta_t  -  eta * m_hat / (sqrt(v_hat) + eps)
```

Persistent state: `m, v ∈ R^N`. FP32 state memory: `M_AdamW = 8N` bytes — for a
small model, more memory than the model itself.

---

## 5. Spectral-state formulation

Let `T(x)` be a linear spectral transform and `T^-1(c)` its inverse. Instead of
storing dense state `s ∈ R^N`, store `C = T(s)` but retain only `K` coefficients.
Define `S_K(C)` as the sparse representation containing the active coefficient
set. Reconstruction is

```
s_hat = T^-1(S_K(C))
```

and the central approximation is `s ≈ s_hat`.

---

## 6. DCT representation

For matrix parameters `W ∈ R^{m×n}` use an orthonormal 2D DCT:

```
C = D_m * S * D_n^T
```

Because the transform is orthonormal, Parseval holds:

```
||s||_F^2 == ||C||_F^2
```

which makes coefficient-energy saturation directly interpretable, and makes the
relative L2 reconstruction error of a truncation exactly `sqrt(1 - retained
energy)` — measurable without performing the inverse transform.

Two DCT selection policies must both be tested.

### DCT-Low

Retain the first `K` coefficients in a deterministic low-frequency ordering.

```
+ no coefficient-index storage
+ no active-set churn
+ trivially dynamic K
- assumes useful optimizer state is low-frequency in raw parameter coordinates
```

### DCT-TopK

Retain the `K` coefficients of largest magnitude, `S_K = TopK_i |C_i|`. Under an
orthonormal transform this minimizes squared reconstruction error for a fixed `K`.

```
+ best K-term DCT reconstruction
+ no low-frequency assumption
- requires coefficient indices (real memory)
- requires selection (real compute)
- the active coefficient set changes over time (history loss, section 16)
```

Both are required. **If TopK succeeds but Low fails, the optimizer state is
DCT-sparse but not low-frequency** — a materially different conclusion with
different systems consequences.

---

## 7. STFT representation

The second transform family is a Short-Time Fourier Transform, which provides
localized frequency information:

```
X(r,k) = sum_n  x[n] * w[n - r*H] * exp(-2*pi*i*k*n / L)
```

with `L` the window length, `H` the hop size, `w` the window. Initial
configuration:

```
window = Hann
hop    = L / 2
real-input FFT
```

For matrix optimizer states, test both:

```
STFT-row:   STFT independently across each matrix row
STFT-flat:  flatten row-major, then a single STFT
```

`STFT-row` is the primary variant; `STFT-flat` creates artificial adjacency
across row boundaries.

STFT coefficients are complex. A complex fp32 coefficient stores two fp32 values,
so `K_STFT` **cannot** be compared directly against `K_DCT`. All cross-transform
comparisons must be made in

```
BYTES STORED
```

never in raw coefficient count.

---

## 8. Dynamic K

Each spectral state has `K_min`, `K_current`, `K_max`, with

```
K_min <= K_t <= K_max
```

For AdamW use **independent** capacities `K_m(t)` and `K_v(t)`: there is no
reason to assume the two moments are equally compressible (see H2). For SGD,
`K_momentum(t)`.

---

## 9. Target reconstruction criterion

Configure a target fidelity `tau` (e.g. `0.95`, `0.99`, `0.999`). For an
orthonormal DCT define retained energy

```
R_K = sum_{i in S_K} |C_i|^2  /  sum_i |C_i|^2
```

then

```
K_required = min { K : R_K >= tau }
```

Worked example:

```
tau           = 0.99
N             = 1,048,576
K_req         = 18,500
K_max         = 32,768
K_req / N     = 1.76%
K_req / K_max = 56.5%      -> not saturated
```

Whereas `K_req = 57,000` against `K_max = 32,768` means the representation is
capacity-limited.

`K_required` must be allowed to exceed `K_max` when computed diagnostically —
over-subscription is a first-class output (section 17.5), so the measurement
never clamps.

---

## 10. STFT reconstruction criterion

Raw STFT coefficient energy must **not** be treated as equivalent to
reconstruction energy: overlapping windows make the representation redundant, so
Parseval does not hold. For STFT, use actual reconstructed-state error

```
E_K = ||s - s_hat_K||^2 / (||s||^2 + eps)
K_required = min { K : E_K <= eps_target }
```

The same metric must also be computed for DCT so that DCT and STFT are comparable
under identical reconstruction criteria.

---

## 11. Dynamic K controller

At each spectral refresh step:

1. estimate the spectrum;
2. compute required `K`;
3. grow or shrink the active representation;
4. enforce `K_max`.

```python
required_k = coefficients_required(spectrum, reconstruction_target)
target_k   = min(max(required_k, k_min), k_max)

if target_k > current_k:
    current_k = target_k                 # grow immediately
elif target_k < current_k:
    if low_usage_has_persisted():        # shrink only on sustained evidence
        current_k = target_k
```

Growth must be faster than shrinkage:

```
Grow immediately if capacity is insufficient.
Shrink only if  K_required < 0.75 * K_current  for H consecutive refreshes.
```

This hysteresis prevents the oscillation

```
K = 100, 180, 100, 180, ...
```

which is not merely wasteful: every eviction destroys optimizer history
(section 16), so a flapping controller silently resets state forever.

---

## 12. Spectral refresh interval

Define `R` = steps between active-set re-evaluations. Test

```
R = 1, 10, 100, 500
```

`R = 1` is maximally adaptive at maximum compute cost. Large `R` assumes temporal
persistence of spectral structure — itself an experimental variable, and a
falsifiable one.

Note what `R` does and does not amortize: the state recursion is per-step and
therefore so is the transform. `R` amortizes only *selection* and *sizing*.

---

## 13. Spectral SGD with momentum

Because the transform is linear, `T(m_t) = beta*T(m_{t-1}) + T(g_t)`. Writing
`M_t = T(m_t)` and `G_t = T(g_t)`:

```
M_t = beta * M_{t-1} + G_t
```

With all coefficients retained this is mathematically equivalent to dense
momentum. Compression comes **only** from truncation, `M_hat_t = S_K(M_t)`:

```
m_hat_t   = T^-1(M_hat_t)
theta_t+1 = theta_t - eta * m_hat_t
```

```python
def spectral_sgd_step(param, grad, state):
    grad_coeff  = transform(grad)
    state.coeff = beta * state.coeff + select_active(grad_coeff)
    if refresh_step():
        resize_active_set(state, grad_coeff)
    momentum = inverse_sparse(state.coeff, state.indices)
    param -= lr * momentum
```

---

## 14. Spectral AdamW: compressed-state version

The experiment first approximates **real AdamW state**; it does not redefine Adam
in coefficient space. By linearity:

```
M_t = b1 * M_{t-1} + (1 - b1) * T(g_t)
V_t = b2 * V_{t-1} + (1 - b2) * T(g_t^2)
```

The second line is the important one. We use `T(g^2)`, **not** `T(g)^2`, because

```
T(g^2) != T(g)^2
```

and the latter would silently change what is being approximated. The cost is a
second forward transform per step; that is the honest price of the second moment
and must appear in the timing block.

At update time reconstruct `m_hat = T^-1(M_t)`, `v_hat = T^-1(V_t)`, then apply
ordinary AdamW:

```
theta_t+1 = (1 - eta*lambda) * theta_t
          - eta * (m_hat / (1 - b1^t)) / ( sqrt(v_hat / (1 - b2^t)) + eps )
```

---

## 15. Positivity problem for v

Dense AdamW guarantees `v_i >= 0`. A truncated spectral reconstruction does not,
and may produce `v_hat_i < 0`.

The first implementation uses `v_hat_i <- max(v_hat_i, 0)`, but this must be
**measured, not silently ignored**. Record:

```
fraction_v_negative
negative_v_energy
max_negative_v
fraction_v_clamped
```

Substantial negativity means the spectral approximation of `v` is structurally
poor even when its global reconstruction error looks acceptable.

**Numerical warning — now measured, not hypothesized.** Clamping at exactly zero
collapses the AdamW denominator to `eps`, making the update `|m_hat| / 1e-8`. A
heavily truncated run does not merely approximate badly; it diverges within a few
steps. `spectral_selftest.py` asserts this directly: at `K/N = 0.01`, parameters
reach `>1e3` times their dense magnitude, and the resulting trajectory error is
not even monotone in `K`, because it is measuring blow-up rather than
approximation.

Consequences for the experiment:

* Truncation quality for AdamW must be assessed on **state reconstruction error
  and update fidelity** (sections 19–20), not on how far a trajectory drifted.
* A run that uses a positive floor has changed the update rule and must be
  reported as such. The screening matrix uses `v_floor = 1e-8` (roughly two
  orders of magnitude below a typical `v` at this scale) and keeps one
  spec-literal `v_floor = 0` run as the control that demonstrates the collapse.

---

## 16. Dynamic coefficient history problem

Suppose coefficient 113 is retained for 1,000 steps and then dropped. Its
accumulated history disappears. If it becomes active again later, that history is
unavailable. This is fundamental to dynamic sparse optimizer state.

Therefore record **active-set churn**. For consecutive active sets `S_t` and
`S_{t-R}`:

```
J_t     = |S_t ∩ S_{t-R}| / |S_t ∪ S_{t-R}|
churn_t = 1 - J_t
```

High churn means spectral modes are unstable and compression is discarding
persistent optimizer history. DCT-Low avoids the problem almost entirely, because
increasing `K` merely extends a deterministic frequency frontier.

---

## 17. Saturation metrics

Saturation is a first-class experimental output, not a debugging aid.

### 17.1 Capacity occupancy

```
O_t = K_t / K_max        # ~0.2 = ample spare capacity;  ~1.0 = fully committed
```

### 17.2 Required-capacity ratio

```
S_t = K_required,t / K_max

S < 1   configured capacity is sufficient
S = 1   exactly at capacity
S > 1   the representation is saturated
```

Probably the single most useful saturation metric.

### 17.3 Residual energy at K_max

```
E_residual = 1 - ( sum_{i=1..K_max} c_(i)^2 / sum_i c_i^2 )
```

Answers: even spending the entire configured budget, how much of the state cannot
be represented at all?

### 17.4 Cap-hit rate

```
H_cap = #{ t : K_t = K_max } / #measured steps
```

`H_cap ≈ 1` means the configured limit is too small or the transform is poorly
matched to the state.

### 17.5 Oversubscription

Allow the diagnostic `K_required` to exceed `K_max`, then

```
O_excess = max(0, K_required - K_max) / N
```

This distinguishes *barely saturated* from *massively under-provisioned*.

---

## 18. Spectral entropy

With normalized coefficient energy `p_i = |c_i|^2 / sum_j |c_j|^2`:

```
H       = - sum_i p_i * log(p_i)
H_norm  = H / log(N)          # 0 = concentrated, 1 = white
K_eff   = exp(H)              # effective coefficient count
```

`K_eff / N` is tracked throughout training. It is a continuous measure of
spectral complexity that does not depend on an arbitrary `tau`, and so acts as a
check that conclusions are not artifacts of the chosen threshold.

---

## 19. State reconstruction metrics

For every state `s`:

```
E_s = ||s - s_hat||_2 / (||s||_2 + eps)
```

Measured **separately** for momentum, Adam `m`, and Adam `v`. These are not
averaged into a single number — the asymmetry between `m` and `v` is one of the
things the experiment exists to detect.

---

## 20. Update fidelity

Reconstruction error is insufficient: what moves the weights is the update. For
an exact dense update `u` and spectral update `u_hat`:

```
C_u = (u . u_hat) / (||u|| * ||u_hat||)     # cosine similarity
R_u = ||u_hat|| / ||u||                     # relative norm
E_u = ||u - u_hat|| / (||u|| + eps)         # relative error
```

An ideal compressed optimizer has `C_u -> 1` and `R_u -> 1`. These must be
correlated against spectral saturation; the key hypothesis is

```
K_required > K_max   =>   C_u falls   =>   model quality falls
```

---

## 21. Instrumented mode versus benchmark mode

Two execution modes are **mandatory**.

### Instrumented mode

Purpose: understand what is happening. Maintains shadow dense optimizer states
(dense momentum, or dense `m` and `v`) which are **never used for the spectral
parameter update**. They enable the true `K_required`, state reconstruction
error, update cosine similarity, saturation, and spectral entropy at any step.

These runs are **not valid memory benchmarks**, because they deliberately retain
dense state for measurement.

### Performance mode

Purpose: measure whether the method actually saves resources. Disables shadow
states, full diagnostic spectra, oracle `K`, and expensive logging. Measures only
the state and buffers the spectral optimizer genuinely requires.

This separation prevents diagnostic machinery from invalidating the resource
benchmark.

**Critically, the two modes must follow the same optimization trajectory.** If
instrumentation changed the path taken, instrumented runs would describe a
different experiment than the one being benchmarked. Any quantity the controller
consumes must therefore be available in performance mode.

---

## 22. Memory metrics

Report separately:

* **Persistent optimizer state** `M_persistent` — coefficient amplitudes,
  coefficient indices, `K` metadata, controller state.
* **Transform scratch memory** `M_scratch`.
* **Peak optimizer memory** `M_optimizer-peak`.
* **Total peak training memory** `M_total-peak`.

And compute

```
R_state = M_spectral-state / M_dense-state
```

For AdamW the target is `R_state << 1`. Note that persistent state and scratch
behave very differently under this method, and conflating them would overstate
the result.

---

## 23. Compute metrics

Measure forward time, backward time, optimizer-step time, transform time,
inverse-transform time, selection/TopK time, refresh time, and total step time.

Primary throughput metric: **tokens/sec**. Primary resource-efficiency metric:
**wall-clock time to a target validation loss** — more informative than optimizer
latency alone, since a spectral optimizer that saves memory but needs
substantially more training time may still be unattractive.

Phase timings on a GPU require synchronization to be meaningful; sampling a
subset of steps is acceptable, and the sampling rate must be reported alongside.

---

## 24. Quality metrics

Record `L_train` and `L_validation`. For token-level models:

```
PPL          = exp(L_token_nats)
bits/token   = L_token_nats / ln(2)
```

For character-level normalization:

```
BPC = L_character_nats / ln(2)
```

**Do not report tokenized cross entropy as "bits per character."** BPC is
meaningful only when the likelihood is normalized per character — in this repo
that means dividing by the measured chars/token of the validation stream.

Record: final validation loss, best validation loss, final perplexity, best
perplexity, BPC / bits-per-token, tokens to target PPL, wall-clock to target PPL.

For stability: NaN count, loss spikes, gradient norm, update norm.

---

## 25. Experimental datasets

**Stage A — fast iteration.** Purpose: quick turnaround, easy reproducibility,
BPC measurement, spectral instrumentation. The original specification suggested a
small character-level corpus (Tiny Shakespeare). This testbed instead uses a
**micro-scale configuration of the existing TinyStories pipeline**, because the
repo already computes true bits/char via measured chars/token, and a second data
path would add a divergence risk for no measurement benefit.

**Stage B — token language modeling.** The repo's default TinyStories
configuration; primary metric validation perplexity.

The binding requirement in both stages: all optimizer variants see exactly the
same training examples, tokenizer, batch order, token count, and model
initialization.

---

## 26. Model sizes

```
Micro   ~1-5M params      debugging and instrumentation
Small   ~20-50M params    primary research results
Medium  ~100-300M params  resource scaling experiment
```

Micro establishes correctness. Small establishes behavior. Medium establishes
whether the memory and compute conclusions survive scale — the one that decides
whether any of this matters in practice.

*(In this repo, "micro" ≈ 2M and "small" ≈ 9.6M; the Medium tier is future work.)*

---

## 27. Eligible optimizer-state tensors

Initial spectral compression targets:

```
attention projection matrices
attention output matrices
MLP up projections
MLP down projections
```

Conventional dense optimizer state is kept for biases, normalization parameters,
and small vectors. Embeddings and the LM head initially remain dense; later
experiments may include them separately.

Record

```
spectral coverage = parameters using spectral state / total trainable parameters
```

Without this number, total-memory results cannot be interpreted.

---

## 28. Experiment matrix

At minimum:

| Optimizer | State    | Transform | Selection |
| --------- | -------- | --------- | --------- |
| SGD-M     | Dense    | —         | —         |
| SGD-M     | Spectral | DCT       | Low       |
| SGD-M     | Spectral | DCT       | TopK      |
| SGD-M     | Spectral | STFT      | TopK      |
| AdamW     | Dense    | —         | —         |
| AdamW     | Spectral | DCT       | Low       |
| AdamW     | Spectral | DCT       | TopK      |
| AdamW     | Spectral | STFT      | TopK      |

For each spectral variant test multiple `K_max/N` values

```
1%, 2%, 5%, 10%, 25%, 50%
```

and reconstruction targets `95%`, `99%`, `99.9%`.

The full Cartesian product is unnecessary initially: start coarse, then
concentrate runs near interesting regions.

---

## 29. Fairness rules

Primary comparisons must use identical model initialization, data ordering, batch
size, gradient accumulation, precision, learning-rate schedule, weight decay,
gradient clipping, and training-token budget.

The **first** comparison uses identical optimizer hyperparameters between dense
and spectral, answering:

> What is the effect of compression itself?

A **second**, separate set of experiments may retune spectral hyperparameters,
answering:

> What is the best quality achievable by this optimizer?

**Do not mix those two questions.**

Batch order must come from a dedicated RNG stream, so that a run consuming extra
randomness for instrumentation still sees identical data.

---

## 30. Random seeds

Use `>= 3` seeds for configurations surviving the initial screening, and report
mean and standard deviation for final quality and convergence time. Single-seed
results are sufficient only during development.

---

## 31. Dynamic K telemetry

For every eligible tensor, periodically record:

```
step, layer, state type, N,
K_current, K_required, K_max,
K_current/N, K_required/N, K_current/K_max, K_required/K_max,
retained energy, residual energy,
spectral entropy, effective K,
active-set churn,
reconstruction error,
update cosine similarity
```

For AdamW additionally:

```
negative-v fraction, negative-v energy, v clamp fraction
```

These records must be sufficient to plot `K(t)` over the whole run.

---

## 32. Important plots

**Quality:** validation loss vs tokens; PPL vs tokens; BPC vs tokens.

**Resource:** peak VRAM vs method; optimizer-state bytes vs method; tokens/sec vs
method; time-to-target-quality vs method.

**Spectral behavior:** `K/N` vs step; `K/K_max` vs step; spectral entropy vs
step; residual energy vs step.

**Fidelity:** update cosine similarity vs step; reconstruction error vs step.

**Saturation relationship (the most important diagnostic):**

```
K_required / K_max   vs   update cosine similarity
K_required / K_max   vs   subsequent validation-loss degradation
```

This directly tests whether spectral saturation is predictive.

---

## 33. Primary hypotheses

* **H1 — Momentum compressibility.** Because momentum temporally averages
  gradients, `K_m/N` will be significantly lower than the effective
  dimensionality of raw gradients.
* **H2 — Adam state asymmetry.** The first moment will be easier to compress than
  the second: `K_m/N < K_v/N`.
* **H3 — Dynamic capacity.** Dynamic `K` will outperform fixed `K` at equal
  average memory, because different training phases need different capacity.
* **H4 — Saturation predicts degradation.** When `K_required > K_max`, state
  reconstruction and update fidelity degrade before or together with validation
  quality.
* **H5 — Localized transform.** STFT will outperform global DCT if optimizer-state
  structure is localized rather than globally smooth.
* **H6 — Fixed basis viability.** If DCT/STFT achieve useful compression they may
  offer a systems advantage over adaptive subspace methods, because the basis
  needs no learned matrix storage and no SVD-style basis discovery — a recognized
  engineering concern in low-rank optimizer-state methods.

---

## 34. Failure criteria

The hypothesis is **weak** if most important tensors show `K_required/N -> 1`, or
if

```
DCT ≈ shuffled DCT
```

(i.e. destroying spatial adjacency does not hurt, so the transform is exploiting
pre-existing sparsity rather than structure), or if spectral representations
require large `K` while an adaptive low-rank representation stays compact.

A further control belongs here: **identity-basis top-K**. If keeping the largest
raw state entries is as compact as keeping the largest DCT coefficients, the
transform contributes nothing and only heavy-tailed magnitudes were ever being
measured.

Practical failure occurs if memory savings are small, transform overhead is high,
PPL/BPC materially degrades, training needs substantially more tokens, active-set
churn is high, or `v` reconstruction repeatedly becomes invalid.

---

## 35. Proposed go/no-go target

```
>= 2x reduction in persistent optimizer-state memory
<= 10% total training-throughput degradation
<= 1% relative validation-loss degradation (or similarly small PPL/BPC change)
stable training across >= 3 seeds
```

A result closer to `K/N <= 10%` with comparable quality would be significantly
more interesting, exceeding what simple modest state compression achieves.

These are engineering targets, not theoretical requirements.

---

## 36. Optional baselines

Once the core experiment works, add 8-bit AdamW, GaLore, COAP, and Muon where
applicable.

The first practical question is whether spectral compression beats dense AdamW.
The stronger question is whether it offers a better

```
memory x compute x quality
```

trade-off than existing memory-efficient optimization techniques.

---

## 37. MVP implementation order

```
1.  Dense SGD-M baseline
2.  Dense AdamW baseline
3.  Full DCT instrumentation (no compression yet; verify transform/inverse)
4.  Offline K_required measurement
5.  DCT-SGD with fixed K
6.  DCT-SGD with dynamic K
7.  DCT-AdamW, m only (v left dense)
8.  DCT-AdamW, m + v
9.  DCT TopK dynamic active set
10. STFT-SGD
11. STFT-AdamW
12. Performance optimization
```

Step 4 is the gate: it costs one dense training run and can falsify the premise
before any compressed-training code is trusted. Step 7 is the other high-value
isolation — it separates the easy, linear half of the problem from the difficult
second-moment reconstruction.

---

## 38. Minimal result table

| Method     | State MB | Peak VRAM | K/N avg | Cap hit % | tok/s | Val loss | PPL | BPC |
| ---------- | -------: | --------: | ------: | --------: | ----: | -------: | --: | --: |
| SGD-M      |          |           |    100% |         — |       |          |     |     |
| DCT SGD-M  |          |           |         |           |       |          |     |     |
| STFT SGD-M |          |           |         |           |       |          |     |     |
| AdamW      |          |           |    100% |         — |       |          |     |     |
| DCT AdamW  |          |           |         |           |       |          |     |     |
| STFT AdamW |          |           |         |           |       |          |     |     |

And a second diagnostic table:

| Method     | m error | v error | Update cosine | Spectral entropy | K required/max | v clamp % |
| ---------- | ------: | ------: | ------------: | ---------------: | -------------: | --------: |
| DCT SGD-M  |         |       — |               |                  |                |         — |
| STFT SGD-M |         |       — |               |                  |                |         — |
| DCT AdamW  |         |         |               |                  |                |           |
| STFT AdamW |         |         |               |                  |                |           |

---

## 39. Central experimental outcome

The experiment estimates

```
K*(t, eps)
```

the minimum spectral description needed to preserve useful optimizer behavior to
tolerance `eps`. The important comparison is not `K < N`, but

```
memory saved by K  >  compute + metadata + quality cost introduced by K
```

If that inequality holds over meaningful model sizes, spectral optimizer-state
compression is useful. If it does not, the measurements still tell us something
important: how much actual information AdamW and momentum carry in their
persistent state, how that information evolves during training, and whether it
exhibits any emergent spectral structure at all.

---

# Implementation reconciliation

Sections 40–42 describe the code as built. They are part of the spec because a
specification that no longer matches its implementation is worse than none.

## 40. Implementation map

| Spec | Where |
|---|---|
| 4.1, 13 — SGD-M dense + spectral | `SpectralSGD` in [spectral/optim.py](../spectral/optim.py) |
| 4.2, 14, 15 — AdamW dense + spectral | `SpectralAdamW` in [spectral/optim.py](../spectral/optim.py) |
| 5, 6 — orthonormal DCT, low-frequency order | [spectral/transforms.py](../spectral/transforms.py) |
| 7 — STFT | **not implemented**; `Transform` interface is the extension point |
| 8, 11, 16 — active set, controller, churn | [spectral/state.py](../spectral/state.py) |
| 9, 10, 17–20 — capacity, entropy, fidelity, negative-v | [spectral/metrics.py](../spectral/metrics.py) |
| 21 — instrumented / performance modes | `SpectralConfig.mode` |
| 22, 23, 24, 31 — memory, timing, quality, telemetry | `memory_report()`, `spectral_summary()`, [spectral/telemetry.py](../spectral/telemetry.py), `summary.json` |
| 25, 26, 27, 29 — data, scales, eligibility, fairness | [train_spectral.py](../train_spectral.py) (`SCALES`, `build_param_groups`, seeded batch generator) |
| 28, 30 — experiment matrix, seeds | [experiments/mvp_screen.jsonl](../experiments/mvp_screen.jsonl) (screen is single-seed by design) |
| 32, 38 — plots and result tables | [spectral_report.py](../spectral_report.py) |
| 34 — structure controls | [spectral_probe.py](../spectral_probe.py) (`dct_shuffled`, `identity_topk`) |
| 37 step 4 — offline `K_required` | [spectral_probe.py](../spectral_probe.py) |
| 37 steps 1–3 — correctness | [spectral_selftest.py](../spectral_selftest.py) |

**Status.** Steps 1–9 implemented, DCT only. Steps 10–11 (STFT) and 12
(performance optimization) are not started. Measured results are in sections 42
and 43.

Run order:

```bash
python spectral_selftest.py                                  # gate: K=N equivalence
python spectral_probe.py --scale micro --steps 3000          # gate: is state compressible?
python train_spectral.py --sweep experiments/mvp_screen.jsonl
python spectral_report.py runs/ --plots
python bench.py runs/<winner>/best.pt                        # repo's own quality gate
```

## 41. Deviations from this specification

Each was a deliberate choice; each is reversible by a flag.

1. **The capacity controller runs on a proxy spectrum in *both* modes.** Section
   21 forbids shadow state from affecting the update, and if instrumented mode
   sized `K` from the oracle it would follow a different trajectory than the runs
   being benchmarked. Instead, both modes size `K` from what a deployed
   implementation can see — the retained coefficients, plus the current
   gradient's contribution outside the active set — and instrumented mode
   *measures* the oracle alongside it, reporting `proxy_over_oracle`. If the
   proxy tracks the oracle badly, dynamic `K` is not deployable, and that is a
   result rather than a hidden defect.

2. **Newly activated coefficients keep the current step's gradient
   contribution** (`seed_new_from_grad`, default on). Section 16's history loss
   is preserved and measured: a re-activated coefficient still arrives with no
   accumulated past. Additions and removals are counted **separately**
   (`activated_coefficients`, `evicted_coefficients`), because churn alone
   conflates them: simply growing `K` activates coefficients that never had
   history to lose, and only an eviction destroys accumulated state. What it does not additionally lose is the present step. Zero-filling
   would discard fresh gradient information as well as old history and would
   confound churn with an implementation choice.

3. **`v_floor` exists, defaulting to `0.0`.** The default is exactly section 15's
   clamp. The knob exists because of the numerical warning in that section; using
   a positive floor changes the update rule and must be stated when reporting.

4. **Stage A is micro-scale TinyStories, not character-level Shakespeare**
   (section 25), to avoid a second data path in a repo whose bits/char is already
   correctly normalized.

5. **An identity-basis control was added to the probe** beyond the shuffled-DCT
   control the specification names (section 34). It is arguably the more
   decisive of the two — it is what showed that `m` is *less* compressible in the
   DCT than in its own coordinates.

6. **`v_only` was added alongside the specified `m_only`** (section 37 step 7).
   The spec isolates the first moment on the reasoning that it is the easy,
   linear half; the measurement says the opposite (H2 is reversed), so the
   mirror variant — spectral `v`, dense `m` — is the one the data supports, and
   is now the configuration the screening matrix leads with. Either moment can
   be spectral independently; a dense moment costs no transform.

## 42. Verification status

| Claim | Status |
|---|---|
| DCT matches an explicit orthonormal DCT matrix; round-trip exact; Parseval holds; retained energy predicts L2 truncation error | **Passing** on GPU host (`3.19e-15` vs the matrix reference) |
| `K = N` spectral SGD ≡ `torch.optim.SGD` | **Passing** (`6.21e-16`) |
| `K = N` spectral AdamW ≡ `torch.optim.AdamW`, incl. m-only and the dense fallback | **Passing** (`3.64e-13`) |
| K controller: grows now, holds in-band, shrinks after `H`, clamps, damps oscillation | **Passing** |
| Truncation monotonicity, DCT-Low prefix invariant, section-15 divergence | **Passing** after the test corrections recorded below |
| Section 37 step 4 probe, micro scale | **Executed** — results below |
| Training loop, telemetry, memory accounting, report generator | **Executed** — section 43 |

The equivalence row is the one that matters: it establishes that in a compressed
run, truncation is the *only* difference from dense training. It passes, so
downstream quality numbers measure compression rather than a bug.

### Corrections made after the first selftest execution

1. **Measurement precision bug.** `spectral/metrics.py` downcast every input to
   fp32, silently capping the fp64 checks at ~1e-7 and making an exact identity
   (retained energy predicts L2 truncation error) appear to fail. Real runs use
   fp32 state, so this never affected training — but it would have degraded every
   reported metric's precision. Fixed: only sub-fp32 dtypes are promoted.
2. **Churn conflated growth with instability** (see section 41, item 2).
3. **Two tests asserted things the design never claimed**: monotone trajectory
   convergence for AdamW under a rule that diverges (section 15), and zero churn
   for DCT-Low under a *dynamic* `K` — growing `K` necessarily changes the active
   set. Replaced with the invariants that do hold: the DCT-Low active set is
   always a prefix of the fixed order and evicts nothing while growing; at fixed
   `K` it never churns at all, while TopK does.

### Results to date

**Section 37 step 4 (the go/no-go probe), micro scale, 3,000 steps, one seed.**
`K_required/N` at `tau = 0.99`, meaned over 16 eligible tensors and 12 probe
points:

| state | dct_topk | dct_low | identity_topk | dct_shuffled |
|---|---:|---:|---:|---:|
| g | 0.717 | 0.989 | 0.670 | 0.735 |
| m | 0.729 | 0.989 | 0.695 | 0.735 |
| v | 0.431 | 0.893 | 0.739 | 0.650 |

This meets two of section 34's failure criteria for the first moment:
`dct_topk ≈ dct_shuffled` (destroying spatial adjacency costs nothing) and
`identity_topk < dct_topk` (the state's own coordinates are more compact than its
DCT). In bytes — the only fair comparison, since TopK stores a 4-byte index per
4-byte value — AdamW state becomes **9.27 B/param against 8.00 dense (1.16x
worse)** with TopK, or 7.53 B/param (0.94x) with Low, against a section-35 target
of 0.50x.

H1 also fails: `m` (0.729) is no more compressible than raw `g` (0.717), so
momentum's temporal averaging does not concentrate the spectrum.

**H2 is reversed.** The second moment is the *easier* one under DCT (0.431), and
it is the only state where the transform beats both controls in both directions
— evidence of genuine spatial structure, consistent with `g^2` being
non-negative and carrying a large low-frequency component. Its ceiling is still
only ~14% state savings, well under the bar.

**Full-CSV analysis** (`spectral_probe.py --analyze`) sharpens all of this.

*Sensitivity to `tau`* — the ranking is stable, but `v`'s magnitude is not:

| state | tau=0.95 | tau=0.99 | tau=0.999 |
|---|---:|---:|---:|
| m (dct_topk / identity) | 0.544 / 0.496 | 0.729 / 0.695 | 0.873 / 0.856 |
| v (dct_topk / identity) | **0.133** / 0.520 | 0.431 / 0.739 | 0.724 / 0.888 |

`m` loses to its own coordinates at every target. `v` beats them at every target,
by 3.9x at `tau=0.95`.

*Residual energy at a fixed budget* (threshold-free, and the clearest view):

| state | K=1% | K=5% | K=10% | K=25% |
|---|---:|---:|---:|---:|
| m dct_topk | 0.897 | 0.696 | 0.537 | 0.261 |
| m identity_topk | 0.853 | 0.622 | 0.461 | 0.212 |
| v dct_topk | **0.140** | 0.095 | 0.069 | 0.031 |
| v identity_topk | 0.632 | 0.449 | 0.350 | 0.198 |

**86% of `v`'s energy sits in 1% of its DCT coefficients**, against 37% for its
own coordinates. That is a large, unambiguous structural signal — and the only
one in the entire measurement.

*Q2, answered: capacity does not evolve.* From step 250 onward `K_required/N` is
flat to three decimals (`m` 0.725→0.730, `v` 0.42–0.44). There are no phases, so
dynamic `K` has nothing to adapt to and H3 has no mechanism at this scale;
fixed-`K` is the honest default. Per tensor kind, `v` varies somewhat
(`proj` 0.339 best, `fc` 0.494 worst) while `m` is flat at ~0.73 everywhere.

*Where this leaves the go/no-go.* Compressing `v` alone and leaving `m` dense:

| v budget | B/param | vs dense 8.00 | implied L2 error on v |
|---|---:|---:|---:|
| K/N = 1% | 4.08 | **0.51x** | 0.374 |
| K/N = 5% | 4.40 | 0.55x | 0.308 |
| K/N = 10% | 4.80 | 0.60x | 0.263 |

That is exactly the section-35 bar (0.50x), reached by compressing one moment.
But retained energy cannot decide it: AdamW consumes `1/sqrt(v)`, which amplifies
precisely the small values a low-order reconstruction gets wrong, and any element
driven below zero must be clamped first (section 15). The deciding measurement is
therefore update fidelity (section 20), which `spectral_probe.py` now computes
directly from the dense state at each budget, reporting `m_only` / `v_only` /
`both` separately alongside the negative-`v` and clamp fractions.

**Update fidelity (section 20), first measurement.** Truncating only `m` and
leaving `v` dense is a clean measurement — no clamp is involved — and it is
decisive:

| K/N | update cosine, m truncated |
|---|---:|
| 1% | 0.250 |
| 5% | 0.447 |
| 10% | 0.573 |
| 25% | 0.777 |

Monotone, as an approximation must be, and poor throughout. Even at a 25% budget
— which already costs *more* than dense state once indices are counted — the
compressed optimizer takes a step only 0.78-aligned with the true one. The first
moment is settled: not usefully compressible in this basis.

The `v_only` and `both` columns of that first run are **not** a measurement of
the approximation, and were discarded. They clamped the reconstruction at exactly
zero (section 15 as written), and the negative fraction sits at ~5% at *every*
budget (0.047, 0.050, 0.051, 0.045) because DCT ringing is set by the transform,
not by `K`. Those elements receive a denominator of `eps` instead of `sqrt(v)`,
making their step ~1e5 times too large; they then dominate the update norm, and
the resulting cosine (0.19–0.23) measures the singularity rather than the
reconstruction — which is why it failed to improve with `K` while `v`'s residual
energy fell from 0.140 to 0.031.

Two conclusions follow regardless of the re-measurement:

1. **A floor on the reconstructed second moment is mandatory, not optional**, for
   any spectral-`v` scheme. The clamped fraction does not shrink appreciably with
   budget, so there is no `K` at which the problem goes away.
2. **Update-fidelity metrics must be reported with `update_rel_norm` beside
   them.** A cosine alone cannot distinguish a poor approximation from a few
   singular elements; the relative norm makes it obvious at a glance.

The probe now evaluates each budget under three policies — `clamp0` (spec
literal), an absolute floor, and a floor at a thousandth of the reconstruction's
own mean (deployable, since for an orthonormal DCT the mean is the DC
coefficient) — and reports cosine and relative norm for each.

**Second measurement, with floors. The result is flat in `K`:**

| floor | 1% | 5% | 10% | 25% |
|---|---:|---:|---:|---:|
| `clamp0` (spec literal) | 0.233 | 0.209 | 0.187 | 0.188 |
| floor ≈ 1e-9 (`rel1e-3`) | 0.730 | 0.664 | 0.650 | 0.682 |
| floor = 1e-8 (`abs1e-8`) | 0.952 | 0.947 | 0.946 | 0.956 |

Across that range the reconstruction of `v` improves 4.5x (residual energy
0.140 → 0.031) and the update cosine does not move. It is monotone in the
**floor** and flat in **K**. So update fidelity here is not a function of how
well the second moment is reconstructed at all — it is a function of how the
small values are handled.

**This is the mechanism behind the whole negative result, and it is not specific
to the DCT.** Coefficient energy is dominated by `v`'s *large* entries, which any
L2-optimal truncation reproduces first. But AdamW divides by `sqrt(v)`, so the
update is governed by `v`'s *small* entries — precisely where a truncated
reconstruction rings, and precisely where relative error is largest. The
section-9 retained-energy criterion is therefore the **wrong objective for a
second moment**: it optimizes accuracy where the optimizer is least sensitive.
Any future attempt at compressing `v` needs a relative-error or log-domain
criterion, not an energy one.

Two controls were added to test whether the transform contributes anything at
the update level at all: `v_scalar` replaces the entire second moment with a
single number (its mean), and `v_floor_only` applies the floor without
truncating. `v_scalar` is `K`-independent by construction, so if `v_only` does
not beat it, the spectral representation is decoration — the same question
`identity_topk` answered for the energy criterion.

Caveats: micro scale (192-wide matrices), early training, single seed. Scale may
move the magnitudes; it is unlikely to reverse the within-tensor shuffle and
identity controls, which compare a tensor against itself.


---

## 43. Outcome (micro scale, 30-config screen, one seed)

The full screening matrix ran: two dense controls and 28 spectral configurations,
3,000 steps each on identical data, initialization and schedule.

### Quality and resources

| Config | val loss | vs dense | state | reduction | tok/s | throughput |
|---|---:|---:|---:|---:|---:|---:|
| **AdamW dense (control)** | 2.2018 | — | 1.000 | 1.0x | 194,542 | — |
| AdamW v-only, K=1% | 2.2129 | +0.50% | 0.561 | 1.78x | 108,110 | −44.4% |
| AdamW v-only, K=5% | 2.2108 | +0.41% | 0.561 | 1.78x | 108,296 | −44.3% |
| AdamW v-only, K=10% | 2.2150 | +0.60% | 0.562 | 1.78x | 109,243 | −43.8% |
| AdamW m-only, K=5% Low | 2.8367 | +28.8% | 0.578 | 1.73x | 109,999 | −43.5% |
| AdamW m+v, K=25% Low | 2.5297 | +14.9% | 0.334 | 2.99x | 75,845 | −61.0% |
| AdamW m+v, K=5% Low | 2.8478 | +29.3% | 0.155 | 6.45x | 76,500 | −60.7% |
| AdamW m+v, K=1% TopK | 3.1629 | +43.7% | 0.125 | 8.00x | 72,850 | −62.6% |
| AdamW m+v, K=5%, `v_floor=0` | 5.5629 | +152.7% | 0.163 | 6.13x | 65,851 | −66.2% |
| **SGD-M dense (control)** | 3.5516 | — | 1.000 | 1.0x | 206,758 | — |
| SGD-M K=5% Low | 3.6368 | +2.40% | 0.156 | 6.41x | 116,977 | −43.4% |
| SGD-M K=5% TopK | 3.6698 | +3.33% | 0.172 | 5.81x | 114,796 | −44.5% |

### Verdict against section 35

Only one configuration has acceptable quality — spectral `v`, dense `m`:

```
quality      <= 1% degradation   ->  +0.37 .. +0.62%   PASS
memory       >= 2x reduction     ->  1.55 .. 1.77x     MISS
throughput   <= 10% loss         ->  -39 .. -44%       FAIL
```

(Ranges span the screen and the post-fix re-run below. Memory is now the re-run's
correctly provisioned figures: 1.77x at K/N = 1%, falling as K grows; the
screen's 1.78x reflected K/N ≈ 0.6%.)

**No-go.** The one variant that trains well costs ~40% of throughput to save at
most 1.77x of optimizer state, which is the section-39 inequality failing outright: the
memory saved does not cover the compute spent. Some of that 44% is
implementation (the DCT here uses a 2N even extension rather than a Makhoul-style
N/2 FFT, worth perhaps 2x), but no plausible constant closes a 4x gap to the 10%
gate.

### Why the v-only result does not rescue the premise

Quality is **flat in `K`**: with `K` pinned exactly (post-fix re-run below),
2.2170 / 2.2126 / 2.2145 at 1% / 5% / 10%, non-monotone and within 0.006 nats.
Ten times more coefficients buys nothing. This is the training-level echo of the fidelity measurement in section 42, and it says the
spectral representation is not what makes the run work — AdamW is simply
insensitive to the accuracy of its second moment, as long as the denominator is
kept away from zero.

That insensitivity is a known property, and it is much better exploited by
existing methods. Adafactor factorizes `v` into row and column vectors: `m + n`
values instead of `m . n`, which for these matrices is **0.65% of N stored as
4-byte values (0.026 B/param), against 0.08 B/param for DCT-TopK at K/N = 1%**.
It is non-negative by construction, so it needs no floor and has no clamp
pathology, and it costs no transform. It reaches the 2x bar this method missed,
with none of the throughput penalty. A spectral second moment is dominated on
every axis.

### Findings worth keeping

1. **DCT-Low beats DCT-TopK for training, consistently**, despite TopK being the
   L2-optimal K-term approximation: 2.8478 vs 2.9120 (m+v, K=5%), 2.5297 vs
   2.5812 (K=25%), 2.8367 vs 2.9211 (m-only), 3.6368 vs 3.6698 (SGD-M). Better
   reconstruction produces a worse optimizer. The mechanism is section 16:
   TopK's active set churns, and every eviction destroys accumulated history,
   while Low's deterministic prefix never evicts. **Active-set stability matters
   more than reconstruction optimality** — a result that inverts the obvious
   choice of selection policy.
2. **Dynamic K did nothing** (H3 fails). With `K_required/N ≈ 0.73` and `K_max`
   at a few percent, the controller pins at `K_max` permanently: dynamic and
   fixed-K runs are bit-identical (3.6368 both). `tau` is likewise inert
   (2.7788 / 2.7823 / 2.7886 for 0.999 / 0.99 / 0.95) for the same reason.
3. **`R = 1` is the best refresh interval** (2.7946 vs 2.8919 at R=100 and 2.9156
   at R=500) but costs a further 27% throughput. Re-selecting often helps because
   it keeps the active set matched to a spectrum that keeps moving — which is
   itself evidence the representation is a poor fit.
4. **Section 15 confirmed in training**: the spec-literal `v_floor = 0` run
   stalls at 5.5629 (+152%), never recovering.
5. **SGD momentum tolerates compression well** (+2.4% at 6.4x state reduction) —
   but SGD-M dense is itself 61% worse than AdamW dense (3.5516 vs 2.2018), so a
   compressible momentum buys nothing anyone wants.

### Correction: the screen never approached K = N, and its K labels overstate K

Two issues surfaced when the sweep's reported state sizes were reconciled against
`K_max`.

**The capacity controller was under-provisioning.** It compared retained
coefficients (which hold accumulated EMAs) against a *one-step* gradient
contribution `(1-b).T(g)`. For an EMA `m <- b.m + (1-b).g` a persistent gradient
converges to `g` itself, so the out-of-set energy looked 10–20x smaller than it
is, and `K` settled at roughly half its cap. Backed out of the reported state
sizes:

| config | `K_max` | K/N actually used |
|---|---:|---:|
| AdamW m+v TopK | 5% | 2.6% |
| AdamW m+v TopK | 10% | 5.2% |
| AdamW m+v TopK | 25% | 12.6% |
| AdamW v-only TopK | 1% / 5% / 10% | 0.56% / 0.56% / 0.68% |

Fixed: the spectrum estimate now uses the steady-state magnitude
(`T(g)` for Adam's moments, `T(g)/(1-beta)` for SGD momentum) while newly
activated coefficients are still seeded with the true one-step contribution. All
numbers in this section predate the fix; the verdict is unaffected, since larger
`K` costs more memory for the modest quality gain the curve already shows.

**A claim of mine was wrong.** I reported the three v-only runs as evidence that
training quality is flat in `K`. They are not: their actual `K/N` was
0.56% / 0.56% / 0.68% — nearly identical — so they never tested `K` at all. The
flat-in-`K` finding then rested solely on the probe's offline sweep (section 42).
It has since been measured in training; see the re-run below.

**The convergence region was never measured.** The screen capped at 25% and
reached 12.6%, while `K = N` is the point where the optimizer is algebraically
AdamW. The `experiments/highk_curve.jsonl` arm fills this in with `--fixed-k` at
K/N = 25 / 50 / 75 / 100%, where the break-even points are:

```
DCT-TopK (4B value + 4B index)   costs as much as dense state at K/N = 50%
DCT-Low  (4B value, no index)    costs as much as dense state at K/N = 100%
probe: m needs K/N = 73% (TopK) / 99% (Low) to retain 99% of its energy
```

Loss must return to the dense control at K/N = 1.0 — but by then TopK occupies
1.89x the memory of the dense state it replaced. If quality only recovers past
break-even, the method has no operating region at all, which is the structural
form of this negative result.

### Re-run of the v-only arm after the controller fix

`experiments/vonly_rerun.jsonl`: the v-only arm on the fixed controller, the same
budgets under `--fixed-k` (K pinned exactly, no controller), and a same-session
dense control. Micro scale, 3,000 steps, seed 42, all on one GPU. The selftest
gate passed first.

| Config | K/N used | val loss | vs dense | state | reduction | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| **AdamW dense (control)** | — | 2.2033 | — | 1.000 | 1.0x | 135,986 |
| v-only dynamic, `K_max` = 1% | 1.00% | 2.2151 | +0.54% | 0.565 | 1.77x | 82,102 |
| v-only dynamic, `K_max` = 5% | 5.00% | 2.2114 | +0.37% | 0.600 | 1.67x | 82,667 |
| v-only dynamic, `K_max` = 10% | 10.00% | 2.2139 | +0.48% | 0.645 | 1.55x | 82,351 |
| v-only fixed K = 1% | 1.00% | 2.2170 | +0.62% | 0.565 | 1.77x | 82,810 |
| v-only fixed K = 5% | 5.00% | 2.2126 | +0.42% | 0.600 | 1.67x | 82,155 |
| v-only fixed K = 10% | 10.00% | 2.2145 | +0.51% | 0.645 | 1.55x | 82,325 |

1. **The fix works.** The dynamic controller now sits at its cap (cap-hit
   0.997–0.999), so dynamic and fixed `K` are effectively the same runs — consistent
   with `K_required/N` being far above any `K_max` tested.
2. **Memory now moves with `K`, and only downward.** With `m` dense, the eligible
   state is `(4 + 8·K/N) / 8` of dense, and 11.2% of state is non-eligible
   (embeddings, head, norms). The ceiling as `K -> 0` is ~1.80x, so **v-only
   cannot reach the 2x bar at any `K`**.
3. **Quality is flat in `K`**, now measured in training: the six runs span
   2.2114–2.2170 with no monotone trend. Single seed, so the spread cannot be
   separated from seed noise; the two dense controls (screen and re-run) differ
   by 0.0015 on identical settings.
4. **TopK churn is very high once `K` is provisioned** — 0.89 / 0.96 / 0.97 at
   1 / 5 / 10%, against ~0.07 in the under-provisioned screen runs — and quality
   does not care. That fits AdamW's insensitivity to the second moment's
   accuracy, given a floor.
5. **Throughput −39%** against the same-session control. This GPU is slower than
   the screen's (dense 136k vs 195k tok/s), so the percentage is not strictly
   comparable with the screen's −44%; both are far outside the 10% gate.

### What would have to be true for this to work

The measurements point at a specific, falsifiable requirement rather than a
vague "needs more work". Any future attempt needs a representation that is
(a) accurate in *relative* terms on the small values of `v`, not in L2 energy;
(b) non-negative by construction, so no floor is needed; and (c) cheap enough to
apply every step. Adafactor already satisfies all three. A fixed orthonormal
transform satisfies none of them, and section 42 explains why: energy-optimal
truncation puts its error exactly where `1/sqrt(v)` is most sensitive.
