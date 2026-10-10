# Experiment Specification: Dynamic Spectral Compression of Optimizer State

**AdamW and SGD with Momentum using DCT and STFT representations**

| | |
|---|---|
| **Status** | Complete at micro scale (section 37 steps 1–9, DCT only). **Outcome: no-go — see section 45.** |
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
truncated run does not merely approximate badly; it fails to train. Measured in
section 45: at floor 0, m+v DCT-Low ends +138% to +143% above dense at
`K/N` = 50–90%, and about 5% of `v`'s elements reconstruct below zero at every
budget (45.1), so no `K` short of `N` avoids it.

Consequences for the experiment:

* Truncation quality for AdamW must be assessed on **state reconstruction error
  and update fidelity** (sections 19–20), not on how far a trajectory drifted.
* A run that uses a positive floor has changed the update rule and must be
  reported as such. The screening matrix uses `v_floor = 1e-8` and keeps one
  spec-literal `v_floor = 0` run as the control that demonstrates the collapse.
  The floor's own cost, measured with no truncation at all (`K = N`), is +0.19%
  validation loss (section 45.4).

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
(performance optimization) are not started. Measured results are in section 45;
the defects fixed along the way are in section 44.

Run order:

```bash
python spectral_selftest.py                                  # gate: K=N equivalence
python spectral_probe.py --scale micro --steps 3000          # gate: is state compressible?
python train_spectral.py --sweep experiments/mvp_screen.jsonl --out runs/mvp_screen
python spectral_report.py runs/mvp_screen --plots            # one folder = one sweep
python bench.py runs/mvp_screen/<winner>/best.pt             # repo's own quality gate
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

Every figure in this table is from `spectral_selftest.py` at commit `b201750` on
an RTX 3060 Ti (`runs/b201750/selftest.log`, 72 checks, all passing).

| Claim | Status |
|---|---|
| DCT matches an explicit orthonormal DCT matrix; round-trip exact; Parseval holds; retained energy predicts L2 truncation error | **Passing** (`3.17e-15` vs the matrix reference, fp64) |
| `K = N` spectral SGD ≡ `torch.optim.SGD` | **Passing** (`6.21e-16`) |
| `K = N` spectral AdamW ≡ `torch.optim.AdamW`: m+v, m-only, v-only, DCT-Low, dense fallback | **Passing** (`3.02e-13`, fp64) |
| Same, at the training shapes, gradients bounded away from zero | **Passing** (worst element `7.49e-14` of its movement, fp64) |
| Truncated path equals an independent truncated-AdamW reference (DCT-Low, fixed K=10%) | **Passing** (`1.88e-15`) |
| Re-selection carry-over, seeding and eviction counts; DCT-Low prefix invariant | **Passing** |
| K controller: grows now, holds in-band, shrinks after `H`, clamps; covers the steady-state requirement | **Passing** |
| fp32 DCT at the training shapes vs fp64 matrix DCT; round trip; Parseval | **Passing** (`3.51e-7`, `6.14e-7`, `3.36e-7`) |
| Reported persistent state equals the bytes actually held | **Passing**, to the byte |
| fp32 `K = N` is exact AdamW | **Does not hold** — a precision limit (section 44), measured in section 45.5 |

The equivalence rows are the ones that matter: in a compressed run, truncation is
the *only* algorithmic difference from dense training.

### Corrections made after the first selftest execution

1. **Measurement precision bug.** `spectral/metrics.py` downcast every input to
   fp32, silently capping the fp64 checks at ~1e-7 and making an exact identity
   (retained energy predicts L2 truncation error) appear to fail. Real runs use
   fp32 state, so this never affected training. Fixed: only sub-fp32 dtypes are
   promoted.
2. **Churn conflated growth with instability** (see section 41, item 2).
3. **Two tests asserted things the design never claimed**: monotone trajectory
   convergence for AdamW under a rule that diverges (section 15), and zero churn
   for DCT-Low under a *dynamic* `K`. Replaced with the invariants that do hold:
   the DCT-Low active set is always a prefix of the fixed order and evicts
   nothing while growing; at fixed `K` it never churns at all, while TopK does.

### Results to date

Moved to section 45. The text that stood here was measured on code with the
defects listed in section 44; it is preserved in git history (commit `4ffa397`).

---

## 43. Outcome (superseded)

This section held the first screen's results and their later corrections. They
were measured before the section-44 fixes and on different hardware, so they are
not comparable with anything current. Every result is re-measured in section 45;
the original text is preserved in git history (commit `4ffa397`).

---

## 44. Harness audit: defects found and fixed

`spectral_selftest.py` originally ran only in fp64, on tiny CPU tensors, with
Gaussian gradients, at `K = N`. Extending it — to the truncated path, the
training shapes and dtype, the memory accounting, and the step cost — and
auditing the training loop found the defects below. Results measured before a
fix are not used anywhere in this document.

0. **Capacity controller under-provisioned** (fixed in `02bf076`). The spectrum
   estimate compared retained EMA coefficients with a one-step `(1-b).T(g)`;
   for an EMA the steady state is `T(g)` (Adam) or `T(g)/(1-beta)` (SGD
   momentum), so dynamic `K` settled well below `K_max` while its label said
   otherwise. Affected: every dynamic-`K` run. Guarded by: the steady-state
   controller check.
1. **Memory accounting understated the state held.** The report counted 4 B per
   TopK index and none for DCT-Low, but the code stored int64 indices for both
   and cached a full int64 low-frequency order for every eligible shape, even
   under TopK. Affected: every state-size figure. Fixed: int32 TopK indices; no
   per-state indices for DCT-Low (its active set is a slice of one shared int32
   prefix of length `K_max` per shape, counted once as `low_prefix_bytes`); the
   full order is built only transiently at refresh. Guarded by: reported bytes
   must equal the bytes of every tensor the optimizer holds.
2. **The DCT was launch-bound.** Twiddles were rebuilt and both axes transposed
   on every call, so cost was independent of size and dominated the step.
   Affected: every throughput figure. Fixed: cached twiddles with the scale
   folded in, a gather for the even extension, transforms along any axis, `irfft`
   zero-padding, and one batched call for `g`/`g^2` and for `m`/`v`. Accuracy
   unchanged (section 42). Guarded by: informational timing in the selftest.
3. **Evaluation shared the training batch RNG,** so the eval schedule changed
   the training data order. Affected: absolute loss values (comparisons within
   one sweep were fair, since all its runs shared the schedule). Fixed:
   evaluation draws from its own generator, re-seeded at every evaluation, so
   training is bit-identical under any eval schedule and every run scores the
   same windows.
4. **Results carried no provenance.** `summary.json` (and the probe's
   `probe_meta.json`) now record the git commit, whether the tree was dirty, and
   the GPU. `spectral_report.py` stamps every table with them and refuses a
   results folder that mixes them.
5. **`--sweep --out` wrote elsewhere.** Children ignored `--out`; fixed.

**A precision limit, not a defect: spectral state in fp32 is not exact AdamW,
even at `K = N`.** Each fp32 coefficient carries ~1e-7 relative rounding, and
that error lands on every element at the scale of the tensor's *largest* values,
while AdamW divides each element by its *own* `sqrt(v)`. An element whose `v`
sits ~7 decades under the tensor maximum (a near-zero gradient) reconstructs at
or below zero, and its step is then set by the floor, or by `eps` at floor 0.
Dense fp32 rounds each element relative to itself and does not have this
problem. In fp64 the code is exact (section 42), so no transform precision fixes
it — only fp64 coefficients (2x the memory) or a floor (a different update rule).
Its effect on real training is measured in section 45.5.

---

## 45. Results

**Provenance.** Every number in this section comes from commit `b201750`, a
clean tree, on one NVIDIA GeForce RTX 3060 Ti, micro scale (`n_layer=4`,
`n_embd=192`, ~2M parameters, 16 eligible matrices, spectral coverage 88.8%),
seed 42, single seed. Regenerate any table with

```bash
python spectral_report.py runs/b201750/<sweep>          # quality/resource tables
python spectral_probe.py --analyze runs/b201750/probe/probe.csv
python spectral_report.py runs/b201750/fp32_kn_check --weights-vs kn_dense
```

Every delta is against the dense control **of the same sweep**. Dense AdamW
reached val loss 2.2238 in all three training sweeps (bit-identical), while its
throughput varied 190,216 / 196,712 / 198,010 tok/s; throughput deltas therefore
carry about ±4% run-to-run noise, and loss deltas none from the control.

### 45.1 Probe: is optimizer state compressible at all? (section 37 step 4)

Dense AdamW, 3,000 steps, probed every 250 steps over 16 tensors.
`K_required/N` at `tau = 0.99`:

| state | dct_topk | dct_low | identity_topk | dct_shuffled |
|---|---:|---:|---:|---:|
| g | 0.717 | 0.989 | 0.670 | 0.735 |
| m | 0.729 | 0.989 | 0.695 | 0.735 |
| v | 0.431 | 0.893 | 0.739 | 0.650 |

Residual energy left unrepresented at a fixed budget (`tau`-free):

| state, basis | K=1% | K=5% | K=10% | K=25% |
|---|---:|---:|---:|---:|
| m dct_topk | 0.897 | 0.696 | 0.537 | 0.261 |
| m identity_topk | 0.853 | 0.622 | 0.461 | 0.212 |
| v dct_topk | 0.140 | 0.095 | 0.069 | 0.031 |
| v identity_topk | 0.632 | 0.449 | 0.350 | 0.198 |

- **`m` is not DCT-compressible.** `dct_topk ≈ dct_shuffled` (0.729 vs 0.735)
  and `identity_topk < dct_topk` (0.695): two of section 34's failure criteria.
  H1 fails: `m` (0.729) is no more compressible than raw `g` (0.717).
- **H2 is reversed.** `v` is the compressible one: 86% of its energy sits in 1%
  of its DCT coefficients, against 37% in its own coordinates, and it is the only
  state where the DCT beats both controls.
- **Q2: capacity does not evolve.** From step 250 on, `K_required/N` stays in
  0.725–0.730 for `m` and 0.414–0.443 for `v`. By tensor kind, `v` ranges from
  0.339 (`proj`) to 0.494 (`fc`); `m` is 0.725–0.731 everywhere.
- In bytes at `tau = 0.99`, both moments compressed cost 9.27 B/param with TopK
  and 7.53 with Low, against 8.00 dense.

Update fidelity (cosine with the dense AdamW update), DCT-TopK truncation:

| K/N | m truncated, v dense | v truncated, floor 0 | v truncated, floor 1e-8 | v = its mean (1 number), floor 1e-8 | v floored, not truncated |
|---:|---:|---:|---:|---:|---:|
| 1% | 0.250 | 0.233 | 0.952 | 0.918 | 0.998 |
| 5% | 0.447 | 0.209 | 0.947 | 0.918 | 0.998 |
| 10% | 0.573 | 0.187 | 0.946 | 0.918 | 0.998 |
| 25% | 0.777 | 0.188 | 0.956 | 0.918 | 0.998 |

Truncating `m` costs update direction at every budget. For `v`, the cosine is
flat in `K` and set by the floor: about 5% of elements reconstruct below zero at
every budget (0.045–0.051), so at floor 0 those elements step at `m/eps`. With
the floor, `K` coefficients of `v` (0.946–0.956) beat a single number (0.918) by
about 0.03 in cosine.

### 45.2 The screen (`experiments/mvp_screen.jsonl`, 30 runs)

Dense controls: AdamW 2.2238 at 190,216 tok/s; SGD-M 3.5667 at 211,688 tok/s.
Spectral AdamW runs use `v_floor = 1e-8` unless stated. "State" is persistent
optimizer state against dense (all parameters, including the 11.2% that always
keep dense state).

| Config | K/N | Val loss vs dense | State | Reduction | Throughput |
|---|---:|---:|---:|---:|---:|
| AdamW v-only TopK | 1% | +0.44% | 0.565 | 1.77x | −24.3% |
| AdamW v-only TopK | 5% | +0.25% | 0.600 | 1.67x | −24.4% |
| AdamW v-only TopK | 10% | +0.37% | 0.645 | 1.55x | −24.2% |
| AdamW m-only Low | 5% | +28.39% | 0.584 | 1.71x | −21.9% |
| AdamW m-only TopK | 5% | +36.48% | 0.600 | 1.67x | −22.8% |
| AdamW m+v Low | 1% | +41.32% | 0.122 | 8.20x | −28.7% |
| AdamW m+v Low | 5% | +28.90% | 0.162 | 6.18x | −29.5% |
| AdamW m+v Low | 25% | +15.30% | 0.362 | 2.76x | −29.3% |
| AdamW m+v TopK | 1% | +50.10% | 0.130 | 7.71x | −30.4% |
| AdamW m+v TopK | 2% | +46.23% | 0.147 | 6.78x | −30.5% |
| AdamW m+v TopK | 5% | +39.63% | 0.201 | 4.98x | −32.7% |
| AdamW m+v TopK | 10% | +33.00% | 0.290 | 3.45x | −30.6% |
| AdamW m+v TopK | 25% | +24.75% | 0.556 | 1.80x | −31.8% |
| AdamW m+v TopK, `v_floor = 0`, 500 steps | 5% | +146.05% | — | — | — |
| SGD-M Low (dynamic = fixed) | 5% | +2.70% | 0.167 | 5.97x | −24.9% |
| SGD-M TopK, dynamic | 3.96% (final) | +3.72% | 0.176 | 5.68x | −26.7% |
| SGD-M TopK, fixed | 5% | +3.63% | 0.201 | 4.98x | −26.4% |

The `v_floor = 0` row ran 500 steps against a 3,000-step control, so its delta
compares unequal budgets; it is a divergence control, not a quality point.

Controls on the AdamW m+v TopK 5% configuration:

| Varied | Values | Val loss vs dense | Throughput |
|---|---|---|---|
| refresh R | 1 / 10 / 100 / 500 | +41.72% / +39.63% / +32.41% / +32.24% | −59.0% / −32.7% / −25.7% / −25.9% |
| tau (K_max = 10%) | 0.95 / 0.99 / 0.999 | +33.15% / +33.00% / +32.95% | — |

Instrumented twins (shadow dense state; memory and throughput not valid):

| Run | m error | v error | update cos | v clamped | churn | proxy/oracle K | saturation r |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW m+v Low 5% | 0.902 | 0.414 | 0.420 | 11.1% | 0.000 | 1.00 | −0.12 |
| AdamW m+v TopK 5% | 0.971 | 0.308 | 0.176 | 11.0% | 1.000 | 0.97 | 0.02 |
| AdamW v-only TopK 1% | — | 0.386 | — | 8.1% | — | — | 0.13 |
| SGD-M TopK 5% | 0.613 | — | 0.723 | — | 0.843 | 1.53 | −0.88 |

### 45.3 The v-only arm (`experiments/vonly_rerun.jsonl`, 7 runs)

Dense control 2.2238 at 196,712 tok/s.

| Config | K/N | Val loss vs dense | State | Reduction | Throughput |
|---|---:|---:|---:|---:|---:|
| v-only TopK, fixed K | 1% | +0.38% | 0.565 | 1.77x | −26.3% |
| v-only TopK, fixed K | 5% | +0.29% | 0.600 | 1.67x | −26.0% |
| v-only TopK, fixed K | 10% | +0.27% | 0.645 | 1.55x | −26.4% |
| v-only TopK, dynamic | 1% | +0.44% | 0.565 | 1.77x | −26.9% |
| v-only TopK, dynamic | 5% | +0.25% | 0.600 | 1.67x | −26.0% |
| v-only TopK, dynamic | 10% | +0.37% | 0.645 | 1.55x | −25.9% |

All six are within 0.27–0.44% of dense, with no monotone trend in `K`. The
floor alone costs +0.19% at `K = N` (section 45.4), so most of this gap is
attributable to the floor rather than to truncation. Memory reduction falls as
`K` grows; with `m` dense the ceiling as `K -> 0` is about 1.80x.

### 45.4 The m+v convergence curve (`experiments/highk_curve.jsonl`, 17 runs)

Fixed `K`, both moments spectral. Dense control 2.2238 at 198,010 tok/s, peak
244 MiB.

| Selection | K/N | v_floor | Val loss vs dense | State | Reduction | Throughput | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low | 5% | 1e-8 | +28.90% | 0.162 | 6.18x | −31.8% | 236 |
| Low | 10% | 1e-8 | +22.55% | 0.212 | 4.72x | −31.5% | 236 |
| Low | 25% | 1e-8 | +15.30% | 0.362 | 2.76x | −32.1% | 239 |
| Low | 50% | 1e-8 | +6.68% | 0.611 | 1.64x | −31.7% | 242 |
| Low | **75%** | 1e-8 | **+2.84%** | 0.861 | **1.16x** | −31.9% | 246 |
| Low | 90% | 1e-8 | +1.13% | 1.011 | 0.99x | −31.6% | 249 |
| Low | 100% | 1e-8 | +0.19% | 1.111 | 0.90x | −31.9% | 250 |
| Low | 50% / 75% / 90% | 0 | +143.40% / +140.55% / +138.02% | — | — | — | — |
| Low | 100% | 0 | −0.00% | 1.111 | 0.90x | −32.1% | 250 |
| TopK | 25% | 1e-8 | +24.59% | 0.556 | 1.80x | −34.0% | 242 |
| TopK | 50% | 1e-8 | +11.23% | 1.000 | 1.00x | −33.9% | 248 |
| TopK | 75% | 1e-8 | +3.43% | 1.444 | 0.69x | −33.8% | 255 |
| TopK | 100% | 1e-8 | +0.19% | 1.888 | 0.53x | −32.6% | 262 |
| TopK | 100% | 0 | −0.00% | 1.888 | 0.53x | −32.4% | 262 |

- At `K = N` with floor 0 both selections reproduce dense loss (2.2237 vs
  2.2238): the pipeline is correct end to end.
- The floor alone costs +0.19% (`K = N`, floor 1e-8).
- Below `K = N`, floor 0 collapses (+138% to +143%).
- **First point within 5% of dense: DCT-Low K/N = 75%** (+2.84%), at 1.16x state
  reduction, −31.9% throughput, and peak GPU memory 2 MiB *above* dense. The
  next point down, 50%, is +6.68%. Points between 50% and 75% were not run.
- **Within 1% of dense, no point saves memory**: Low 90% is +1.13% at 0.99x.

### 45.5 fp32 at K = N on real gradients (`experiments/fp32_kn_check.jsonl`)

300 steps, weights compared element by element against the dense run:

| Run | Val loss | Eligible weights off by >0.1% of max\|p\| | Worst element |
|---|---:|---:|---:|
| dense, repeated | 3.6473 | 0 (0.0%) | 0 |
| spectral TopK K = N, floor 0 | 3.6472 | 264,794 (15.0%) | 7.2% |
| spectral TopK K = N, floor 1e-8 | 3.6643 | 1,671,523 (94.5%) | 51.7% |

Dense is bit-reproducible, so these differences are real. At floor 0 the
weights diverge measurably and the loss does not; the floor is the larger
perturbation.

### 45.6 Verdict against section 35

| Configuration | Quality ≤ 1% | State reduction ≥ 2x | Throughput loss ≤ 10% |
|---|---|---|---|
| AdamW v-only, K/N 1–10% | **pass** (+0.25% to +0.44%) | miss (1.55–1.77x) | fail (−24% to −27%) |
| AdamW m+v, best within 1% | none saves memory | — | — |
| AdamW m+v, Low K/N = 75% | miss (+2.84%) | miss (1.16x) | fail (−31.9%) |
| SGD-M Low K/N = 5% | fail (+2.70%) | pass (5.97x) | fail (−24.9%) |

**No-go.** No configuration meets all three bars, and none meets two of them.

### 45.7 Findings

1. **DCT-Low beats DCT-TopK at every matched configuration**, though TopK is
   the L2-optimal K-term approximation: m+v 5% 2.8666 vs 3.1050; m+v 25% 2.5641
   vs 2.7741; m-only 2.8552 vs 3.0351; SGD-M fixed 3.6629 vs 3.6963; curve at
   25/50/75% 2.5641/2.3723/2.2870 vs 2.7706/2.4734/2.3000. TopK's active set
   turns over completely at every refresh at R = 10 (churn 1.000), Low's never
   (0.000).
2. **Re-selecting less often is better for TopK**: R = 500 +32.24% against
   R = 1 +41.72%. This is consistent with eviction destroying accumulated
   history (section 16).
3. **Dynamic K does nothing for AdamW.** Every dynamic AdamW run ends at
   `K_max` with a 100% cap-hit rate, and dynamic Low 5% equals fixed Low 5%
   exactly (2.8666 in both). For SGD-M TopK the controller holds 3.96% of a 5%
   cap and lands within 0.1% of fixed K. `tau` is inert (+32.95% to +33.15%). H3
   fails at this scale.
4. **The deployable proxy tracks the oracle K** for AdamW (proxy/oracle 1.00 Low,
   0.97 TopK) and over-estimates it for SGD-M TopK (1.53).
5. **Saturation predicts fidelity loss only for SGD-M** (r = −0.88); for AdamW
   it does not (−0.12, 0.02, 0.13). H4 holds for momentum, not for Adam.
6. **A floor is mandatory below K = N** and costs +0.19% by itself at K = N.
7. **SGD-M tolerates compression** (+2.70% at 5.97x), but SGD-M dense is 60.4%
   worse than AdamW dense (3.5667 vs 2.2238).

### 45.8 What would have to be true for this to work

The measurements point at a specific requirement. A representation of `v` would
need to be (a) accurate in *relative* terms on its small values, since AdamW
divides by them, rather than in L2 energy, which section 45.1 shows a fixed
orthonormal truncation optimizes; (b) non-negative by construction, so no floor
is needed (45.4); and (c) cheap enough to apply every step (45.2–45.4: −22% to
−34%). A factored second moment (Adafactor: one value per row and per column,
`m + n` instead of `m.n`) satisfies (b) by construction and stores 0.69% of `N`
for these matrices — arithmetic on storage only. Its quality and speed were
**not** measured in this experiment; comparing against it is the obvious next
run.
