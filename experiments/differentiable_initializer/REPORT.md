# A differentiable Payne Zero initializer: are its label derivatives usable?

**Date:** 2026-10-01 to 2026-10-02
**Scope:** Payne Zero's five-label atmosphere initializer, ported to JAX and tested as a source of
label derivatives for [`leejjoon/differentiable_stellar_spectroscopy`](https://github.com/leejjoon/differentiable_stellar_spectroscopy) (DSS).
Every number below comes from a script in this directory and a result file under `results/`.

---

## 1. Summary

**Question.** DSS is a differentiable near-infrared synthesis pipeline in JAX. Its atmosphere is not in
the gradient graph. DSS re-converges ATLAS12 outside the graph and supplies the label Jacobian by
finite differences through a `jax.custom_jvp` seam: nine extra solves per Jacobian. DSS tried an
emulated atmosphere first (Kurucz-a1). That attempt failed DSS's Gate G1: the emulator's label
Jacobians were wrong by 3–42% even though its values were good. Can Payne Zero's fast-path
initializer, made differentiable, supply those derivatives instead?

**What was done.**
1. The five-label initializer was ported to JAX (`pz_jax_init.py`). It matches the original Torch/NumPy
   code to ≤ 1e-5 in every field at five test labels. A full label Jacobian costs about 2 ms on CPU.
2. Its output was checked against what DSS expects. The depth grid, the fields and the abundance
   conventions match DSS's exactly, so it can feed `dss.couple.forward.predict` without interpolation.
3. Reference label Jacobians were built from **finite differences of re-converged Payne Zero
   atmospheres**: 34 full solves at ±h and ±2h per label, at a tolerance 100× tighter than production.
   The bases were two stars: the Sun and a metal-poor giant.
4. The emulator's autodiff Jacobian was scored against the references at two levels:
   - the atmosphere itself, using DSS's G1 metrics;
   - the H-band (1550–1600 nm) **spectrum**, which is what a fit consumes.

**Answer.** The emulator's label derivatives are good enough to drive a fit, and much better than the
emulator DSS rejected. They are not good enough to use directly as uncertainties.

| | Sun | metal-poor giant |
|---|---|---|
| ∂T/∂Teff (the cell DSS's emulator failed by 2.6–14.5%) | **0.8%** | **1.5%** |
| flux Jacobian, Teff | 3.6% | 4.5% |
| flux Jacobian, logg | 7.5% | 2.6% |
| flux Jacobian, [M/H] (atmosphere part only) | 1.1% | 3.4% |
| flux Jacobian, [α/M] (atmosphere part only) | 7.0% | 10.6% |
| cosine of every flux Jacobian | ≥ 0.998 | ≥ 0.997 |

The emulator fails in one specific place. The gas-pressure and column-mass derivatives are wrong in
the upper atmosphere (up to about 2× above log τ ≈ −1 for Teff). The [α/M] column is 8–20% off.
This is exactly the error mechanism DSS documented, appearing exactly where its arithmetic predicts.
Most of it does not reach the H-band spectrum, which forms deeper.

**Recommendation for DSS.** Use a hybrid seam:
- **Value:** take the atmosphere from a converged solve, as DSS does now.
- **Tangent during the fit:** take it from the emulator's autodiff.
- **At convergence:** recompute the finite-difference seam Jacobian once, for the uncertainties.

A Gauss–Newton step with a Jacobian 2–10% off still converges to the same optimum, because the
optimum is set by exact residuals. So this removes most of the solver calls without biasing the
result.

**Not established.** A cool giant like Arcturus did not fit in this container's memory. Cool stars
are where DSS found the worst emulator errors, so this is the most important missing check (§8).

---

## 2. Background

### 2.1 Payne Zero's fast path, and why it has no label gradients

`payne_zero_synthesis.synthesize_from_labels` predicts an atmosphere with the initializer, rebuilds
populations, and synthesizes. It never runs the iterative solver. The chain from labels to flux leaves
the autograd graph at four places:

1. **Initializer** (`payne_zero_atmosphere/warm_start.py`):
   - The network runs under `torch.no_grad()`.
   - The PCA decode is done in NumPy.
   - The prediction is written out as a fixed-digit text deck and parsed back. That rounding is
     load-bearing for the certified solver, but it has zero derivative almost everywhere.
2. **Population bridge** (`payne_zero_synthesis/pipeline.py:308`): host-fp64 equation of state, with
   detached iterative solves.
3. **Pipeline ingestion** (`pipeline.py:1109`): `np.asarray(..., float64)` on every column.
4. **Host-side table lookups** inside the continuum opacity (for example `continuum.py:1971`).

Synthesis is differentiable with respect to line parameters at a fixed atmosphere. That is what
`linelist_calibration` uses. It is not differentiable with respect to anything that moves the
atmosphere.

### 2.2 DSS and Gate G1

DSS (`plans/phase1_failure_analysis.md`) tested the Kurucz-a1 atmosphere emulator.
- **Values:** median errors of 0.4–7%.
- **Label Jacobians:** wrong by 3–42%. It failed 17 of 19 held-out cells at the G1 threshold
  (rel-L2 < 5%, cosine > 0.995).
- **Fixes:** Sobolev retraining improved only the trained points and made held-out points worse.

DSS's diagnosis is general. A value-trained network's error is a smooth field of a few millidex
that varies over hundreds of kelvin, and the true slopes are small (about 9×10⁻⁵ dex/K in log T).
So the error field's tilt competes with the true derivative, and a good value fit puts no bound on
the derivative. DSS therefore keeps the atmosphere out of the graph. `dss/couple/atmosphere.py:486`
(`AtlasResolver.state_fn`) maps labels (Teff, logg, [M/H], [α/M]) to a (4, 80) array of T, P_gas, m,
n_e on log τ_Ross = −6.875 + 0.125j. The value comes from ATLAS12 and the tangent from finite
differences of re-converged models.

Payne Zero's initializer has never been tested this way. Payne Zero itself only uses it as a warm
start.

---

## 3. The JAX initializer (`pz_jax_init.py`)

**The checkpoint** (`source_data_files/atmosphere_emulator/five_label/checkpoint.pt`, format
`payne_zero_complete_atmosphere_latent_v2`):
- **Inputs:** 5 features (5040/Teff, logg, [M/H], [α/M], ξ), standardized.
- **Network:** an MLP of 6 hidden layers × 1024 with SiLU activations, outputting 160 PCA coefficients.
- **Decode:** a 160 × 480 PCA basis, giving 80 layers × 6 coordinate fields.
- **Training:**
  - 50,000 converged Payne Zero atmospheres, plus 2,000 for early stopping.
  - Its loss term `derivative_weight` = 0.1 matches profile slopes *in depth*, not label derivatives.
  - Eight named reference stars were excluded from training (`fixed_gate_slugs_excluded`), including
    `sun` and `giant`.

**The port** reproduces `AtmosphereInitializer.predict`:
- standardize the inputs;
- run the MLP in float32, as the original does;
- de-standardize the PCA coefficients and expand them;
- decode the six fields:
  - column mass = cumsum of 10^c₀ (always increasing);
  - T = T_grey(τ) · 10^c₁;
  - P_gas, n_e, κ_Ross = 10^c;
  - g_rad = scale · sinh(c₅).

Two things are deliberately left out. The `np.clip` guards are inactive inside the training support.
The text-deck round trip is a staircase function.

**Parity with the original code** (`results/parity.json`): the maximum relative difference per field
at five labels is ≤ 6.5e-6 for column mass, T, P_gas, n_e and κ_Ross. For the signed g_rad,
normalized by its maximum, it is ≤ 4.2e-5. This is float32 rounding in the MLP.

**Cost:** `jax.jacfwd` of ln(T, P_gas, m, n_e) with respect to 5 labels takes about 1–2 ms after
JIT compilation, on CPU.

**Compatibility with DSS** (`results/checks.json`):
- **Depth grid:** the checkpoint's `standard_rosseland_optical_depth` is exactly
  10^(−6.875 + 0.125j), j = 0…79, which is DSS's canonical grid.
- **Fields:** `JaxInitializer.dss_state` returns DSS's (4, 80) layout of T, P_gas, m, n_e.
- **Abundance conventions:** both codes use identical solar log abundances for Z = 3–99 (maximum
  difference 0.0), the same α set (O, Ne, Mg, Si, S, Ca, Ti), and the same helium fraction (0.078370).
- **Microturbulence:** DSS's seam has no microturbulence label, so ξ must be pinned to the value DSS
  uses.

---

## 4. Reference Jacobians (`fd_reference.py`)

### 4.1 Construction

**Bases.**
- **Sun:** Teff 5777 K, logg 4.44, [M/H] 0, [α/M] 0, ξ 1.0 km/s.
- **Metal-poor giant:** 5000 K, logg 2.50, [M/H] −1.50, [α/M] +0.40, ξ 1.5 km/s. This is an
  in-support point that I chose, not a named excluded star, so it is not verified to be held out.
- **Arcturus** (4286 K, logg 1.66, [M/H] −0.52, [α/M] +0.30) was attempted but did not fit in memory (§4.3).

**Perturbations.** Each of Teff, logg, [M/H] and [α/M] was perturbed by ±h and ±2h, with
h = 25 K, 0.05, 0.05 and 0.05 respectively. The same steps as DSS's V1.3. Abundances are stored to
0.01 dex, so the composition steps are exact. That gives 17 solves per base.

**Solver settings.**
- Production physics: molecules, convection and the full source-line catalogs.
- Each solve is started fresh from the emulator at the perturbed labels.
- Convergence stop: maximum deep-layer relative temperature change below 5×10⁻⁶ on two consecutive
  iterations. The all-layer change must be below 5×10⁻⁵, with at least 5 iterations and at most 60.
  Production uses 5×10⁻⁴ with one consecutive iteration.
- The state is taken in float64 from `result.atmosphere`, never through the text deck.

**Depth registration.** T, P_gas, m and n_e are interpolated with PCHIP in log₁₀ onto the canonical
grid, using each model's own integrated Rosseland depth, as DSS does. The converged models are already
on that grid to a median of 1.0×10⁻⁴ dex (Sun) and 1.5×10⁻⁴ dex (giant). The maximum is about
0.01 dex at the surface.

**References.** Two per cell:
- **Richardson** (4J(h) − J(2h))/3, scored only on layers where |J(h) − J(2h)|/|J| < 5%. This is
  DSS's V1.3 construction.
- **Central ±h**, on every layer, reported with its own uncertainty: the disagreement between the
  forward and backward one-sided differences. An emulator error no larger than that spread is inside
  the reference's noise. It cannot be called a failure.

### 4.2 Convergence

| | solves | converged at 5e-6 | stopped at the cap | mean iterations | mean wall time |
|---|---|---|---|---|---|
| Sun | 17 | 15 | 2: Teff+2h at 1.3e-4, [α/M]+2h at 1.7e-5 | 31.6 | 285 s |
| giant | 17 | 9 | 8: six of them at 6e-6 to 1.3e-5; Teff+2h at 1.1e-4 and [M/H]−2h at 2.8e-4 | 55.1 | 363 s |

**Teff + 50 K never converged at either base.** Its per-iteration change plateaued around 1e-4. DSS saw
the same limit-cycling with ATLAS12. The affected Richardson cells lose layers to the stability
filter. The central ±h reference does not use those points.

**The tolerance is not a limiting error** (`results/checks.json`). Re-solving the Sun base at 2e-5
instead of 5e-6 moves ln T by a median of 2.1×10⁻⁵ over the band. That is 0.5% of the 25 K Teff
signal (4.2×10⁻³).

Wall times are for 4 CPU threads, at roughly 10–15 s per iteration.

### 4.3 Running in a 15 GB container

Stock Payne Zero keeps every source catalog resident in RAM. The atmosphere opacity uses about 7 GB of
line lists, and `np.concatenate` over the three 1.4 GB predicted-line shards peaks near 8.4 GB on its
own. The out-of-memory killer stopped the first two Sun attempts at about 13.9 GB.

**The fix** (`fd_reference.py`, experiment-local; no repository code was changed):
- `build-catalog` writes the three shards, in order, into one 4.2 GB `.npy`.
- The standard and diatomic catalog readers are patched to return memory maps (`mmap_mode="r"`):
  identical bytes, held as file-backed pages.
- **Result:** the Sun solve ran at about 8.9 GB of process memory plus 4.7 GB of reclaimable file
  pages (`results/logs/mem.log`).

**Arcturus** still ran out of memory in iteration 3, at 13.9 GB of process memory. In a cool giant
many more molecular and weak atomic lines pass the opacity selection, so the per-iteration working
arrays grow. It needs a machine with more memory, not a code change.

---

## 5. Atmosphere-level results (`compare.py`, `results/atmosphere_jacobian.json`)

**Metrics:** rel-L2 = ‖J_emu − J_ref‖/‖J_ref‖ and cosine, both over −3 ≤ log τ ≤ 1, on
d ln(field)/d label. **G1 pass:** rel-L2 < 5% and cosine > 0.995. The "spread" column is the
reference's own one-sided uncertainty (§4.1).

**Value errors of the emulator at the base** (median |Δ ln| over the band):

| | T | P_gas | m | n_e |
|---|---|---|---|---|
| Sun | 0.06% | 2.3% | 0.23% | 0.95% |
| giant | 0.06% | 3.3% | 0.41% | 1.1% |

### 5.1 Sun

| cell | Richardson rel / cos / layers | central ±h rel | ref. spread | verdict |
|---|---|---|---|---|
| T / Teff | 0.009 / 1.0000 / 33 | 0.008 | 0.020 | **pass** |
| P_gas / Teff | 0.089 / 0.9960 / 27 | 0.110 | 0.048 | **fail** |
| m / Teff | 0.143 / 0.9920 / 33 | 0.142 | 0.034 | **fail** |
| n_e / Teff | 0.054 / 0.9986 / 33 | 0.053 | 0.046 | marginal |
| T / logg | 0.064 / 0.9980 / 13 | 0.145 | 0.200 | inconclusive |
| P_gas / logg | 0.027 / 1.0000 / 33 | 0.027 | 0.005 | pass |
| m / logg | 0.018 / 1.0000 / 33 | 0.018 | 0.001 | pass |
| n_e / logg | 0.017 / 0.9999 / 33 | 0.015 | 0.017 | pass |
| T / [M/H] | 0.051 / 0.9988 / 27 | 0.102 | 0.115 | inconclusive |
| P_gas / [M/H] | 0.020 / 0.9998 / 33 | 0.019 | 0.014 | pass |
| m / [M/H] | 0.024 / 1.0000 / 33 | 0.024 | 0.012 | pass |
| n_e / [M/H] | 0.032 / 0.9996 / 28 | 0.039 | 0.045 | pass |
| T / [α/M] | 0.159 / 0.9877 / 21 | 0.159 | 0.169 | inconclusive |
| P_gas / [α/M] | 0.078 / 0.9970 / 33 | 0.078 | 0.047 | fail (mild) |
| m / [α/M] | 0.082 / 0.9975 / 33 | 0.081 | 0.047 | fail (mild) |
| n_e / [α/M] | 0.077 / 0.9976 / 25 | 0.087 | 0.064 | marginal |

### 5.2 Metal-poor giant

| cell | Richardson rel / cos / layers | central ±h rel | ref. spread | verdict |
|---|---|---|---|---|
| T / Teff | 0.017 / 0.9998 / 33 | 0.015 | 0.036 | **pass** |
| P_gas / Teff | 0.048 / 0.9998 / 33 | 0.046 | 0.039 | pass |
| m / Teff | 0.084 / 0.9994 / 33 | 0.084 | 0.034 | fail |
| n_e / Teff | 0.037 / 0.9995 / 33 | 0.032 | 0.072 | pass |
| T / logg | 0.059 / 0.9984 / 18 | 0.096 | 0.128 | inconclusive |
| P_gas / logg | 0.011 / 1.0000 / 33 | 0.011 | 0.005 | pass |
| m / logg | 0.016 / 1.0000 / 33 | 0.016 | 0.002 | pass |
| n_e / logg | 0.018 / 0.9998 / 33 | 0.016 | 0.021 | pass |
| T / [M/H] | 0.059 / 0.9983 / 22 | 0.090 | 0.079 | inconclusive |
| P_gas / [M/H] | 0.010 / 1.0000 / 33 | 0.010 | 0.022 | pass |
| m / [M/H] | 0.029 / 1.0000 / 33 | 0.029 | 0.024 | pass |
| n_e / [M/H] | 0.046 / 0.9997 / 25 | 0.052 | 0.033 | pass |
| T / [α/M] | 0.102 / 0.9963 / 25 | 0.163 | 0.493 | inconclusive |
| P_gas / [α/M] | 0.205 / 0.9994 / 33 | 0.205 | 0.040 | **fail** |
| m / [α/M] | 0.088 / 0.9990 / 33 | 0.088 | 0.040 | fail |
| n_e / [α/M] | 0.081 / 0.9994 / 24 | 0.121 | 0.160 | inconclusive |

### 5.3 Reading the tables

- **Temperature against Teff passes at both bases** (0.8%, 1.5%), well inside the reference's
  uncertainty. That is the most important cell for line formation. It is also the one where DSS's
  emulator failed: −14.5% at τ = 1 at the Sun, with a band median of −2.6%, and 17 of 19 held-out
  cells failing overall.

- **Pressure-like fields against logg and [M/H] pass** at both bases (1–5%).

- **Temperature against logg, [M/H] and [α/M] cannot be judged here.** T barely responds to these
  labels, and the forward and backward differences disagree by 8–49%, as much as or more than the
  emulator's apparent error. Deciding these cells would need larger steps or a smoother reference.

- **Pressure and column mass against Teff fail at the Sun, and the failure is localized in depth.**
  `rel_l2_by_log_tau` in the JSON gives:

  | log τ range | P_gas / Teff error | m / Teff error |
  |---|---|---|
  | −5 to −3 | 78% | 79% |
  | −3 to −1 | 93% | 120% |
  | −1 to 1 | 7% | 10% |
  | 1 to 2 | 1.6% | 1.9% |

  In the upper layers the emulator's d ln P/d Teff is about −8×10⁻⁵ K⁻¹ against a true −3.5×10⁻⁵,
  roughly 2× too steep. Below log τ ≈ 0 the two agree to about 2% (for example log τ = 0.875:
  −6.24×10⁻⁵ against −6.39×10⁻⁵). This is DSS's mechanism exactly. The emulator's pressure value
  error is about 2–3%, while the true ∂ln P/∂Teff in the upper atmosphere is only about 0.035 per
  1000 K. Any tilt in a 2% error field across a few hundred kelvin is comparable to the signal.
  In the giant the same pattern is far milder. P_gas/Teff is off by 10.5% (log τ −5 to −3),
  7.6% (−3 to −1) and 4.2% (−1 to 1); m/Teff peaks at 17% (−3 to −1). So the upper-atmosphere
  failure varies from star to star in size, which is also what DSS's mechanism predicts.

- **The [α/M] column is the weakest.** The worst case is the giant's P_gas/[α/M] at 20%.

---

## 6. Spectrum-level results (`flux_jacobian.py`, `flux_split.py`)

An atmosphere-level failure matters only insofar as it changes the spectrum's derivative, which is
what a fit uses. For each label, two central differences of normalized flux were computed with
**identical abundances on both arms**:

- **reference:** Payne Zero synthesis of the atmospheres re-converged at l ± h;
- **emulator:** the converged base atmosphere moved along the emulator's autodiff tangent, ±h in each
  field (ln T, ln P_gas, ln m, ln n_e). This is what the hybrid seam would supply.

A third arm moves only T along the emulator tangent and takes P_gas, m and n_e from the reference.
It attributes the error to temperature or to the pressure-like fields.

The window is 1550–1600 nm, the H band where DSS and Payne Zero calibrate, with R_grid = 100,000
(3,175 pixels). Molecular lines are on, everything is in float64, and synthesis runs on CPU.

**Flux-Jacobian error** (`results/runs/*/flux_jacobian.json`):

| label | Sun rel-L2 | Sun cos | Sun gain | Sun T-only arm | giant rel-L2 | giant cos | giant gain | giant T-only arm |
|---|---|---|---|---|---|---|---|---|
| Teff | 0.036 | 0.9999 | 1.032 | 0.005 | 0.045 | 0.9991 | 1.015 | 0.012 |
| logg | 0.075 | 0.9979 | 1.034 | 0.013 | 0.026 | 0.9999 | 0.979 | 0.031 |
| [M/H] | 0.006 | 1.0000 | 1.001 | 0.002 | 0.013 | 1.0000 | 0.988 | 0.008 |
| [α/M] | 0.051 | 0.9987 | 1.007 | 0.011 | 0.053 | 0.9986 | 0.990 | 0.009 |

"Gain" is the projection of the emulator's Jacobian onto the reference's: ⟨J_emu, J_ref⟩ / ‖J_ref‖².
A gain of 1.03 means the fit would see that label's sensitivity 3% too large.

**[M/H] and [α/M], scored on the atmosphere part only** (`results/runs/*/flux_split.json`). The direct
abundance term (base atmosphere held fixed, abundances at l ± h) is common to both arms, so it flatters
the scores above. Removing it:

| label | Sun: atmosphere share of the total | Sun: emulator error on it | giant: atmosphere share | giant: emulator error on it |
|---|---|---|---|---|
| [M/H] | 59% | 1.1% (cos 0.9999) | 38% | 3.4% (cos 0.9999) |
| [α/M] | 73% | 7.0% (cos 0.9976) | 50% | 10.6% (cos 0.9966) |

**Reading.**
- **Every flux Jacobian points the right way** (cosine ≥ 0.997). Magnitudes are within 1–8% for
  Teff, logg and [M/H], and within 7–11% for the atmosphere part of [α/M].
- **The upper-atmosphere pressure failure barely reaches the H band.** With the emulator's temperature
  derivative and the reference pressure derivatives, every label is at 0.2–3.1%. So the residual flux
  error comes from the pressure-like fields, but is much smaller than their atmosphere-level error,
  because the H-band continuum and most lines form deeper.
- **Exception: giant logg.** There the T-only arm (3.1%) is worse than the full emulator (2.6%), so the
  T and pressure errors partly cancel.
- **Wavelength dependence.** Bands whose lines form higher (strong-line cores, the blue and UV) will be
  more sensitive to the upper-atmosphere pressure error. These numbers do not transfer to them without
  re-measurement.

---

## 7. What this means for DSS

Ranked by how directly the measurements support each option:

1. **Hybrid seam (recommended).** Keep DSS's `AtlasResolver` for the value, with ATLAS12 or the Payne
   Zero solver. Replace the tangent rule in `state_fn`'s `custom_jvp` with the emulator's autodiff
   Jacobian at the current labels. Recompute the finite-difference seam Jacobian once at convergence
   for uncertainties.
   - **Why it's safe:** the fixed point of Gauss–Newton or Levenberg–Marquardt depends only on exact
     residuals. A Jacobian that is 2–10% off, with cosine ≥ 0.997, changes the path and iteration
     count, not the answer.
   - **What it saves:** the nine re-solves per Jacobian, which are DSS's dominant cost per step.
   - **Integration:** `JaxInitializer.log_state_jacobian` already returns d ln(T, P_gas, m, n_e)/d label
     on DSS's grid. DSS's tangent is in linear units, so multiply by the state value.

2. **Emulator for both value and gradient (fully differentiable fast path).** Useful for exploration,
   initialization and sampling-based work at moderate accuracy. It inherits the emulator's value
   errors: about 2–3% in P_gas and about 0.06% in T. DSS's §5 shows these slope errors are smooth over
   hundreds of kelvin, so they bias the labels rather than adding noise.

3. **Not recommended:** emulator derivatives as the final Jacobian for uncertainties. [α/M] (7–11%) and
   logg (up to 7.5%) are too far off, and the cool-star regime is unmeasured.

**Separate observation.** Payne Zero's solver is a reimplementation of ATLAS12 with the same
abundance conventions as DSS. Here it re-converged a model in 3–9 minutes on 4 threads. As the value
source behind DSS's seam it would be in-process and warm-started by the same initializer. Whether it is
consistent enough with DSS's pyKurucz ATLAS12 to keep Phase 3's 0.06 K seam budget would have to be
re-measured.

---

## 8. Limitations

1. **Two stars, one band.** The Sun and one metal-poor giant, in 1550–1600 nm. That is not a survey of
   the label space or of wavelength.
2. **No cool giant.** Arcturus ran out of memory (§4.3). DSS found its emulator's errors worst toward
   cool, molecule-rich atmospheres. This is the most important open check: the same scripts at 4300 K
   on a machine with about 32 GB.
3. **The reference is Payne Zero's own solver, not ATLAS12.** The emulator was trained to imitate this
   solver, which favours it. Against DSS's ATLAS12 references, agreement could be somewhat lower.
4. **The giant base is not a verified hold-out.** The Sun matches an excluded reference star. The giant
   is an arbitrary in-support point and may lie near training models.
5. **Incomplete convergence.** Of the 34 solves, 24 met the 5×10⁻⁶ target. Seven more stopped at the
   iteration cap between 6×10⁻⁶ and 1.7×10⁻⁵, still at least 30× tighter than production. Three are
   unusable: Teff + 2h at both bases, and the giant's [M/H] − 2h, between 1.1×10⁻⁴ and 2.8×10⁻⁴. All
   three are 2h points. They degrade only the Richardson column, where the stability filter drops the
   affected layers. The central ±h references and all flux tests use only ±h solves, and each of those
   converged or stopped below 2×10⁻⁵.
6. **Weak-signal cells.** T against logg, [M/H] and [α/M] are unresolved because the true response is
   smaller than the finite-difference noise at these steps.
7. **Only the five-label initializer.** The eight-label CNO and direct-abundance initializers were not
   tested. Direct-abundance mode, with about 80 [X/Fe] labels, is where autodiff would save the most
   (about 80 solves per Jacobian), and it is unmeasured.
8. **The hybrid seam itself was not run inside DSS.** §7 is a recommendation derived from these
   measurements, not a demonstrated DSS fit.

---

## 9. Reproducing

**Requirements.**
- Python 3.11, `pip install -e .` in the repository root, plus `jax` (CPU is fine) and `scipy`.
- The runtime data: `git lfs pull --include="source_data_files/**"`, about 7 GB, then verify with
  `python payne_zero_atmosphere/install_runtime_data.py --manifest source_data_files/runtime_data_manifest.json verify --root source_data_files`.
- About 15 GB of RAM for the Sun and the metal-poor giant; more for Arcturus.
- About 5 GB of free disk for `work/`.

```bash
cd experiments/differentiable_initializer
export PAYNE_ZERO_DATA_ROOT=$PWD/../../source_data_files NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4

python pz_jax_init.py export                 # work/five_label.npz (44 MB, from checkpoint.pt)
python pz_jax_init.py parity                 # results/parity.json
python fd_reference.py build-catalog         # work/predicted_atomic_lines_all.npy (4.2 GB)

python fd_reference.py solve --base sun              # 17 solves, ~80 min on 4 threads; resumable
python fd_reference.py solve --base metalpoor_giant  # 17 solves, ~100 min

python compare.py results/runs/sun results/runs/metalpoor_giant      # results/atmosphere_jacobian.json
python flux_jacobian.py results/runs/sun --wl 1550 1600 --r-grid 100000
python flux_jacobian.py results/runs/metalpoor_giant --wl 1550 1600 --r-grid 100000
python flux_split.py results/runs/sun
python flux_split.py results/runs/metalpoor_giant
python checks.py                             # results/checks.json (DSS checkout optional, via $DSS_REPO)
```

The recorded solves already exist under `results/runs/`. The analysis steps (`compare.py` onward) can
be re-run without re-solving. `flux_*` still need the runtime data for synthesis.

---

## 10. Files

| path | contents |
|---|---|
| `pz_jax_init.py` | JAX initializer; `export` (weights to `work/`) and `parity` commands |
| `fd_reference.py` | reference solves; `build-catalog`; low-memory catalog patch |
| `compare.py` | atmosphere-level Jacobian scores (Richardson and central ±h with reference spread) |
| `flux_jacobian.py` | spectrum-level Jacobian test (reference, emulator, T-only arms) |
| `flux_split.py` | direct versus atmosphere split for [M/H] and [α/M] |
| `checks.py` | τ-grid alignment, tolerance sensitivity, DSS convention identity |
| `results/parity.json` | port parity at five labels |
| `results/atmosphere_jacobian.json` | §5 tables, including depth-resolved errors |
| `results/checks.json` | §3 and §4 supporting checks |
| `results/runs/<base>/<label>_{p,m}{1,2}.npz`, `base.npz` | each solve: labels, convergence, diagnostics, canonical-grid and native T, P_gas, m, n_e |
| `results/runs/sun/base_tol2e-5.npz` | the Sun base at the looser tolerance (§4.2 check) |
| `results/runs/<base>/flux_jac_<label>.npz` | wavelength and the three flux-Jacobian arms |
| `results/runs/<base>/flux_jacobian.json`, `flux_split.json` | §6 tables |
| `results/logs/` | solver and synthesis logs; `mem.log` holds the memory trace of the patched run. `campaign.log` also contains the out-of-memory Arcturus attempt |
| `work/` (git-ignored) | exported weights and the combined catalog, regenerated by the commands above |
