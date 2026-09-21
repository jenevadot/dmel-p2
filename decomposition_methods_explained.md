# Signal Decomposition Methods for Time Series — EMD / EEMD / CEEMD / CEEMDAN / VMD / STL / IMF
Companion to `DMEL_study_notes.md` and `Informer_internals_and_reading_list.md`.

Context: the DMEL paper (Applied Soft Computing 192, 2026, 114792) name-drops VMD, EMD,
CEEMDAN and STL in its literature review (lines 90-125 of the extracted text) and uses
**CEEMDAN -> IMFs** as its Eq. 1.

---

## 0. The one idea behind all of them: additive decomposition

    q(t) = SUM_i c_i(t) + r(t)

The forecasting logic ("decomposition-ensemble" paradigm) is **divide and conquer**:
each c_i is smoother / narrower-band / more stationary than q, hence easier to predict.
Predict each separately, then sum. Exactly DMEL's Eq. 1 and Eq. 14.

They differ in HOW the components are defined:

| Approach | Components defined by | Adaptive? |
|---|---|---|
| Fourier / FFT | fixed sinusoids at fixed frequencies | NO - prescribed basis |
| Wavelet (DWT) | fixed dyadic scales of a chosen mother wavelet | NO - prescribed basis |
| **STL** | a KNOWN period + smoothness | NO - you supply the period |
| **EMD family** | the data's own local extrema (sifting) | YES - fully data-driven |
| **VMD** | optimization: minimize total bandwidth | PARTLY - you supply K |

---

## 1. IMF — Intrinsic Mode Function

The *output unit* of the EMD family. Huang et al. (1998) define an IMF by two conditions:

1. Over the whole record, the number of local extrema and the number of zero-crossings
   are equal or differ by at most one. (=> oscillates about zero, no riding waves.)
2. At every point, the mean of the upper envelope (through the maxima) and the lower
   envelope (through the minima) is zero. (=> locally symmetric about zero.)

So an IMF is a **zero-mean oscillation whose amplitude AND frequency may both drift in
time** — an AM-FM signal, not a sinusoid.

```
IMF1  /\/\/\/\/\/\/\/\/\/\/\/\/\/\   fastest, small amplitude (noise / storm response)
IMF2   /\  /\  /\  /\  /\  /\  /\
IMF3     /‾\    /‾\    /‾\
...
IMF9        ___/‾‾‾‾‾\___              slowest
resid   ————————————————————          monotone trend
```

**Why this matters:** Fourier forces one fixed frequency per component forever. A river
doesn't behave that way — a flood pulse is a burst of high-frequency energy that exists
for 3 days then vanishes. An IMF represents that natively; a sinusoid cannot (it needs
dozens of Fourier terms smeared across the record). Hence EMD-family dominance in
hydrology / wind / load forecasting.

---

## 2. EMD — Empirical Mode Decomposition
*Huang et al. 1998, Proc. R. Soc. Lond. A 454:903.*

Algorithm = **sifting**:

```
r0 = q(t)
for i = 1, 2, 3, ...:
    h = r_{i-1}
    repeat:                                        # inner sifting loop
        find all local maxima and all local minima of h
        upper = cubic spline through the maxima
        lower = cubic spline through the minima
        m     = (upper + lower) / 2                # local mean
        h     = h - m                              # remove it
    until h satisfies the two IMF conditions       # e.g. Cauchy SD < 0.2
    IMF_i = h
    r_i   = r_{i-1} - IMF_i                        # peel it off
    if r_i has < 2 extrema (monotone): break
residual = r_i                                     # trend
```

### Properties
- Fully **adaptive** — no basis chosen a priori; IMFs derive from the data's own extrema.
- Acts as a **dyadic filter bank**: IMF1 = highest-frequency band, characteristic period
  roughly **doubles** with each subsequent IMF.
- Therefore number of IMFs ~ **log2(N)**.
  Check DMEL: N ~ 1461 daily values, log2(1461) ~ 10.5 -> they obtained 9 IMFs +
  residual. Consistent.
- Handles **nonlinear and non-stationary** data — the whole selling point for runoff.

### Weaknesses (these motivate every later variant)
- **MODE MIXING** — the headline flaw. One IMF contains wildly different scales, or one
  physical scale is smeared across several IMFs. Caused by **intermittency**: an isolated
  spike (a single flood event) abruptly changes the local extrema pattern, so sifting
  "loses track" of the scale it was following.
- **END / BOUNDARY EFFECTS** — splines must be extrapolated beyond the first and last
  extrema -> artifacts at both ends of the record. Brutal for forecasting: the most
  distorted point is THE MOST RECENT ONE, i.e. exactly what your prediction depends on.
- **Not shift-invariant / not causal** — append one new observation and ALL past IMFs
  change. (Central to the leakage problem, §7.)
- Purely algorithmic, no solid theory; sensitive to interpolation, stopping criterion, noise.

---

## 3. EEMD — Ensemble EMD
*Wu & Huang 2009, Adv. Adaptive Data Analysis 1:1. "Noise-assisted data analysis" (NADA).*

Fixes mode mixing by DELIBERATELY ADDING WHITE NOISE:

```
for k = 1 .. K              (K ~ 100)
    q_k = q + eps * w_k                 # w_k = white-noise realization
    run plain EMD on q_k  ->  IMF_i^k
IMF_i = (1/K) * SUM_k IMF_i^k           # ensemble average
```

**Why adding noise HELPS** (sounds backwards): white noise populates the entire
time-frequency plane UNIFORMLY, supplying a *uniform reference scale distribution*.
Sifting then always finds extrema at every scale, so the dyadic filter bank behaves
consistently and never loses track during intermittency. Noise averages out across
realizations; signal does not.

### Remaining problems
- **Residual noise survives** in the IMFs, decaying only as 1/sqrt(K).
- **Reconstruction is NOT exact**: SUM IMF_i != q. Bad — you lose the guarantee that
  summing per-component forecasts even fits the right target.
- Different noise realizations can yield DIFFERENT NUMBERS of IMFs, so averaging "IMF5"
  across runs can mix non-comparable modes.
- Cost = K x EMD.

---

## 4. CEEMD — Complementary EEMD
*Yeh, Shieh & Huang 2010.*
**NOTE: this is NOT the same as CEEMDAN.** Confusing the two is extremely common.

Adds noise in **+/- PAIRS**:

```
q + eps*w   ->  EMD  ->  one set of IMFs
q - eps*w   ->  EMD  ->  another set
average the two
```

Because +eps*w and -eps*w cancel exactly when averaged, the **added-noise residue is
eliminated** rather than merely attenuated. Gives EEMD's anti-mode-mixing benefit with
far less leftover noise and a smaller ensemble. Still does not guarantee exact
reconstruction the way CEEMDAN does.

---

## 5. CEEMDAN — Complete EEMD with Adaptive Noise   [*] THE ONE DMEL USES
*Torres, Colominas, Schlotthauer & Flandrin, ICASSP 2011.*

Two crucial changes vs EEMD:
1. Noise is injected **at every stage** of the decomposition, not once at the start.
2. What gets added is **not raw white noise but the EMD MODES of the noise** —
   E_j(w) = the j-th IMF of noise realization w. At stage j you add noise that already
   lives in the right frequency band.

Let E_1(.) = "first IMF of":

    IMF_1 = (1/K) SUM_k E_1( q       + eps_0     * E_1(w_k) ),   r_1 = q     - IMF_1
    IMF_2 = (1/K) SUM_k E_1( r_1     + eps_1     * E_2(w_k) ),   r_2 = r_1   - IMF_2
    ...
    IMF_i = (1/K) SUM_k E_1( r_{i-1} + eps_{i-1} * E_i(w_k) ),   r_i = r_{i-1} - IMF_i

### Why it wins
- **"Complete" = EXACT RECONSTRUCTION**: q = SUM IMF_i + r_n holds BY CONSTRUCTION
  (each residue is defined by subtraction). This is what licenses DMEL's Eq. 1 and its
  additive recomposition Eq. 14.
- **Deterministic mode count** — same number of IMFs every run, so ensemble averaging
  is coherent.
- Much less residual noise; fewer sifting iterations.
- **"Adaptive noise"** = amplitude eps_i rescaled at each stage (typically to the std of
  the current residue) so injected noise stays proportional to what remains.

### DMEL's settings (§4.1)
noise std eps = 0.2, K = 100 realizations -> 9 IMFs at both stations.
Their Sobol analysis (Appendix A) found `noise_std` the LEAST sensitive parameter —
a genuine robustness result for CEEMDAN.

### Successor worth knowing: ICEEMDAN
*Colominas, Schlotthauer & Torres 2014, Biomed. Signal Process. Control 14:19.*
CEEMDAN can still leave residual noise in modes and produce **spurious early modes**;
ICEEMDAN fixes this by extracting the **local means** of the realizations instead of
their first modes. Current default recommendation. DMEL neither uses nor discusses it —
a fair reviewer question.

---

## 6. VMD — Variational Mode Decomposition
*Dragomiretskiy & Zosso 2014, IEEE Trans. Signal Processing 62:531.*

A COMPLETELY DIFFERENT PHILOSOPHY: no sifting, no recursion, no heuristics. Pose
decomposition as a **variational optimization problem**. Each mode u_k is assumed to be
a narrow-band AM-FM signal concentrated around center frequency omega_k, and all modes
are extracted CONCURRENTLY:

    min over {u_k},{omega_k}
        SUM_k || d/dt [ ( delta(t) + j/(pi t) ) * u_k(t) ] * exp(-j omega_k t) ||_2^2
    subject to   SUM_k u_k = q

Reading it right to left: Hilbert-transform each mode to its **analytic signal**, shift
its spectrum to baseband via exp(-j omega_k t), then penalize the squared L2 norm of its
time-derivative — which IS its **bandwidth**. So: minimize total bandwidth subject to
exact reconstruction. Solved by **ADMM**, with Wiener-filter-style mode updates in the
Fourier domain and each omega_k updated as the **centroid of its mode's power spectrum**.

| Pros | Cons |
|---|---|
| Mathematically well-posed, provable convergence | **Must pre-specify K**, the number of modes |
| Robust to noise (bandwidth prior regularizes) | Also need penalty alpha (bandwidth tightness) |
| No EMD-style mode mixing | Wrong K -> mode duplication or under-splitting |
| Far better boundary behaviour than EMD | Not adaptive in mode count |

That mandatory K is why the literature is flooded with **"VMD optimized by PSO / GWO /
SSA / <metaheuristic>"** papers — they are all just searching for K and alpha.
EMD/CEEMDAN's appeal is that it finds the mode count itself.

---

## 7. STL — Seasonal-Trend decomposition using Loess
*Cleveland, Cleveland, McRae & Terpenning 1990, J. Official Statistics.*
Classical STATISTICS, not signal processing.

    q(t) = T(t) + S(t) + R(t)        trend + seasonal + remainder

Machinery = **LOESS** (locally weighted polynomial regression) smoothers, two nested loops:
- **Inner loop:** detrend -> smooth each *cycle-subseries* (all Januaries together, all
  Februaries together, ...) -> low-pass filter -> seasonal -> deseasonalize -> smooth ->
  trend -> repeat.
- **Outer loop:** compute **robustness weights** downweighting outliers, redo inner loop.
  This is why STL resists spikes.

**Critical limitation:** STL needs a **known, fixed period** — 365 for daily-annual, 7
for daily-weekly (`MSTL` extends to several prescribed periods). It separates the
periodicities YOU NAME; it CANNOT discover arbitrary intrinsic scales. In exchange it is
fast, interpretable, robust, and yields 3 components instead of 9.

This is the STL in the `STL-Transformer-ARIMA` paper DMEL cites (Zeng et al.:
trend -> Transformer, seasonal+residual -> ARIMA) — the same "route components to
different models" idea DMEL generalizes.

---

## 8. Side-by-side comparison

| | EMD | EEMD | CEEMD | CEEMDAN | VMD | STL |
|---|---|---|---|---|---|---|
| Year | 1998 | 2009 | 2010 | 2011 | 2014 | 1990 |
| Type | recursive sifting | noise-assisted ensemble | +/- paired ensemble | staged adaptive-noise ensemble | optimization | LOESS regression |
| Mode count | automatic | automatic (varies!) | automatic | automatic (fixed) | **you choose K** | 3 (fixed) |
| Mode mixing | severe | much better | better | better | avoided | n/a |
| Exact reconstruction | yes | **no** | ~ | **yes ("complete")** | yes | yes |
| Residual noise | none added | 1/sqrt(K) | cancels | small | none | none |
| Boundary behaviour | poor | poor | poor | poor | good | good |
| Needs known period | no | no | no | no | no | **yes** |
| Cost | 1x | Kx | 2Kx | Kx (heavier/stage) | moderate | cheap |
| Theory | heuristic | heuristic | heuristic | heuristic | solid | solid |

Others in this literature: **DWT / MODWT** (wavelets — MODWT is the shift-invariant one
to prefer for forecasting), **SSA** (Singular Spectrum Analysis: trajectory matrix +
SVD), **EWT** (Empirical Wavelet Transform: builds wavelet filters from the observed
spectrum; an EMD/wavelet hybrid).

---

## 9. How this maps onto DMEL specifically

Given the dyadic filter-bank property, the EXPECTED period bands are roughly 2^i days
(rule of thumb, NOT measured from their Figs. 7-8, which are images in the PDF):

| Component | Approx. period | Physical meaning | Shuangpai SE | DMEL routing |
|---|---|---|---|---|
| IMF1-IMF2 | ~2-8 d | individual storm response, measurement noise | 0.189, 0.223 | HF -> Informer |
| IMF3-IMF5 | ~8-64 d | synoptic / sub-seasonal, flood sequences | 0.191, 0.259, 0.217 | HF -> Informer |
| IMF6-IMF9 | ~64-1000 d | **seasonal + annual + trend**, baseflow recession | 0.051 -> 0.0014 | summed into RIMF6 (SE 0.099) -> LSTM |

Two observations for your critique:

1. **The SE values in Table 2 decrease monotonically with IMF index** — exactly what the
   dyadic filter bank predicts. So MRS's entropy ordering largely just recovers the
   frequency ordering CEEMDAN already provided for free. One could split by IMF INDEX
   instead of sample entropy; the paper never tests that baseline.

2. **DMEL's low-frequency channel ~ STL's T + S**, and its high-frequency channel ~
   STL's remainder R, subdivided into 5 bands. So DMEL is effectively "STL with a
   data-adaptive number of remainder bands, plus per-band model routing." That framing
   makes the contribution look more incremental, and suggests a cheap missing ablation:
   **STL (T+S -> LSTM, R -> Informer)**, 3 components instead of 6.

---

## 10. !! THE LEAKAGE TRAP (most important practical point)

**The EMD family is not causal and not shift-invariant.** Each IMF value at time t was
computed from splines through extrema lying on BOTH SIDES of t — including the future.

So if you decompose the ENTIRE 2002-2005 series and THEN split 7:1.5:1.5 (what DMEL's
§3.1/§4.1 describe), every test-set IMF value embeds future information, and the training
IMFs embed the test period. This is **look-ahead bias** — the single most criticized
practice in the decomposition-ensemble literature. It can inflate NSE/KGE dramatically
and is a strong candidate explanation for part of DMEL's 22-212 % margins.

```
LEAKY (common, and what DMEL appears to do)
   decompose FULL series  ->  split  ->  train  ->  test

CAUSAL
   decompose on TRAIN ONLY -> fit models
   for each new test time t:  re-decompose q(1..t) only  -> predict t+1..t+h
   (expanding/rolling window; accept the boundary artifact at t)
```

The catch: the causal version incurs EMD's **end effect** at the forecast origin at every
step — which is why honestly-evaluated decomposition-ensemble models report FAR smaller
gains. If you reproduce DMEL, **running both versions and reporting the gap is the single
most valuable experiment you could add.**

---

## 11. Reading list for this topic

1. **Huang et al. (1998)**, Proc. R. Soc. Lond. A 454:903 — EMD + IMF origin paper.
   Read §4 for the IMF definition and sifting.
2. **Wu & Huang (2009)**, Adv. Adaptive Data Analysis 1:1 — EEMD; clearest explanation of
   WHY noise helps.
3. **Torres et al. (2011)**, ICASSP — CEEMDAN. Only 4 pages, read all of it.
4. **Colominas et al. (2014)**, Biomed. Signal Process. Control 14:19 — ICEEMDAN;
   explains CEEMDAN's residual-noise and spurious-mode defects.
5. **Dragomiretskiy & Zosso (2014)**, IEEE TSP 62:531 — VMD.
6. **Cleveland et al. (1990)** — STL. Or the free and friendlier
   **Hyndman & Athanasopoulos, Forecasting: Principles and Practice, Ch. 3** ->
   otexts.com/fpp3/decomposition.html
7. **Leakage caveat literature** — search *"data leakage in decomposition-based hybrid
   forecasting"* / *"is wavelet-ANN hybrid forecasting valid?"*. **Zhang et al. (2015),
   J. Hydrol.** on misuse of wavelet decomposition in hydrologic forecasting is the
   classic warning.

### Python
- `PyEMD` — `EMD`, `EEMD`, `CEEMDAN` (package name on PyPI: `EMD-signal`)
- `vmdpy` — VMD
- `statsmodels.tsa.seasonal` — `STL`, `MSTL`, `seasonal_decompose`
- `pywt` — wavelets, incl. `swt` (stationary/undecimated ~ MODWT-style)

---

# 12. MEASURED RESULTS — `runoff_emd_deep_dive.py`

Pure-stdlib experiment on a synthetic daily runoff series (N=1461 = 4 years, DMEL's
record length) built from FOUR KNOWN components: baseflow (~1600 d drift), seasonal
(365 d), flood (Poisson arrivals + exponential recession — INTERMITTENT), white noise.
Ground truth known ⇒ every claim is checkable. Output: `runoff_emd_output.txt`.

## 12.0 Validation against Torres et al. 2011 (Fig. 1, 512-sample delta)

| | paper | ours |
|---|---|---|
| EEMD modes | 13 | 7 |
| CEEMDAN modes | 9 | 7 |
| CEEMDAN reconstruction | exact (Eq. 5) | **0.000e+00 ✓** |
| EEMD reconstruction | not exact | 2.783e-03 |

The mode-COUNT gap did **not** reproduce. Why (verified, not guessed):
- A delta has exactly **1 maximum and 1 minimum**, so plain EMD on it returns **ZERO
  modes**. Every mode EEMD/CEEMDAN produce comes ENTIRELY FROM THE ADDED NOISE — which
  is Wu & Huang's original argument for why noise helps at all.
- Mode count is therefore set by the dyadic depth of the NOISE decomposition: our EMD
  gives 8 modes on 512 samples of white noise (log2 512 = 9), and both ensemble methods
  saturate at 7 regardless of `max_imf`.
- The paper's EEMD=13 comes from EEMD **over-splitting**, which depends on the sifting
  stopping criterion. Ours = fixed 8 iterations; theirs = Rilling/Flandrin Cauchy
  criterion, which sifts far more aggressively. Our gentler sifting never manufactures
  the spurious modes, so the gap collapses.

⚠️ Implementation note: the original computes IMF₁ from **raw white noise** (`x + ε₀wᵢ`,
step 1) and only uses noise MODES `Eⱼ(wᵢ)` from stage 2 onward. Easy to get wrong.

## 12.1 Mode mixing is real and measurable WITHOUT ground truth

New metric — **amplitude swing** = std(IMF during storms) / std(IMF in quiet periods).
A genuine IMF has ONE characteristic scale ⇒ stationary amplitude ⇒ swing ≈ 1.

| IMF | period (d) | r(noise) | r(flood) | r(seasonal) | r(baseflow) | amp swing | verdict |
|---|---|---|---|---|---|---|---|
| 1 | 3.2 | 0.126 | −0.017 | −0.061 | −0.036 | **4.7×** | MIXED |
| 2 | 6.9 | 0.018 | 0.501 | 0.001 | −0.006 | **4.7×** | MIXED |
| 3 | 14.5 | 0.015 | 0.362 | −0.034 | −0.021 | **3.4×** | MIXED |
| 4 | 27.8 | 0.027 | 0.355 | −0.050 | −0.025 | **2.6×** | MIXED |
| 5 | 55.1 | 0.041 | 0.242 | −0.020 | 0.057 | 1.4× | clean |
| 6 | 121.8 | 0.035 | 0.231 | 0.406 | −0.180 | 1.2× | clean |
| 7 | 265.6 | −0.008 | 0.154 | 0.460 | 0.156 | 1.2× | clean |
| 8 | 584.4 | 0.021 | 0.101 | 0.350 | 0.243 | 0.7× | clean |
| res | — | −0.028 | 0.055 | −0.010 | **0.921** | — | — |

The single 'flood' process is spread across IMF2–IMF7; top-2 IMFs hold only **51%** of it.

## 12.2 ⚠️ REPLICATION KILLED OUR OWN FIRST CONCLUSION

Repeated on 3 INDEPENDENT series (different storms, different noise):

| method | flood top-2 share | seasonal top-2 share | worst swing |
|---|---|---|---|
| EMD | 57.6% | 90.9% (sd 9.1pt) | 4.8× |
| EEMD | 54.6% | 86.0% (sd 6.2pt) | 4.3× |
| CEEMDAN | **43.3%** | 82.8% (sd 6.0pt) | 4.8× |

- **NOT ROBUST** — the seasonal ranking. On seed 20260205 CEEMDAN looked clearly best
  (78.1% → 91.2%); on seed 7 it was **worst** (96.9% → 77.9%). It FLIPS SIGN. A
  single-seed claim here is pure noise. **This is exactly the trap decomposition-
  ensemble papers fall into** when they report one station and one split.
- **ROBUST** — CEEMDAN spreads the flood process over MORE modes than plain EMD
  (~58% → ~43%, same direction in all 3 series). Not a bug: injected noise creates
  extra extrema, subdividing an impulsive event into more bands. Noise-assisted
  averaging fixes mixing from scale AMBIGUITY; a flood spike isn't ambiguous, it is
  genuinely **broadband**. No linear decomposition can put a 3-day pulse in one octave.
- **ROBUST** — amplitude swing stays 4–5× for ALL THREE methods.

⇒ CEEMDAN's real justification is **exactness**, not de-mixing:
EEMD reconstruction error **21.0%** of signal RMS (residuals averaged in properly);
EMD and CEEMDAN **0.0000%**. EEMD also gave **3 different mode counts** over 100 runs
(7/8/9: 13/76/11), so its "IMF5" averages non-comparable modes.

## 12.3 ⭐ THE LEAKAGE MEASUREMENT (strongest result)

Fix calendar day t₀=1261. Decompose `q[0:L]` for growing L, read the IMF value AT THE
SAME DAY each time. If EMD were causal these would all be identical:

```
 L (days used)  extra future days   IMF1[t0]   IMF2[t0]   IMF3[t0]
          1262                  0     1.4817     0.4099    10.6646   <- honest real-time
          1263                  1     1.2096     0.1046    10.8347
          1266                  4     0.9882    -0.7457     3.3616
          1276                 14     3.3624    -1.6948     1.5572
          1311                 49     3.4737    -1.7479     1.6123
          1381                119     3.4737    -1.7592     1.6194
          1461                199     3.4737    -1.7592     1.6059   <- full-record
```

Over 60 anchor days:

| | mean abs revision | mean abs value | median ratio | **corr(real-time, revised)** |
|---|---|---|---|---|
| IMF1 | 22.33 | 9.11 | 125% | **0.392** |
| IMF2 | 31.56 | 9.32 | 284% | **0.209** |
| IMF3 | 34.41 | 7.55 | 404% | **−0.010** |

Were decomposition causal, those correlations would be **1.000**. The real-time feature
and the leaky feature are effectively **DIFFERENT VARIABLES** — and it's WORST in the
high-frequency modes, precisely the ones DMEL routes to the Informer. The median-ratio
column guards against a small-denominator artifact.

Mechanism confirmed visually: in the last 90 days the last local max is at day 88 of 89,
so the final day(s) are **pure spline extrapolation** — invented, not supported by any
extremum. That artifact sits exactly at the forecast origin.

## 12.4 Bottom line for DMEL

1. ✅ CEEMDAN choice **justified** — but by exact reconstruction (Eq. 14 sums channels),
   not by de-mixing.
2. ⚠️ CEEMDAN **did not isolate the storm response** — it fragmented it further. This is
   the mechanism behind DMEL's own admitted Limitation (1) re: sudden changes. The
   Informer receives 4–5 channels each holding a partial view of the same floods — the
   redundancy MRS was meant to remove.
3. ℹ️ 9 IMFs is **expected** (log2 1461 = 10.5), not a finding.
4. ⚠️ Sample entropy is **largely redundant with IMF index** (period ratios ≈ 2.0,
   complexity monotone in index — their Table 2 shows it). Index-based split untested.
5. 🔴 **Leakage is the top threat to validity.** Does not cancel by giving baselines the
   same preprocessing; it inflates everyone and inflates most whatever leans hardest on
   the HF channels — DMEL by design.
6. 🔬 Decisive missing experiment: `KGE(leaky) − KGE(causal)` at 1/3/5/7 steps.
7. 🔬 Missing baselines: **naive persistence** ŷ(t+h)=q(t), and **ICEEMDAN**.
8. ⚠️ Methodological: DMEL repeats model training 10× (good, + Wilcoxon) but the
   decomposition and split are **fixed**. Per §12.2, that is where the variance lives.
