# DMEL — Study Notes
Wang et al., *DMEL: A novel dual-modal ensemble learning architecture for multi-step runoff prediction*, Applied Soft Computing 192 (2026) 114792.

---

## 0. Problem being solved

- **Runoff** = river discharge (m³/s) at a gauge, daily resolution.
- Daily runoff is **nonlinear + non-stationary + heavy-tailed**. Shuangpai: min 3.3, max 6420, mean 332, std 509 m³/s (flashy). Fenghuang: min 0.2, max 128, mean 12.4 m³/s (smooth).
- Two gaps the paper targets:
  1. Classic "decomposition–ensemble" pipelines use **one model type on all IMFs** → model/data mismatch, many subseries → compute + error accumulation.
  2. Most papers only do **1-step** prediction → useless lead time. Effective horizon is ~3 days in practice.
- Claim: extend effective horizon **3 → 7 days** without losing accuracy (7-step NSE/KGE > 0.8 at both stations).

---

## 1. Multi-step runoff prediction

- Input window L = **7 days** of runoff; output h ∈ **{1, 3, 5, 7}** days.
- Two standard strategies:
  - **Recursive**: feed prediction back → compounding error.
  - **Direct / generative** (used here, via Informer's generative decoder): all h steps in a single forward pass → no self-error feedback. Paper credits this for the slow accuracy decay.
- Degradation, 1→7 step (Shuangpai NSE): Informer −10.8 %, Transformer −44.8 %, LSTM −44.4 %, RNN −48.1 %, TimesNet −53.6 %, PatchTST −52.3 %. DMEL stays at NSE 0.8255 / KGE 0.8759.

---

## 2. Modal Recognition Strategy (MRS)

### Normal practice (baseline paradigm)
1. EMD / EEMD / CEEMDAN / VMD / STL → k IMFs.
2. One predictor per IMF, usually all the same architecture.
3. Sum predictions.
Problems: 9–12 models, high input dim, cross-component error accumulation, one model forced to fit both chaotic IMF1 and near-linear IMF9.

### MRS = CEEMDAN + Sample Entropy recombination + routing
1. **CEEMDAN** (noise_std = 0.2, 100 realizations) → IMF1..IMF9 + residual. Adaptive noise avoids mode mixing.
2. **Sample Entropy** per IMF, m = 2, r = 0.2·std.  SE = −ln(A(m+1)/B(m)).
3. **Recombine** IMFs with similar SE:  RIMF_j = Σ_{i∈G_j} IMF_i.
   - Shuangpai (SE_orig = 0.1769): IMF1..IMF5 (SE 0.19–0.26) kept alone; IMF6+7+8+9 (0.0509→0.0014) → RIMF6 (SE 0.099). **9 → 6 inputs (−33 %)**
   - Fenghuang (SE_orig = 0.4523): IMF1, IMF2 alone; IMF3+4+5 → RIMF3 (SE 0.6825); IMF6..9 → RIMF4 (SE 0.1074). **9 → 4 inputs (−55.6 %)**
4. **Route** by SE relative to SE(original series):
   - SE ≥ SE_orig → **high-frequency** → Informer
   - SE ≈ 0–0.1  → **low-frequency**  → LSTM

### Payoff (ablation vs CEEMDAN-only, Tables 7–9)
- Runtime −28.7 to −44.2 % (MRS-Informer vs CEEMDAN-Informer).
- 7-step Shuangpai MRS-Informer vs CEEMDAN-Informer: NSE +9.7 %, KGE +23.4 %.
- Gains are *smaller in 1-step* (NSE +4.0 %) and *larger in multi-step* → MRS mainly suppresses high-frequency noise propagation / error accumulation.

### Adaptivity (Appendix B)
SE threshold is dynamic: inject 5–10 % Gaussian noise → IMF entropies rise → borderline components get **reclassified HF and rerouted to Informer**, shielding the LSTM trend channel. 7-step KGE only drops 0.8759 → 0.8541 (noise-5 %) → 0.8319 (noise-10 %).

### Why Informer (HF) + LSTM (LF)
| Channel | Signal | Model | Reason |
|---|---|---|---|
| High-freq | spiky mutations, flood peaks, weak local autocorr. | Informer | ProbSparse attention O(L log L), distilling (Conv1d+ELU+MaxPool), generative decoder → all h steps at once; strong at abrupt change + long-range deps. |
| Low-freq | smooth seasonal trend, baseflow | LSTM | Gated recurrence = smoothness/memory prior; cheap (130 s vs 359 s); attention wasted on a near-deterministic curve. |

Evidence for the allocation:
- **Swap ablation** MRS-LSTM-Informer: 7-step KGE 0.6711 vs DMEL 0.8759 (Shuangpai) — big loss.
- **SHAP**: HF high values → large positive SHAP (peaks); the single LF component has the **largest mean |SHAP|** (dominant trend carrier).
- **LIME**: peak days (7/2, 11/1) → HF dominates the local explanation; dry stable days (11/11, 12/28) → LF dominates.

### Caveats / critique
- "High/low frequency" is really **high/low sample entropy** (complexity proxy, not a spectral band). Fenghuang RIMF3 with SE 0.68 is labelled high-frequency.
- Threshold = entropy of raw series → arbitrary, station dependent.
- Decomposition appears to be applied to the **whole series before** the 7:1.5:1.5 split → classic decomposition-based **data leakage** risk, not discussed.
- Recomposition is a plain **sum**, no learned combiner (cf. Wu et al. [37] using Random Forest).
- Sobol (Appendix A): sensitive params are `embed_dim` (Shuangpai) and `tolerance` (Fenghuang); `noise_std` insensitive at both → the partition is data-driven, not transferable.
- No precipitation forcing in the main experiments → cannot physically anticipate an unseen storm.

---

## 3. HF vs LF separation — yes, explicitly

```
q(t) --CEEMDAN--> IMF1..IMF9        (IMF1 fastest .. IMF9 ~trend)
          |
   SampleEntropy(IMFi)
          |
   group similar-SE IMFs (Eq.12) -> RIMF1..RIMFj
          |
   compare SE(RIMFj) with SE(q)
     |-- SE >= SE(q)  -> HIGH FREQ -> INFORMER (per component)
     |-- SE ~ 0-0.1   -> LOW  FREQ -> LSTM
          |
   yhat = sum(yhat_Informer) + yhat_LSTM      (Eq.14, superposition)
```
- Separation happens **in preprocessing**, not via a learned gate / mixture-of-experts.
- Each RIMF has its **own trained model instance** (Table 7 counts subsequences 6 / 4).
- Proof it works on fast dynamics: **Peak Error % (EP)** — DMEL 0.09 % on 2005/7/2 (1-step), 7.45 % on the largest flood peak 6/21, and still 7.84 % / 9.81 % at 7-step where baselines reach 100–168 %. Violin plots (Figs. 19–20): narrowest error distribution at every horizon.

---

## 4. Training pipeline

```
STAGE 0  DATA
  Daily runoff 2002-2005: Shuangpai (Xiangjiang) + Fenghuang (Yuanjiang), Dongting Lake / Yangtze.
  Extensions: St. Joe River USA (+7 covariates: dayl, prcp, srad, swe, tmax, tmin, vp);
              Jemez River NM USA 1980-2010 (semi-arid generalization).
  Chronological split train : val : test = 7 : 1.5 : 1.5
        |
STAGE 1  MRS
  CEEMDAN (0.2 std, 100 noise) -> IMF1..9
  SampleEntropy(m=2, r=0.2 std) -> group similar SE -> RIMFs
  route: SE >= SE_orig => HIGH-FREQ set ; SE ~0-0.1 => LOW-FREQ set
        |                                   |
STAGE 2a HIGH-FREQ CHANNEL             STAGE 2b LOW-FREQ CHANNEL
  window in=7, out=h in {1,3,5,7}        window in=7, out=h
  normalize                              normalize
  INFORMER per RIMF_High:                LSTM per RIMF_Low:
    input = FS + PE + SE                   forget/input/output gates
    ProbSparse self-attention              hidden 256, 2 layers
    distilling: MaxPool(ELU(Conv1d))
    2 enc / 1 dec, 8 heads, d_ff 1024, GELU
    generative decoder -> all h at once
        \__________  training loop (identical both channels)  __________/
             loss = MSE ; Adam ; lr 1e-4 ; epochs 100 ; batch 32
             validation set -> hyperparameter tuning / early monitoring
        |
STAGE 3  ENSEMBLE
  yhat = sum(yhat_Informer(RIMF_High)) + yhat_LSTM(RIMF_Low) ; inverse-scale
        |
STAGE 4  EVALUATION (test set, 10 independent runs, averaged)
  accuracy: RMSE, NSE, KGE (+ R^2 scatter)
  extremes: EP on monthly block maxima
  significance: Wilcoxon signed-rank (p<0.05, marked *)
  cost: #subseries, runtime, 2.38 M params, 0.0172 GFLOPs, 0.60-0.95 s/epoch, 2.5-3.0 ms inference
  interpretability: SHAP (global), LIME (local peak vs stable)
  ablations: MRS vs CEEMDAN-only ; channel swap (MRS-LSTM-Informer)
  sensitivity: Sobol S1/ST on noise_std, embed_dim, tolerance
  stress: noise 5/10 %, missing 5/10 % (linear interpolation)
  baselines (all with MRS for fairness): Informer, Transformer, LSTM, RNN, TimesNet, PatchTST
```

### Metrics
| Metric | Definition | Ideal | Role |
|---|---|---|---|
| RMSE (15) | sqrt(mean((yhat-y)^2)) | 0 | absolute error in m³/s; peak-dominated; not cross-station comparable |
| NSE (16) | 1 - SSE/SS_total(mean benchmark) | 1 | standard hydrologic skill; <=0 means worse than predicting the mean; lenient on bias |
| KGE (17) | 1 - sqrt((a-1)^2+(b-1)^2+(R-1)^2), a=variability ratio, b=mean-bias ratio, R=Pearson | 1 | decomposes into correlation + variability + bias; headline metric for all % gains |
| R^2 | scatter goodness-of-fit (Figs 11-12) | 1 | diagnostic only |
| EP (19) | abs(q_pred-q_obs)/q_obs *100 on block maxima (1-month blocks) | 0 | flood-peak accuracy, operationally relevant |
| Wilcoxon signed-rank | paired non-parametric over 10 runs | p<0.05 | improvements not seed luck |
| params / FLOPs / latency / s-per-epoch | 2.38 M, 0.0172 G, 2.5-3 ms, 0.6-0.95 s | low | real-time & edge deployability |
| Sobol S1 / ST | variance-based sensitivity | - | which MRS hyperparams matter |

Note: Eq. 17 prints a = sigma(q/qhat), b = mu(q/qhat); read as the standard KGE ratio-of-moments definition.

### Headline numbers
- Shuangpai 1-step: DMEL RMSE 126.97, NSE 0.9307, KGE 0.9706 (Informer 198.51 / 0.8096 / 0.7917).
- Shuangpai 7-step: DMEL 183.80 / 0.8255 / 0.8759 ; KGE gains +37.5 % (Informer) up to +211.8 % (TimesNet).
- Fenghuang 7-step: DMEL 2.3004 / 0.8517 / 0.8784.
- Jemez (semi-arid) 7-step: DMEL NSE 0.9085 vs Informer 0.8532.
- St. Joe (multivariate) 1-step: DMEL RMSE -7.69 % vs Informer.

---

## 5. Stated limitations (§6.2)
1. MRS mitigates but does not remove non-stationarity → extreme peaks/troughs still degrade.
2. Purely data-driven, **no hydrological physics**; future: couple with SWAT.
3. Error accumulation beyond 7 days (monthly horizon) untested; future: recursive correction.

## 6. Extra critique for discussion
- Univariate input (runoff only) in main experiments → no causal forcing (rain, soil moisture, snowmelt).
- Only 4 years of daily data (~1460 samples) at the two main stations; test set ≈ 220 days → small-sample risk despite 10 repeats.
- Additive recomposition ignores nonlinear interaction between modes.
- Possible leakage from decomposing before splitting; a truly operational version must decompose causally (expanding window).
