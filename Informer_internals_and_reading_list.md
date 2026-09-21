# Informer Internals — Deep Dive + Reading List
Companion to `DMEL_study_notes.md`.
Primary source: Zhou, Zhang, Peng, Zhang, Li, Xiong, Zhang, *"Informer: Beyond Efficient
Transformer for Long Sequence Time-Series Forecasting"*, AAAI 2021 (best paper), arXiv:2012.07436.

> **Erratum in the DMEL paper:** its Eq. 4 reads
> `ProbSparse(Q,K,V) = Softmax(QKᵀ/√d)V`, which is plain *canonical* attention.
> Nothing sparse about it. The real mechanism is below. (Its Eq. 5, the
> distilling formula, IS correct.)

---

## 0. The three bottlenecks Informer attacks

| Problem in vanilla Transformer | Cost | Informer's fix |
|---|---|---|
| Canonical self-attention is quadratic | O(L²) time & memory / layer | **ProbSparse self-attention** → O(L log L) |
| Stacking J encoder layers | O(J·L²) memory → OOM | **Self-attention distilling** → O((2−ε)L log L) |
| Autoregressive step-by-step decoding | L_y passes + **error accumulation** | **Generative-style decoder** → 1 pass |

---

## 1. ProbSparse self-attention

### 1.1 Empirical observation
Attention score distributions in trained models are **long-tailed**: a few query-key
dot products dominate; most queries yield a near-**uniform** row. A uniform row is useless:

    if p(k_j | q_i) ~ 1/L_K  then  Attn(q_i,K,V) = sum_j p(k_j|q_i) v_j ~ mean(V)

So such a "lazy" query contributes only the average of V — don't spend compute on it.

### 1.2 Query sparsity measurement (KL vs uniform)

    M(q_i, K) = ln SUM_j exp(q_i k_j^T / sqrt(d))          <- log-sum-exp
                - (1/L_K) SUM_j (q_i k_j^T / sqrt(d))      <- arithmetic mean

- large M  -> peaked row -> DOMINANT / active query (informative)
- small M  -> near uniform -> LAZY query (replaceable by mean(V))

### 1.3 Two approximations to make it cheap

(a) **max replaces log-sum-exp** (LSE is dominated by its max; also avoids overflow):

    M_bar(q_i, K) = max_j (q_i k_j^T / sqrt(d)) - (1/L_K) SUM_j (q_i k_j^T / sqrt(d))

(b) **random key sampling**: evaluate M_bar using only U = c·ln(L_Q) randomly sampled
keys instead of all L_K. Justified by the concentration bound in Lemma 1 / Prop. 1.

Then keep only the **top-u queries**, u = c·ln(L_Q), with `c` = *sampling factor* (default 5):

    A(Q,K,V) = Softmax( Q_bar K^T / sqrt(d) ) V

`Q_bar` = sparse matrix holding only those u queries. All other output positions are
filled with `mean(V)` (encoder) or `cumsum(V)` (masked decoder, preserves causality).

**=> O(L log L) time and memory per layer.**

### 1.4 Implementation sketch (mirrors models/attn.py: ProbAttention)

```python
def prob_sparse_attention(Q, K, V, c=5, mask=None):
    # Q:[B,H,Lq,d]  K,V:[B,H,Lk,d]
    B, H, Lq, d = Q.shape;  Lk = K.shape[2]

    U_part = min(Lk, int(c * ceil(log(Lq))))   # how many KEYS to sample
    u      = min(Lq, int(c * ceil(log(Lk))))   # how many QUERIES to keep

    # (1) score every query against a random subset of keys
    idx      = randint(0, Lk, (Lq, U_part))
    K_sample = K[:, :, idx, :]                              # [B,H,Lq,U_part,d]
    QK_s     = (Q.unsqueeze(-2) @ K_sample.transpose(-2,-1)).squeeze(-2)

    # (2) max-mean sparsity measure
    M   = QK_s.max(-1).values - QK_s.mean(-1)               # [B,H,Lq]
    top = M.topk(u, sorted=False).indices                   # active queries

    # (3) FULL attention, but only for those u queries
    scores = gather(Q, top) @ K.transpose(-2,-1) / sqrt(d)   # [B,H,u,Lk]
    if mask is not None: scores = scores.masked_fill(mask, -inf)
    ctx_top = softmax(scores, -1) @ V                        # [B,H,u,d]

    # (4) lazy queries fall back to mean(V)  (cumsum(V) if causal)
    out = V.mean(-2, keepdim=True).expand(B,H,Lq,d).clone()
    out = scatter(out, top, ctx_top)
    return out
```

---

## 2. Self-attention distilling  (= DMEL Eq. 5, correct)

Consecutive encoder layers carry redundant value maps. Between layers, privilege
dominant features and halve the time axis:

    X_{j+1} = MaxPool( ELU( Conv1d( [X_j]_AB ) ) )

- Conv1d: kernel width 3 along the TIME dimension (local feature filter)
- ELU activation
- MaxPool stride 2  ->  L becomes L/2 per distilling layer

Total encoder memory: **O((2-eps) L log L)**.

**Stacked replicas:** main stack takes length L, a second stack L/2, a third L/4,
each with progressively fewer layers; outputs concatenated -> robustness against the
resolution loss caused by distilling.

---

## 3. Generative-style decoder  (the part that actually matters for DMEL)

    X_de = Concat( X_token , X_0 )   in R^{(L_token + L_y) x d_model}

- `X_token` = **start token**: a real slice of the KNOWN input immediately before the
  forecast origin, length = `label_len`. Smarter than NLP's single `<s>` token — the
  decoder is seeded with actual recent dynamics.
- `X_0` = **zero placeholders**, length L_y = `pred_len`, values 0 but carrying their
  real **timestamps** (month / day / weekday embeddings).

Forward path:
1. **Masked ProbSparse self-attention** over X_de (mask = -inf, no peeking ahead)
2. **Full canonical multi-head cross-attention** to encoder memory
3. Single **fully-connected** projection to output dim

**One forward pass emits all L_y steps.** No prediction is ever fed back ->
no compounding error, O(1) inference passes. This is exactly what DMEL cites in
§4.2 ("generative decoder's ability to output all predictions at once, thereby
avoiding error accumulation").

---

## 4. Uniform input representation  (= DMEL Eq. 3, `Input = FS + PE + SE`)

    X_feed[i] = alpha * u_i + PE_(Lx(t-1)+i) + SUM_p [ SE_(Lx(t-1)+i) ]_p

- **FS / u_i** — value/feature scalar projected to d_model by a **1-D convolution**
  (kernel 3, stride 1), not a plain linear layer -> embeds local SHAPE, not a point.
- **PE** — fixed **sinusoidal** positional encoding (local ordering in the window).
- **SE** — learnable **global timestamp** embeddings, one per calendar attribute
  (minute, hour, weekday, day, month, holiday), summed. Injects global hierarchical
  seasonality. Crucial in hydrology: day-of-year ~ monsoon phase.
- alpha = 1 when inputs are normalized.

---

## 5. Hyperparameters to know

| Name | Meaning | Informer default | DMEL |
|---|---|---|---|
| `seq_len` (L_x) | encoder input length | 96-720 | **7** |
| `label_len` | decoder start-token length | 48 | not reported |
| `pred_len` (L_y) | horizon | 24-720 | 1/3/5/7 |
| `factor` (c) | ProbSparse sampling factor | 5 | not reported |
| `d_model` | width | 512 | 256 |
| `n_heads` | heads | 8 | 8 |
| `e_layers` / `d_layers` | enc / dec layers | 2 (or 3,2,1 stacks) / 1 | 2 / 1 |
| `d_ff` | FFN width | 2048 | 1024 |
| `attn` | `prob` or `full` | prob | — |
| `distil` | distilling on/off | True | — |
| `embed` | timeF / fixed / learned | timeF | — |
| activation | — | GELU | GELU |

---

## 6. !! Critique this unlocks for DMEL

With **seq_len = 7**:

    u      = c * ln(L_Q) = 5 * ln 7 ~ 5 * 1.95 ~ 10  -> min(7, 10) = 7
    U_part = c * ln(L_K) = 5 * ln 7 ~ 10             -> min(7, 10) = 7

**=> u = L_q and U = L_k: ProbSparse degenerates EXACTLY to full canonical attention.**
Zero sparsity, zero efficiency gain. Likewise, distilling (conv-3 + stride-2 pool) on a
length-7 sequence collapses it to ~3 timesteps; with e_layers=2 that is one distilling
step discarding over half of an already tiny temporal resolution.

So DMEL's contribution-2 claim — *"leverages the advantages of the Informer model in
ProbSparse self-attention mechanisms and distillation operations"* — **is not operative
at L = 7**. What genuinely does the work:

1. the **generative decoder** (direct multi-step, no recursive error)  <- real, important
2. the **timestamp embeddings** (seasonality prior)                    <- plausibly real
3. **cross-attention** between the 7-day encoding and the h placeholders <- real

Secondary: forecasting 7 days ahead from a 7-day receptive field with **no precipitation
input** is a very thin information base. Informer's whole premise is LONG input; DMEL
never exercises it.

**Omitted cheap experiment:** sweep `seq_len in {7, 30, 90, 365}` and compare
`attn='prob'` vs `attn='full'`. If identical at L=7 (they should be), that settles it.

**This also reframes the baselines.** TimesNet and PatchTST collapsing to NSE ~0.33 at
7-step is suspicious — both are strong LTSF models, but both NEED long lookbacks:
PatchTST wants seq_len >> patch_len (here patch_len=3, seq_len=7 -> ~3 patches), and
TimesNet's FFT-based period detection is meaningless on 7 samples. The comparison may be
decided by the shared 7-day window, not by architecture quality.

---

# READING LIST

## Tier 0 — prerequisites (skip if solid)
1. Vaswani et al., *Attention Is All You Need* — arXiv:**1706.03762**. §3 only.
2. **The Annotated Transformer** (Harvard NLP, 2022 rewrite) — nlp.seas.harvard.edu/annotated-transformer/
   Line-by-line PyTorch. Best way to internalize Q/K/V and masking.
3. **Karpathy, "Let's build GPT: from scratch, in code, spelled out"** (YouTube, ~2h).
   Builds attention from running-mean -> weighted-aggregation, i.e. exactly the mental
   model needed for "lazy query ~ mean(V)".

## Tier 1 — the core (do these)
4. **[*] Zhou et al., Informer** — arXiv:**2012.07436**.
   Order: Fig. 2 (long-tail motivation) -> §4.1 ProbSparse + Prop. 1 -> §4.2 distilling
   -> §4.3 generative decoder -> **Appendix A/B** (proofs + Algorithm 1 pseudocode).
   Most readers stop at §4 and miss the appendix, where the details live.
5. **[*] Official code: github.com/zhouhaoyi/Informer2020**
   Read exactly 3 files: `models/attn.py` (`ProbAttention._prob_QK`,
   `_get_initial_context`, `_update_context`), `models/encoder.py` (`ConvLayer` =
   distilling), `models/embed.py` (`DataEmbedding` = Eq. 3). Then run
   `main_informer.py` on ETTh1 with `--attn prob` vs `--attn full` and diff.
6. **github.com/thuml/Time-Series-Library** (THUML) — unified re-implementations of
   Informer, Autoformer, FEDformer, PatchTST, TimesNet, iTransformer, DLinear under one
   training loop. The correct way to run a FAIR version of DMEL's baseline comparison.
7. **Lilian Weng, "The Transformer Family v2.0"** (2023) —
   lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/
   Best taxonomy of sparse/efficient attention; situates ProbSparse among neighbours.

## Tier 2 — efficient-attention family (why ProbSparse and not something else)
8.  Child et al., **Sparse Transformer** — arXiv:1904.10509 (fixed/strided patterns)
9.  Kitaev et al., **Reformer** — arXiv:2001.04451 (LSH bucketing; closest in spirit)
10. Beltagy et al., **Longformer** — arXiv:2004.05150 (sliding window + global tokens)
11. Wang et al., **Linformer** — arXiv:2006.04768 (low-rank);
    Choromanski et al., **Performer** — arXiv:2009.14794 (kernel / linear attention)
12. Tay et al., **Efficient Transformers: A Survey** — arXiv:2009.06732

## Tier 3 — time-series transformers after Informer (to judge DMEL's baselines)
13. Wu et al., **Autoformer** — arXiv:2106.13008. Series decomposition INSIDE the
    architecture + Auto-Correlation. **This is the learned end-to-end alternative to
    MRS's external CEEMDAN** — highly relevant critique of DMEL.
14. Zhou et al., **FEDformer** — arXiv:2201.12740 (frequency-domain attention; again
    does internally what DMEL does manually in preprocessing)
15. **[*] Zeng et al., "Are Transformers Effective for Time Series Forecasting?"**
    AAAI 2023 — arXiv:**2205.13504**. A one-layer linear model (DLinear/NLinear) beats
    Informer/Autoformer/FEDformer on most LTSF benchmarks. Read ADVERSARIALLY against
    DMEL: DMEL has no linear baseline at all — a reviewer-grade objection.
16. Nie et al., **PatchTST** — arXiv:2211.14730 (a DMEL baseline; check required seq_len)
17. Wu et al., **TimesNet** — arXiv:2210.02186 (a DMEL baseline; FFT needs long inputs)
18. Liu et al., **iTransformer** — arXiv:2310.06625 (attend over VARIATES; natural next
    step for multivariate runoff)
19. Lim et al., **Temporal Fusion Transformer** — arXiv:1912.09363 (interpretable
    multi-horizon with static + known-future covariates; the right architecture if you
    want to add FORECAST rainfall)
20. Wen et al., **Transformers in Time Series: A Survey** — arXiv:2202.07125

## Tier 4 — hydrology (the domain, where DMEL is weakest)
21. **[*] Kratzert et al. (2018), "Rainfall-runoff modelling using LSTM networks"**,
    HESS 22:6005. Canonical DL-for-runoff paper. Note it uses METEOROLOGICAL FORCINGS,
    unlike DMEL.
22. Kratzert et al. (2019), **EA-LSTM** + CAMELS regional modelling, HESS 23:5089.
23. Gauch et al. (2021), **multi-timescale LSTM (MTS-LSTM)**, HESS. Learns daily AND
    hourly dynamics jointly — the principled ML answer to "separate fast from slow
    dynamics", vs DMEL's signal-decomposition answer.
24. **Nearing et al. (2024), "Global prediction of extreme floods in ungauged
    watersheds"**, Nature 627. State of the art, operational scale.
25. **github.com/neuralhydrology/neuralhydrology** — reference library + CAMELS loaders.
    Build on this to reproduce/extend DMEL rather than from scratch.

## Tier 5 — the specific tools DMEL borrows
26. Torres et al. (2011), **CEEMDAN**, ICASSP.
27. Richman & Moorman (2000), **Sample Entropy**, Am J Physiol 278:H2039.
28. **Gupta et al. (2009)**, "Decomposition of the MSE and NSE performance criteria",
    J. Hydrol. 377:80 — the **KGE** paper. Explains why KGE (alpha, beta, R) beats NSE
    and what KGE = 0.87 actually means.
29. Coles, *An Introduction to Statistical Modeling of Extreme Values*, Ch. 3 —
    **block-maxima** method used in DMEL §5.5.

---

## 4-session plan with exercises

| Session | Read | Do |
|---|---|---|
| 1 | #1, #2, #3 | Implement scaled dot-product + multi-head attention from scratch; verify against `nn.MultiheadAttention`. |
| 2 | #4 §4 + Appendix, #5 | Instrument `_prob_QK`: print `u`, `U_part`, selected query indices for seq_len = 7, 96, 720. **Confirm that at L=7 it selects ALL queries** -> reproduce the §6 critique yourself. |
| 3 | #15, #13, #17, #16 | In Time-Series-Library run Informer / DLinear / PatchTST on one runoff series at seq_len in {7,30,96,365}, pred_len=7. Does DMEL's baseline ranking survive a fair lookback? |
| 4 | #21, #28, #24 | Re-evaluate: add **DLinear** and **naive persistence** (yhat(t+h)=q(t)) baselines, plus a **precipitation-forced LSTM**. Report KGE with its alpha/beta/R decomposition, not just the scalar. |

The session 3-4 experiments are the ones that decide whether DMEL's 22-212 % improvement
claims are architectural or an artifact of a 7-day window plus missing baselines.
