#!/usr/bin/env python3
"""
dyadic_filter_bank_demo.py  --  pure stdlib, no numpy/scipy needed.

Demonstrates, on the console:
  1. What a FILTER BANK is (uniform vs dyadic/octave band layout)
  2. The iterated two-channel splitting tree that generates dyadic bands
  3. That EMD *empirically* behaves as a dyadic filter bank, by running a
     from-scratch EMD on white noise and measuring the mean period of each IMF
  4. Why the number of IMFs is ~log2(N), and what that means for DMEL

Run:  python3 dyadic_filter_bank_demo.py
"""

import math
import random

W = 78                                    # console width for plots


# ───────────────────────────────────────────────────── utility: ascii plotting
def plot(y, height=5, width=W, label="", mark_zero=True):
    """Render a 1-D sequence as an ASCII waveform (min/max per column)."""
    n = len(y)
    if n == 0:
        return
    cols = []
    for c in range(width):
        a = int(c * n / width)
        b = max(a + 1, int((c + 1) * n / width))
        seg = y[a:b]
        cols.append((min(seg), max(seg)))

    lo = min(c[0] for c in cols)
    hi = max(c[1] for c in cols)
    if hi - lo < 1e-12:
        hi, lo = lo + 1e-12, lo - 1e-12

    rows = []
    for r in range(height):
        # row r covers value band [v_lo, v_hi)
        v_hi = hi - (hi - lo) * r / height
        v_lo = hi - (hi - lo) * (r + 1) / height
        line = []
        for (cmin, cmax) in cols:
            if cmax >= v_lo and cmin <= v_hi:      # the column's span hits this row
                line.append("█")
            elif mark_zero and v_lo <= 0.0 <= v_hi:
                line.append("·")
            else:
                line.append(" ")
        rows.append("".join(line))

    if label:
        print(f"  {label}")
    for line in rows:
        print("  " + line)


def bar(frac, width, ch="█"):
    k = max(1, int(round(frac * width)))
    return ch * k


# ─────────────────────────────────────────── utility: natural cubic spline
def cubic_spline(xs, ys, xq):
    """Natural cubic spline through (xs,ys), evaluated at the list xq."""
    n = len(xs)
    if n < 2:
        return [ys[0] if ys else 0.0] * len(xq)
    if n == 2:                                     # fall back to a straight line
        x0, x1, y0, y1 = xs[0], xs[1], ys[0], ys[1]
        m = (y1 - y0) / (x1 - x0)
        return [y0 + m * (x - x0) for x in xq]

    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    # tridiagonal system for second derivatives (natural BC: c[0]=c[-1]=0)
    al = [0.0] * n
    for i in range(1, n - 1):
        al[i] = 3.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1])
    l = [1.0] + [0.0] * (n - 1)
    mu = [0.0] * n
    z = [0.0] * n
    for i in range(1, n - 1):
        l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (al[i] - h[i - 1] * z[i - 1]) / l[i]
    c = [0.0] * n
    b = [0.0] * (n - 1)
    d = [0.0] * (n - 1)
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    out = []
    seg = 0
    for x in xq:
        while seg < n - 2 and x > xs[seg + 1]:
            seg += 1
        while seg > 0 and x < xs[seg]:
            seg -= 1
        dx = x - xs[seg]
        out.append(ys[seg] + b[seg] * dx + c[seg] * dx * dx + d[seg] * dx ** 3)
    return out


# ───────────────────────────────────────────────────────── EMD from scratch
def extrema(y):
    """Indices of local maxima and minima."""
    mx, mn = [], []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            mx.append(i)
        elif y[i] < y[i - 1] and y[i] <= y[i + 1]:
            mn.append(i)
    return mx, mn


def _anchor(idx, y, n):
    """Mirror-extend an extrema index list to both boundaries (reduces end effects)."""
    if not idx:
        return [0, n - 1], [y[0], y[n - 1]]
    xs = list(idx)
    ys = [y[i] for i in xs]
    # left mirror
    if xs[0] != 0:
        xm = -xs[1] if len(xs) > 1 else -xs[0]
        xs = [min(xm, -1)] + xs
        ys = [ys[1] if len(ys) > 1 else ys[0]] + ys
    # right mirror
    if xs[-1] != n - 1:
        xm = 2 * (n - 1) - (xs[-2] if len(xs) > 1 else xs[-1])
        xs = xs + [max(xm, n)]
        ys = ys + [ys[-2] if len(ys) > 1 else ys[-1]]
    return xs, ys


def emd(signal, max_imf=12, n_sift=8):
    """Classic Huang sifting. Returns (list_of_imfs, residual)."""
    n = len(signal)
    grid = list(range(n))
    res = list(signal)
    imfs = []
    for _ in range(max_imf):
        mx, mn = extrema(res)
        if len(mx) + len(mn) < 3:            # monotone-ish -> it's the trend
            break
        h = list(res)
        for _s in range(n_sift):
            mx, mn = extrema(h)
            if len(mx) < 1 or len(mn) < 1:
                break
            xu, yu = _anchor(mx, h, n)
            xl, yl = _anchor(mn, h, n)
            up = cubic_spline(xu, yu, grid)
            lo = cubic_spline(xl, yl, grid)
            h = [h[i] - 0.5 * (up[i] + lo[i]) for i in range(n)]
        imfs.append(h)
        res = [res[i] - h[i] for i in range(n)]
    return imfs, res


def zero_crossings(y):
    return sum(1 for i in range(1, len(y)) if (y[i - 1] <= 0 < y[i]) or (y[i - 1] >= 0 > y[i]))


def mean_period(y):
    """Mean period in samples = 2*N / (number of zero crossings)."""
    z = zero_crossings(y)
    return (2.0 * len(y) / z) if z else float("inf")


def variance(y):
    m = sum(y) / len(y)
    return sum((v - m) ** 2 for v in y) / len(y)


# ═══════════════════════════════════════════════════════════════════ SECTION 1
def section1():
    print("=" * W)
    print("1.  WHAT IS A FILTER BANK?".center(W))
    print("=" * W)
    print("""
  A FILTER BANK is just a set of band-pass filters that together split one
  signal into several frequency bands, so that summing the bands gives the
  signal back.

      q(t) ──┬──► [ filter 1 ]──► band 1  (fast wiggles)
             ├──► [ filter 2 ]──► band 2
             ├──► [ filter 3 ]──► band 3
             └──► [ filter k ]──► band k  (slow drift)
                                  Σ = q(t)

  The only question is: WHERE DO YOU PUT THE BAND EDGES?
""")
    fs_label = "0                             frequency                          Nyquist"

    print("  (a) UNIFORM / LINEAR filter bank   — what the FFT and the STFT give you")
    print("      every band has the SAME width  (Δf constant)")
    print()
    k = 8
    seg = W - 8
    edges = "".join("|" + "-" * (seg // k - 1) for _ in range(k)) + "|"
    print("      " + edges)
    labels = "".join(f"{('b'+str(i+1)):^{seg//k}}" for i in range(k))
    print("      " + labels)
    print("      " + fs_label[:seg])
    print("""
      Problem: band 1 covers 0-6% of the spectrum, and so does band 8.
      But in real signals the INTERESTING slow structure is all crammed
      into band 1, while band 8 is mostly noise. Wasteful.
""")

    print("  (b) DYADIC / OCTAVE-BAND filter bank   — what wavelets and EMD give you")
    print("      each band is HALF the frequency of the previous one")
    print("      (dyad = two; 'dyadic' = successive halving, powers of 2)")
    print()
    # widths halve as we go down in frequency; draw from high freq (wide) to low (narrow)
    total = W - 10
    print("      high freq ◄────────────────────────────────────────► low freq")
    w = total
    rows = []
    for i in range(1, 8):
        w_i = max(1, total // (2 ** i))
        rows.append((i, w_i))
    # draw as nested bands, widest first
    for (i, w_i) in rows:
        band_lo = 1.0 / (2 ** i)          # normalized freq band edges
        band_hi = 1.0 / (2 ** (i - 1))
        print("      " + "█" * w_i + f"  band {i}:  f ∈ [{band_lo:.4f}, {band_hi:.4f}] × Nyquist")
    print("""
      Each band is one OCTAVE. Band i covers  [fs/2^(i+1), fs/2^i].
      In the TIME domain that means the characteristic PERIOD DOUBLES
      every band:   2, 4, 8, 16, 32, 64, 128 ... samples.
""")


# ═══════════════════════════════════════════════════════════════════ SECTION 2
def section2():
    print("=" * W)
    print("2.  WHY 'DYADIC'? THE ITERATED TWO-CHANNEL SPLIT".center(W))
    print("=" * W)
    print("""
  You get octave bands by repeatedly splitting only the LOW half in two.
  (This is Mallat's algorithme à trous / the wavelet cascade.)

     q ──►┬─[HP]──► detail 1          period ~2      ← finest
          │
          └─[LP]──►┬─[HP]──► detail 2  period ~4
                   │
                   └─[LP]──►┬─[HP]──► detail 3  period ~8
                            │
                            └─[LP]──►┬─[HP]──► detail 4  period ~16
                                     │
                                     └─[LP]──► ... ──► approximation (TREND)

  Every stage halves the remaining bandwidth, so:
      • band centre frequencies are  f/2, f/4, f/8, f/16 ...
      • band periods are             2,   4,   8,   16   ...
      • you can only do this log2(N) times before you run out of samples
        ⇒ THE NUMBER OF BANDS IS ~log2(N)

  Key property: CONSTANT-Q.
      Q = centre frequency / bandwidth = constant for every band.
      Absolute bandwidth is wide at high f, narrow at low f.

      ⇒ good TIME resolution for fast events (a flood spike)
        good FREQUENCY resolution for slow cycles (the annual cycle)
      This is the right trade-off for natural signals, and it is why
      music uses octaves, audio uses Mel/Bark scales, and 1/f noise
      looks self-similar on a log axis.
""")


# ═══════════════════════════════════════════════════════════════════ SECTION 3
def section3():
    print("=" * W)
    print("3.  EMD IS A DYADIC FILTER BANK — LIVE MEASUREMENT".center(W))
    print("=" * W)
    print("""
  Nobody designed EMD to be dyadic. It EMERGES from sifting.
  The classic test (Flandrin, Rilling & Gonçalves 2004; Wu & Huang 2004):
  run EMD on pure WHITE NOISE — which has no structure at all, equal power
  at every frequency — and look at what comes out.

  If EMD were arbitrary, the IMFs would be arbitrary. Let's see.
""")
    random.seed(7)
    N = 2048
    noise = [random.gauss(0, 1) for _ in range(N)]

    print(f"  Input: {N} samples of Gaussian white noise")
    plot(noise, height=4, label="white noise q(t)")
    print()
    print("  Running from-scratch EMD (cubic-spline sifting)... ", end="", flush=True)
    imfs, res = emd(noise, max_imf=11, n_sift=8)
    print(f"done → {len(imfs)} IMFs + residual")
    print()

    # ---- the measurement table
    print("  " + "-" * (W - 4))
    print(f"  {'IMF':<5}{'mean period':>13}{'ratio to':>11}{'#extrema':>10}{'variance':>11}{'var ratio':>11}")
    print(f"  {'':<5}{'(samples)':>13}{'previous':>11}{'':>10}{'':>11}{'':>11}")
    print("  " + "-" * (W - 4))
    prev_T = None
    prev_v = None
    ratios = []
    for i, c in enumerate(imfs, 1):
        T = mean_period(c)
        mx, mn = extrema(c)
        ne = len(mx) + len(mn)
        v = variance(c)
        rT = (T / prev_T) if prev_T else None
        rv = (prev_v / v) if (prev_v and v > 0) else None
        if rT:
            ratios.append(rT)
        print(f"  {i:<5}{T:>13.2f}{(f'{rT:.2f}x' if rT else '  —'):>11}"
              f"{ne:>10}{v:>11.4f}{(f'{rv:.2f}x' if rv else '  —'):>11}")
        prev_T, prev_v = T, v
    print("  " + "-" * (W - 4))
    if ratios:
        gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        print(f"  geometric mean of period ratios = {gm:.3f}      (dyadic ⇒ ≈ 2.0)")
    print()

    # ---- visual: period doubling on a log2 axis
    print("  MEAN PERIOD OF EACH IMF, plotted on a log2 axis:")
    print("  (if EMD is dyadic, these land at EQUAL SPACING)")
    print()
    Ts = [mean_period(c) for c in imfs]
    lo = math.log2(max(2.0, min(Ts)))
    hi = math.log2(max(Ts))
    span = max(1e-9, hi - lo)
    axis_w = W - 26
    for i, T in enumerate(Ts, 1):
        pos = int((math.log2(max(2.0, T)) - lo) / span * (axis_w - 1))
        line = [" "] * axis_w
        line[pos] = "◆"
        print(f"  IMF{i:<2} T={T:7.1f}  |{''.join(line)}|")
    print(f"  {'':>15}|{'log2(period) →':<{axis_w}}|")
    print("""
  Equal spacing on a log2 axis == each period is a constant MULTIPLE of the
  last == dyadic. (Deviation at the last IMFs is real: they contain only a
  couple of cycles, so their period estimate is very noisy.)
""")

    # ---- visual: waveforms, same time window
    print("  THE IMFs THEMSELVES (first 256 samples of each, same x-scale).")
    print("  Watch the oscillation density HALVE each time:")
    print()
    win = 256
    for i, c in enumerate(imfs[:7], 1):
        seg = c[:win]
        mx, mn = extrema(seg)
        plot(seg, height=3, label=f"IMF{i}   ({len(mx)+len(mn)} extrema in {win} samples, "
                                  f"mean period ≈ {mean_period(c):.0f})")
        print()
    plot(res, height=3, label="residual (monotone trend)")
    print()
    print("""  ⇒ EMD took a signal with NO structure and produced a clean
     octave-band decomposition. The dyadic behaviour is a property of the
     SIFTING ALGORITHM, not of the data.
""")
    return imfs, res


# ═══════════════════════════════════════════════════════════════════ SECTION 4
def section4():
    print("=" * W)
    print("4.  WHY SIFTING PRODUCES OCTAVES (the intuition)".center(W))
    print("=" * W)
    print("""
  Sifting builds envelopes by interpolating BETWEEN CONSECUTIVE EXTREMA.
  So IMF1 can only capture oscillations at the finest extrema spacing —
  the fastest thing present, period ≈ 2 samples.

  Now subtract IMF1. Every OTHER extremum belonged to IMF1, so removing it
  roughly HALVES the density of extrema in the residue:

     original   ▲ ▼ ▲ ▼ ▲ ▼ ▲ ▼ ▲ ▼ ▲ ▼ ▲ ▼ ▲ ▼      spacing 2
     - IMF1
     residue    ▲   ▼   ▲   ▼   ▲   ▼   ▲   ▼        spacing 4
     - IMF2
     residue    ▲       ▼       ▲       ▼            spacing 8
     - IMF3
     residue    ▲               ▼                    spacing 16
                          ...
     residue    ▲                                    monotone → STOP

  Extrema spacing doubles ⇒ period doubles ⇒ octave bands ⇒ DYADIC.
  And since you halve the extrema count each round, you run out after
  about log2(N) rounds. That is the whole mechanism.
""")


# ═══════════════════════════════════════════════════════════════════ SECTION 5
def section5():
    print("=" * W)
    print("5.  CONSEQUENCES FOR DMEL (daily runoff, N ≈ 1461 days)".center(W))
    print("=" * W)
    N = 1461
    print(f"""
  log2({N}) = {math.log2(N):.1f}   →  expect ~10 components.
  DMEL reports 9 IMFs + residual at BOTH stations.  ✓ consistent.

  Predicted period band of each IMF (≈ 2^i days) and what it physically is:
""")
    rows = [
        (1, "2–4 d", "measurement noise, single storm response", "0.1885", "HF → Informer"),
        (2, "4–8 d", "individual flood events", "0.2226", "HF → Informer"),
        (3, "8–16 d", "synoptic weather passages", "0.1911", "HF → Informer"),
        (4, "16–32 d", "multi-week wet/dry spells", "0.2593", "HF → Informer"),
        (5, "32–64 d", "intra-seasonal (MJO-like)", "0.2167", "HF → Informer"),
        (6, "64–128 d", "seasonal transition", "0.0509", "LF ┐"),
        (7, "128–256 d", "semi-annual / monsoon", "0.0305", "LF ├ summed"),
        (8, "256–512 d", "ANNUAL CYCLE", "0.0162", "LF ├ → RIMF6"),
        (9, "512–1024 d", "inter-annual", "0.0014", "LF ┘  → LSTM"),
    ]
    print(f"  {'IMF':<5}{'period':<12}{'physical meaning':<41}{'SE':<9}{'route'}")
    print("  " + "-" * (W - 4))
    for (i, p, m, se, r) in rows:
        print(f"  {i:<5}{p:<12}{m:<41}{se:<9}{r}")
    print("  " + "-" * (W - 4))
    print("""
  ── Three things this buys you for reading the paper ───────────────────────

  (1) DMEL's Table 2 sample entropies DECREASE MONOTONICALLY with IMF index.
      That is *exactly* what a dyadic filter bank predicts — longer period =
      smoother = lower complexity. So the "Modal Recognition Strategy" is
      largely REDISCOVERING the frequency ordering CEEMDAN already handed it
      for free. Splitting by IMF index would likely work as well; the paper
      never tests that baseline.

  (2) The HF/LF split lands almost exactly at the SEASONAL boundary
      (IMF5 | IMF6, i.e. ~64 days). So "high frequency" = weather, and
      "low frequency" = climate/seasonality + baseflow. That is a physically
      sensible cut — arguably the paper's most defensible design choice, even
      though it is justified only statistically (entropy) and never
      hydrologically.

  (3) ⚠️ The slowest IMFs are STATISTICALLY FRAGILE. IMF9 has a period of
      ~512–1024 days inside a 1461-day record: that is ~1.5–3 cycles. You
      cannot estimate a cycle from 2 cycles. With only 4 years of data the
      annual and inter-annual IMFs are nearly unidentifiable, and they are
      precisely the ones routed to the LSTM — which carries the LARGEST mean
      |SHAP| in the paper's own interpretability analysis (§5.4).
      So the dominant predictor is the least well-estimated component.
      A 20–30 year record would make this far more credible.
""")


if __name__ == "__main__":
    print()
    section1()
    print()
    section2()
    print()
    section3()
    print()
    section4()
    print()
    section5()
    print("=" * W)
    print("done.".center(W))
    print("=" * W)
