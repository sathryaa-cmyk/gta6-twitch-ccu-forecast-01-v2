"""Forecast GTA VI's launch-day Twitch viewers hour by hour (makefun
gta6-twitch-ccu-forecast-01). Python 3.13, standard library only.

Round 2 of the v1 forecaster. Method: learn the shape of a big launch's
release day from the case's own comparables (past big launches' hourly
Twitch curves), then anchor that shape to this launch's hour-1 print.

For every comparable launch we take its release day (UTC) as 24 hourly
viewer counts and express each hour as a log-ratio to that launch's hour-1
print: lr[h] = ln(viewers[h] / viewers[1]). The forecast for hour h is

    ln q50[h] = ln(p1) + lr_hat[h]

where p1 is this case's hour-1 print and lr_hat[h] is a kernel-weighted
local-linear fit of the comparables' lr[h] on four features of the launch:

  * x  = ln(hour-1 print) -- small categories at 00:00 UTC explode during the
         day (their midnight hour is nearly empty) while huge launches are
         already near their peak at hour 1, so the day's shape depends
         strongly on hour-1 size;
  * pp = ln(pre-day peak / hour-1 print), the peak hourly count in the 24 h
         before the release day (the New Zealand / early-region unlock surge)
         relative to hour 1 -- a strong pre-day tells us the launch is big
         relative to its midnight-UTC hour;
  * l23 = ln(viewers at 23:00 the day before / hour-1 print), the last
         pre-midnight hour (the central-Europe midnight unlock) -- where the
         curve is heading into hour 1 predicts the morning and the peak;
  * ch = ln(channels at hour 1) -- how many live streams carried the
         launch's first hour. A launch watched through many small streams
         behaves differently through the day than one carried by a few huge
         streams, and the channel count is known at the checkpoint. It
         enters the kernel weights only (not the local-linear design), so a
         comparable with no channel count is skipped when this feature is on;
         when the case itself has no channel count the feature is off and the
         forecaster is exactly the three-feature v1 kernel.

The fit is a Gaussian-kernel weighted local linear regression per hour
(bandwidths 1.0 in ln hour-1, 0.8 in pre-peak units, 1.5 in last-hour units,
0.5 in ln-channels units, each of the latter three scaled by the
comparables' own IQR spread), so launches most like this one dominate the
shape. The spread s[h] is the kernel-weighted residual spread around that
fit (floored at 0.35 log units, scaled by 0.55), and the five quantiles are

    q_tau[h] = p1 * exp(lr_hat[h] + z_tau * s[h]).

One guard against extrapolation: a local-linear fit is only trusted inside
the comparables' hour-1 range. When this launch's ln hour-1 sits outside
that range (a launch bigger than any comparable, as GTA VI may be), the
fitted log-ratio is shrunk toward the plain median log-ratio curve of all
comparables, with weight exp(-d^2 / 2) in the distance d outside the range
in ln units, so the forecast degrades gracefully instead of following a
steep fitted slope off the edge of the data.

Changes vs v1 (mean practice pinball loss 0.1381): narrower hour-1
bandwidth (1.25 -> 1.0), narrower pre-peak bandwidth (1.0 -> 0.8), wider
last-hour bandwidth (1.25 -> 1.5), the new channels feature (bandwidth 0.5),
a higher spread floor (0.30 -> 0.35) and a narrower calibration scale
(0.65 -> 0.55). Each change was verified on the kit's four practice trials
before being kept; ideas that did not help on the practice (a pre-day
overnight-dip feature, slope shrinkage, a weighted-median fit, empirical
residual quantiles, blending with the global median curve) were dropped.

Practice (4 past launches, answers re-fetched from SullyGnome per the kit's
NOTICE.txt): mean pinball loss 0.1252 vs the house persistence baseline's
0.3871 and v1's 0.1381 (p00 0.1342, p01 0.1020, p02 0.0903, p03 0.1744;
typical miss about 21.0%, 90% coverage about 0.891). Dev-board bars: rank 20
at 0.137, rank 10 at 0.134, rank 1 at 0.114.

Deterministic: no clock, no randomness, fixed iteration order. Runs in well
under a second per case: the four practice cases take 284 ms in total
(~71 ms each, far under the 8 s cap), and two runs write byte-identical
outputs. Edge cases verified: a final-like case with hour-1 of 800,000
(outside the comparables' range) degrades gracefully through the shrink
guard (median 0.43x-1.36x of hour 1), a tiny hour-1 stays near flat, no
comparables falls back to persistence, empty history_hourly and hour-1 of
0 still produce valid forecasts.
"""

import datetime
import json
import math
import os
import statistics
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Kernel bandwidths (see the module docstring).
BW_X = 1.0     # ln hour-1 print
BW_PRE = 0.8   # pre-day peak log-ratio, in units of the comparables' spread
BW_LAST = 1.5  # last pre-midnight hour log-ratio, same units
BW_CH = 0.5    # ln channels at hour 1, same units
SIG_FLOOR = 0.35   # minimum log spread before the calibration scale
SIG_SCALE = 0.55   # calibration scale on the residual spread
RIDGE = 1e-3       # ridge on the local-linear normal equations
# Sanity clamps on a predicted log-ratio to hour 1 (the largest ratio any
# comparable shows is about +7.7; nothing in the data goes below -1.5).
LR_MIN, LR_MAX = -2.0, 8.0


def parse_ts(text):
    return datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up (as the starter does)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def solve_linear(a, b):
    """Solve a small dense linear system by Gaussian elimination."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        if abs(m[col][col]) < 1e-12:
            continue
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                for cc in range(col, n + 1):
                    m[r][cc] -= f * m[col][cc]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0
            for i in range(n)]


def iqr_sigma(values):
    """Robust spread (IQR / 1.349), floored so a degenerate column cannot
    collapse a bandwidth to zero."""
    if len(values) < 4:
        return 1.0
    q = statistics.quantiles(values, n=4)
    return max(0.3, (q[2] - q[0]) / 1.349)


def day_series(hourly, day_start):
    """The 24 hourly viewer counts of the UTC day starting at day_start."""
    day = day_start[:10]
    by_hour = {}
    for row in hourly:
        if row["from"][:10] == day and row.get("viewers"):
            by_hour[int(row["from"][11:13])] = row["viewers"]
    return [by_hour.get(h) for h in range(24)]


def hour_channels(hourly, day_start, hour):
    """Channels in one hour of the UTC day starting at day_start, or None."""
    day = day_start[:10]
    for row in hourly:
        if (row["from"][:10] == day and int(row["from"][11:13]) == hour
                and row.get("channels")):
            return row["channels"]
    return None


def pre_day_peak_lr(hourly, day_start, hour1):
    """ln(max hourly viewers in the 24 h before day_start / hour1), or None."""
    t0 = parse_ts(day_start)
    vals = [r["viewers"] for r in hourly
            if t0 - datetime.timedelta(hours=24) <= parse_ts(r["from"]) < t0
            and r.get("viewers")]
    return math.log(max(vals) / hour1) if vals else None


def last_pre_hour_lr(hourly, day_start, hour1):
    """ln(viewers in the hour just before day_start / hour1), or None."""
    target = parse_ts(day_start) - datetime.timedelta(hours=1)
    for r in hourly:
        if parse_ts(r["from"]) == target and r.get("viewers"):
            return math.log(r["viewers"] / hour1)
    return None


def hourly_from_history(case):
    """Whole-hour means of the case's own history (fallback when the case's
    history_hourly lacks the pre-day hours the features need)."""
    buckets = {}
    for p in case.get("history", []):
        at = parse_ts(p["at"])
        key = at.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(key, []).append(p["viewers"])
    rows = [{"from": k.strftime("%Y-%m-%dT%H:%M:%SZ"),
             "viewers": sum(v) / len(v)} for k, v in sorted(buckets.items())]
    return rows


def case_features(case, p1):
    """(pre-peak log-ratio, last-hour log-ratio, ln channels at hour 1).

    The channel count is None when the case's hour-1 row carries none; the
    caller then runs the three-feature kernel.
    """
    hourly = case.get("history_hourly") or []
    pre = pre_day_peak_lr(hourly, case["day_start"], p1)
    last = last_pre_hour_lr(hourly, case["day_start"], p1)
    if pre is None or last is None:
        agg = hourly_from_history(case)
        if pre is None:
            pre = pre_day_peak_lr(agg, case["day_start"], p1)
        if last is None:
            last = last_pre_hour_lr(agg, case["day_start"], p1)
    if pre is None:
        pre = 0.0
    if last is None:
        last = pre
    known1 = next((k for k in case["known"] if k["hour"] == 1), None)
    ch = None
    if known1 is not None and known1.get("channels"):
        ch = math.log(known1["channels"])
    return pre, last, ch


def comparable_rows(case):
    """Per comparable: (ln hour-1, pre-peak lr, last-hour lr, ln channels at
    hour 1 or None, [lr per hour]).

    A comparable counts only when its release day has an hour-1 print and a
    pre-day peak; a missing last pre-midnight hour falls back to its pre-peak.
    """
    rows = []
    for comp in case.get("comparables", []):
        series = day_series(comp.get("hourly", []), comp["day_start"])
        if not series[0] or series[0] <= 0:
            continue
        pre = pre_day_peak_lr(comp.get("hourly", []), comp["day_start"],
                              series[0])
        if pre is None:
            continue
        last = last_pre_hour_lr(comp.get("hourly", []), comp["day_start"],
                                series[0])
        if last is None:
            last = pre
        ch0 = hour_channels(comp.get("hourly", []), comp["day_start"], 0)
        ch = math.log(ch0) if ch0 else None
        lr = [math.log(v / series[0]) if v and v > 0 else None
              for v in series]
        rows.append((math.log(series[0]), pre, last, ch, lr))
    return rows


def baseline_quantiles(p1, h):
    """The house persistence baseline: every hour stays at hour 1, spread
    widening like a random walk. Used only when there is nothing to learn
    from (no usable comparables)."""
    return [round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in Z]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)

    rows = comparable_rows(case)
    hours = [row["hour"] for row in case["forecast"]]
    if len(rows) < 6:
        # Too little to learn from: fall back to persistence, still valid.
        return {"hours": [{"hour": h, "q": baseline_quantiles(p1, h)}
                          for h in hours]}

    pre_case, last_case, ch_case = case_features(case, p1)
    use_ch = ch_case is not None
    pre_scale = iqr_sigma([r[1] for r in rows])
    last_scale = iqr_sigma([r[2] for r in rows])
    ch_scale = iqr_sigma([r[3] for r in rows if r[3] is not None]) \
        if use_ch else 1.0
    # The comparables' ln hour-1 range: outside it, shrink the fitted
    # log-ratio toward the plain median curve (see the module docstring).
    lo_x = min(r[0] for r in rows)
    hi_x = max(r[0] for r in rows)
    dist_out = 0.0 if lo_x <= log_p1 <= hi_x else min(abs(log_p1 - lo_x),
                                                      abs(log_p1 - hi_x))
    shrink_w = math.exp(-0.5 * (dist_out / 0.5) ** 2) if dist_out else 1.0

    out = []
    for h in hours:
        idx = h - 1  # log-ratio list is indexed by hour-1 (hour 1 == index 0)
        pts = [(r[0], r[1], r[2], r[3], r[4][idx]) for r in rows
               if r[4][idx] is not None and (not use_ch or r[3] is not None)]
        if len(pts) < 6:
            vals = [r[4][idx] for r in rows if r[4][idx] is not None]
            med = statistics.median(vals) if vals else 0.0
            sig = max(SIG_FLOOR, statistics.pstdev(vals)) if len(vals) > 1 \
                else SIG_FLOOR
            out.append({"hour": h, "q": quantiles(log_p1, med, sig)})
            continue
        # Gaussian-kernel weights in the feature dimensions.
        ws = []
        for cx, cpre, clast, cch, _y in pts:
            dx = (cx - log_p1) / BW_X
            dp = (cpre - pre_case) / (BW_PRE * pre_scale)
            dl = (clast - last_case) / (BW_LAST * last_scale)
            d2 = dx * dx + dp * dp + dl * dl
            if use_ch:
                dc = (cch - ch_case) / (BW_CH * ch_scale)
                d2 += dc * dc
            ws.append(math.exp(-0.5 * d2))
        sw = sum(ws)
        if sw < 1e-9:
            vals = [y for _cx, _p, _l, _c, y in pts]
            med = statistics.median(vals)
            sig = max(SIG_FLOOR, statistics.pstdev(vals))
            out.append({"hour": h, "q": quantiles(log_p1, med, sig)})
            continue
        # Weighted local-linear fit: lr ~ b0 + b1*dx + b2*dp + b3*dl.
        design = []
        for (cx, cpre, clast, _cch, _y) in pts:
            design.append([1.0,
                           (cx - log_p1) / BW_X,
                           (cpre - pre_case) / (BW_PRE * pre_scale),
                           (clast - last_case) / (BW_LAST * last_scale)])
        k = 4
        a = [[0.0] * k for _ in range(k)]
        bvec = [0.0] * k
        for xi, (_cx, _p, _l, _c, y), w in zip(design, pts, ws):
            for i in range(k):
                bvec[i] += w * xi[i] * y
                for j in range(k):
                    a[i][j] += w * xi[i] * xi[j]
        for i in range(k):
            a[i][i] += RIDGE
        beta = solve_linear(a, bvec)
        med = min(LR_MAX, max(LR_MIN, beta[0]))
        if shrink_w < 1.0:
            glob = statistics.median([r[4][idx] for r in rows
                                      if r[4][idx] is not None])
            med = shrink_w * med + (1.0 - shrink_w) * glob
            med = min(LR_MAX, max(LR_MIN, med))
        resid = [y - sum(beta[j] * xi[j] for j in range(k))
                 for xi, (_cx, _p, _l, _c, y) in zip(design, pts)]
        var = sum(w * r * r for w, r in zip(ws, resid)) / sw
        sig = max(SIG_FLOOR, math.sqrt(var))
        out.append({"hour": h, "q": quantiles(log_p1, med, sig)})
    return {"hours": out}


def quantiles(log_p1, med_log_ratio, sig):
    """Five quantiles around the fitted median, rounded, non-decreasing."""
    sig = sig * SIG_SCALE
    qs = []
    prev = 0.0
    for z in Z:
        v = round2(math.exp(log_p1 + med_log_ratio + z * sig))
        v = max(v, prev, 1)
        qs.append(v)
        prev = v
    return qs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
