"""GR6J (Pushpalatha et al. 2012): GR4J + exponential store for slow recessions, threshold exchange.

  x1 production store [mm]     x2 exchange coefficient [mm/d]   x3 routing store [mm]
  x4 UH time base [d]          x5 exchange threshold [-]        x6 exponential store coefficient [mm]
Follows airGR's frun_GR6J: UH1 gets 0.9*Pr, its output split 0.6 routing / 0.4 exponential store;
F = x2 * (R/x3 - x5) is added to the routing store, the exponential store and the direct branch.
"""
import numpy as np
from numba import njit

from shingle.models.gr4j import unit_hydrographs


@njit(cache=True)
def _core(P, E, x1, x2, x3, x5, x6, uh1, uh2, s, r, ex, q9, q1):
    n = P.shape[0]
    n1, n2 = uh1.shape[0], uh2.shape[0]
    q_out = np.empty(n)
    s_out = np.empty(n)
    r_out = np.empty(n)
    e_out = np.empty(n)
    aet = np.empty(n)
    exch = np.empty(n)
    for t in range(n):
        p, e = P[t], E[t]
        if p >= e:
            pn = p - e
            tw = np.tanh(pn / x1)
            ps = x1 * (1 - (s / x1) ** 2) * tw / (1 + s / x1 * tw)
            es = 0.0
        else:
            pn = 0.0
            tw = np.tanh((e - p) / x1)
            es = s * (2 - s / x1) * tw / (1 + (1 - s / x1) * tw)
            ps = 0.0
        s = s - es + ps
        perc = s * (1 - (1 + (4.0 / 9.0 * s / x1) ** 4) ** -0.25)
        s -= perc
        pr = perc + (pn - ps)
        for k in range(n1 - 1):
            q9[k] = q9[k + 1]
        q9[n1 - 1] = 0.0
        for k in range(n2 - 1):
            q1[k] = q1[k + 1]
        q1[n2 - 1] = 0.0
        for k in range(n1):
            q9[k] += 0.9 * pr * uh1[k]
        for k in range(n2):
            q1[k] += 0.1 * pr * uh2[k]
        q9n, q1n = q9[0], q1[0]
        q9[0] = 0.0
        q1[0] = 0.0

        f = x2 * (r / x3 - x5)
        r_new = r + 0.6 * q9n + f
        f_r = f if r_new >= 0 else -(r + 0.6 * q9n)
        r = max(0.0, r_new)
        qr = r * (1 - (1 + (r / x3) ** 4) ** -0.25)
        r -= qr

        ex = ex + 0.4 * q9n + f
        ar = min(33.0, max(-33.0, ex / x6))
        if ar > 7:
            qre = ex + x6 / np.exp(ar)
        elif ar < -7:
            qre = x6 * np.exp(ar)
        else:
            qre = x6 * np.log(np.exp(ar) + 1)
        ex -= qre

        qd_raw = q1n + f
        qd = max(0.0, qd_raw)
        f_d = f if qd_raw >= 0 else -q1n
        q_out[t] = qr + qre + qd
        s_out[t], r_out[t], e_out[t] = s, r, ex
        aet[t] = es + min(p, e)
        exch[t] = f_r + f + f_d
    return q_out, s_out, r_out, e_out, aet, exch, s, r, ex


def run(P, E, x, s_frac=0.5, r_frac=0.5):
    x1, x2, x3, x4, x5, x6 = [float(v) for v in x]
    uh1, uh2 = unit_hydrographs(x4)
    q9, q1 = np.zeros(len(uh1)), np.zeros(len(uh2))
    s0, r0, e0 = s_frac * x1, r_frac * x3, 0.0
    q, s, r, ex, aet, exch, s_end, r_end, e_end = _core(np.ascontiguousarray(P, float), np.ascontiguousarray(E, float),
                                                       x1, x2, x3, x5, x6, uh1, uh2, s0, r0, e0, q9, q1)
    return {"q": q, "s": s, "r": r, "exp": ex, "aet": aet, "exchange": exch,
            "storage0": s0 + r0 + e0, "storage1": s_end + r_end + e_end + q9.sum() + q1.sum()}


BOUNDS = [(10.0, 2500.0), (-20.0, 10.0), (5.0, 1000.0), (0.5, 10.0), (-4.0, 4.0), (0.1, 200.0)]
