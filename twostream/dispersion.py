"""Linear electrostatic dispersion relations for a two-species electron
two-stream system, in a cold-fluid and a warm-kinetic (Maxwellian) treatment.

Normalisation
-------------
    w_pe = 1             total electron plasma frequency (n_e = n_1 + n_2)
    v_ref = 1            reference thermal speed, so K = k * (v_ref / w_pe)
    a_j  = n_j / n_e     density fraction (sum_j a_j = 1)
    u_j                  drift speed of species j in units of v_ref
    th_j = v_th,j/v_ref  thermal speed of species j in units of v_ref
    z_j  = (w - K u_j) / (sqrt(2) K th_j)

Maxwellian convention: f ~ exp(-v^2 / (2 v_th^2)), v_th = sqrt(T/m).

    cold:          1 - sum_j a_j / (w - K u_j)^2                 = 0
    warm kinetic:  1 + sum_j a_j/(K th_j)^2 * [1 + z_j Z(z_j)]   = 0

Z is the plasma dispersion function (Fried & Conte). A species is a tuple
(a_j, u_j, th_j); th_j is ignored by the cold model.
"""

from __future__ import annotations

import numpy as np
from scipy.special import wofz

SQRT2 = np.sqrt(2.0)


def Z(zeta):
    """Plasma dispersion function Z(zeta) = i sqrt(pi) w(zeta).

    wofz is entire, so this is the analytic continuation for Im(zeta) < 0 too.
    """
    return 1j * np.sqrt(np.pi) * wofz(zeta)


def eps_kinetic(w, K, species):
    """Warm kinetic dielectric function."""
    eps = 1.0 + 0.0j
    for a, u, th in species:
        z = (w - K * u) / (SQRT2 * K * th)
        eps += a / (K * th) ** 2 * (1.0 + z * Z(z))
    return eps


def deps_kinetic(w, K, species):
    """Analytic d(eps_kinetic)/dw, using Z'(z) = -2 (1 + z Z(z))."""
    d = 0.0 + 0.0j
    for a, u, th in species:
        z = (w - K * u) / (SQRT2 * K * th)
        zp = -2.0 * (1.0 + z * Z(z))
        d += a / (K * th) ** 2 * (Z(z) + z * zp) / (SQRT2 * K * th)
    return d


def cold_poly(K, species):
    """Coefficients (highest power first) of the cold relation cleared of
    denominators: prod_j (w-Ku_j)^2 - sum_j a_j prod_{i!=j} (w-Ku_i)^2 = 0."""
    facs = [np.poly1d([1.0, -K * u]) ** 2 for _, u, _ in species]
    total = np.poly1d([1.0])
    for f in facs:
        total = total * f
    for j, (a, _, _) in enumerate(species):
        rest = np.poly1d([1.0])
        for i, f in enumerate(facs):
            if i != j:
                rest = rest * f
        total = total - a * rest
    return total.coefficients


def solve_cold(K, species):
    """Most unstable (largest Im w) root of the cold relation at one K."""
    roots = np.roots(cold_poly(K, species))
    return roots[np.argmax(roots.imag)]


def newton_kinetic(K, species, w0, tol=1e-12, itmax=100):
    """Damped complex Newton iteration on eps_kinetic(w) = 0.
    Returns (w, converged)."""
    w = complex(w0)
    for _ in range(itmax):
        f = eps_kinetic(w, K, species)
        if not np.isfinite(f):
            return w, False
        if abs(f) < tol:
            return w, True
        df = deps_kinetic(w, K, species)
        if df == 0 or not np.isfinite(df):
            return w, False
        step = f / df
        if abs(step) > 1.0:
            step /= abs(step)
        w -= step
    return w, abs(eps_kinetic(w, K, species)) < 1e-8


def _seeds(K, species, w_prev):
    out = [] if w_prev is None else [w_prev]
    out.extend(np.roots(cold_poly(K, species)))
    for wr in np.linspace(-3.0, 3.0, 13):
        for wi in (0.02, 0.1, 0.3):
            out.append(wr + 1j * wi)
    return out


def solve_kinetic(K, species, w_prev=None):
    """Most unstable kinetic root found at one K (nan if none converged)."""
    best = None
    for s in _seeds(K, species, w_prev):
        with np.errstate(all="ignore"):   # diverging seeds are simply rejected
            w, ok = newton_kinetic(K, species, s)
        if ok and np.isfinite(w) and (best is None or w.imag > best.imag):
            best = w
    return best if best is not None else complex(np.nan, np.nan)


def scan(species, K, model="kinetic"):
    """Return (omega_r(K), gamma(K)) for model 'cold' or 'kinetic'.

    Only gamma > 0 is meaningful. Where nothing is unstable the solver returns the root
    with the largest imaginary part it can find, and for the kinetic model that root can
    jump between damped branches from one K to the next.

    The kinetic root is continued in K outwards from the point of maximum cold
    growth, which keeps the solver on one branch.
    """
    K = np.asarray(K, dtype=float)
    wr = np.full(K.size, np.nan)
    gm = np.full(K.size, np.nan)

    if model == "cold":
        for i, k in enumerate(K):
            w = solve_cold(k, species)
            wr[i], gm[i] = w.real, w.imag
        return wr, gm
    if model != "kinetic":
        raise ValueError("model must be 'cold' or 'kinetic'")

    i0 = int(np.argmax([solve_cold(k, species).imag for k in K]))
    w = solve_kinetic(K[i0], species)
    wr[i0], gm[i0] = w.real, w.imag
    for order in (range(i0 + 1, K.size), range(i0 - 1, -1, -1)):
        w_prev = complex(wr[i0], gm[i0])
        for i in order:
            w = solve_kinetic(K[i], species, w_prev)
            wr[i], gm[i] = w.real, w.imag
            if np.isfinite(w):
                w_prev = w
    return wr, gm


def peak(K, gamma):
    """(gamma_max, K_at_max) of a scan, ignoring nan entries."""
    g = np.where(np.isfinite(gamma), gamma, -np.inf)
    i = int(np.argmax(g))
    return float(g[i]), float(np.asarray(K)[i])


def symmetric_two_stream(drift, vth):
    """Two equal counter-streaming beams, drifts +/-drift, thermal speed vth."""
    return [(0.5, +drift, vth), (0.5, -drift, vth)]
