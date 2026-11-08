import numpy as np

from twostream import eps_kinetic, scan, solve_cold, solve_kinetic, symmetric_two_stream

GAMMA_MAX = 1 / (2 * np.sqrt(2))   # analytic cold symmetric two-stream maximum
Q_AT_MAX = np.sqrt(3 / 8)          # K*u at the maximum


def test_cold_max_growth_rate_matches_analytic():
    # (w^2 - q^2)^2 = w^2 + q^2 with q = K u gives gamma_max = 1/(2 sqrt 2) at q^2 = 3/8
    u = 2.0
    w = solve_cold(Q_AT_MAX / u, symmetric_two_stream(u, 1.0))
    assert abs(w.imag - GAMMA_MAX) < 1e-9
    assert abs(w.real) < 1e-6


def test_cold_stability_boundary_at_Ku_equal_1():
    u = 2.0
    sp = symmetric_two_stream(u, 1.0)
    assert solve_cold(0.99 / u, sp).imag > 0
    assert solve_cold(1.01 / u, sp).imag < 1e-6


def test_kinetic_root_satisfies_dispersion_relation():
    sp = symmetric_two_stream(3.0, 1.0)
    w = solve_kinetic(0.3, sp)
    assert abs(eps_kinetic(w, 0.3, sp)) < 1e-8
    assert 0 < w.imag < GAMMA_MAX


def test_kinetic_approaches_cold_for_cold_beams():
    u = 20.0
    sp = symmetric_two_stream(u, 0.2)   # drift >> thermal speed
    K = np.linspace(0.8, 1.1, 31) * Q_AT_MAX / u
    gk = scan(sp, K, "kinetic")[1].max()
    assert abs(gk - GAMMA_MAX) / GAMMA_MAX < 0.02


def test_warm_marginal_wavenumber_matches_analytic_value():
    """At omega = 0 the symmetric relation gives K_c^2 = 2 z D(z) - 1, z = u/(sqrt(2) vth),
    with D the Dawson function. This is independent of the root finder."""
    from scipy.special import dawsn

    u, vth = 3.0, 1.0
    z = u / (np.sqrt(2) * vth)
    kc = np.sqrt((2 * z * dawsn(z) - 1) / vth**2)
    assert abs(kc - 0.4237) < 5e-4
    sp = symmetric_two_stream(u, vth)
    assert scan(sp, np.array([kc * 0.98]), "kinetic")[1][0] > 0
    assert scan(sp, np.array([kc * 1.02]), "kinetic")[1][0] < 0
    # and the dielectric function vanishes at omega = 0, K = K_c
    assert abs(eps_kinetic(0.0 + 0.0j, kc, sp)) < 1e-9


def test_three_species_cold_and_kinetic_roots_satisfy_their_relations():
    sp = [(0.3, 1.0, 0.1), (0.3, -1.0, 0.1), (0.4, 0.0, 0.1)]
    K = 0.4
    w = solve_cold(K, sp)
    residual = 1 - sum(a / (w - K * u) ** 2 for a, u, _ in sp)
    assert abs(residual) < 1e-8
    wk = solve_kinetic(K, sp)
    assert abs(eps_kinetic(wk, K, sp)) < 1e-8
