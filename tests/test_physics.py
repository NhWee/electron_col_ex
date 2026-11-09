"""Physics regression tests: formulas, normalizations and sign conventions
that have bitten this project before (or could). Everything here runs in
seconds and needs no MadGraph5."""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import dblquad, quad

import compute_cross_section as ccs
import simulate_mumu as sm


# --- ISR radiator ------------------------------------------------------------


def electron_structure_function(z: float, eta: float) -> float:
    """Single-electron O(eta^2) leading-log structure function (Kuraev-Fadin
    form), z = energy fraction kept by one beam, eta = beta/2 per beam:

        D(z) = eta (1-z)^(eta-1) [1 + 3/4 eta + (9/32 - pi^2/12) eta^2]
               - eta/2 (1+z)
               + eta^2/8 [-4(1+z) ln(1-z) - (1+3z^2)/(1-z) ln z - 5 - z]

    Written independently of the two-beam radiators in compute_cross_section
    so that convolving two of them is a real cross-check."""
    soft = eta * (1 - z) ** (eta - 1) * (1 + 0.75 * eta + eta**2 * (9 / 32 - np.pi**2 / 12))
    hard = -0.5 * eta * (1 + z) + eta**2 / 8 * (-4 * (1 + z) * np.log1p(-z) - (1 + 3 * z**2) / (1 - z) * np.log(z) - 5 - z)
    return soft + hard


def two_beam_convolution(sqrt_s: float, born) -> float:
    """int dz1 dz2 D(z1) D(z2) born(sqrt(z1 z2 s)), with u_i = (1-z_i)^eta
    removing both endpoint singularities."""
    eta = ccs.isr_beta(sqrt_s) / 2

    def per_beam(u):
        u = min(max(u, 1e-300), 1 - 1e-15)
        z = min(max(1 - u ** (1 / eta), 1e-300), 1 - 1e-12)
        # D(z) dz/du, with the soft part's eta (1-z)^(eta-1) dz/du = 1 done analytically
        soft = 1 + 0.75 * eta + eta**2 * (9 / 32 - np.pi**2 / 12)
        hard = electron_structure_function(z, eta) - eta * (1 - z) ** (eta - 1) * soft
        return z, soft + hard * (1 / eta) * u ** (1 / eta - 1)

    def integrand(u2, u1):
        z1, w1 = per_beam(u1)
        z2, w2 = per_beam(u2)
        return w1 * w2 * born(sqrt_s * np.sqrt(z1 * z2))

    value, _ = dblquad(integrand, 0, 1, 0, 1, epsabs=0, epsrel=1e-7)
    return value


def one_radiator_convolution(sqrt_s: float, radiator, born, x_max: float = 1 - 1e-12) -> float:
    """Direct quadrature of int H(x) born(sqrt(s(1-x))) dx, independent of
    the substitution convolve_isr() uses."""
    beta = ccs.isr_beta(sqrt_s)
    value, _ = quad(lambda x: radiator(x, beta) * born(sqrt_s * np.sqrt(1 - x)), 0, x_max, limit=400)
    return value


def unit_bw(sqrt_s):
    return ccs.breit_wigner(sqrt_s, 1.0, ccs.PDG_MZ_GEV, ccs.PDG_GAMMAZ_GEV)


def test_isr_beta_at_z_pole():
    # README 0.5: beta = 0.1077 at sqrt(s) = M_Z
    assert ccs.isr_beta(ccs.PDG_MZ_GEV) == pytest.approx(0.1077, abs=2e-4)


@pytest.mark.parametrize("radiator, norm", [
    (ccs.qed_radiator_o2, 1.0),
    (ccs.qed_radiator_o1, 1.0),
    (ccs.qed_radiator_simplified, 0.951),
])
def test_radiator_normalization(radiator, norm):
    """The leading-log radiators conserve cross section, int_0^1 H dx = 1
    (plus prescription); the old simplified one integrates to 0.951."""
    beta = ccs.isr_beta(ccs.PDG_MZ_GEV)
    total, _ = quad(lambda x: radiator(x, beta), 0, 1, limit=400)
    assert total == pytest.approx(norm, abs=1e-6 if norm == 1.0 else 2e-3)


def test_single_electron_structure_function_normalization():
    eta = ccs.isr_beta(ccs.PDG_MZ_GEV) / 2
    total, _ = quad(lambda z: electron_structure_function(z, eta), 0, 1, limit=400)
    assert total == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("sqrt_s", [89.45, ccs.PDG_MZ_GEV, 93.0])
def test_o2_radiator_is_two_single_electron_structure_functions(sqrt_s):
    """The two-beam O(beta^2) radiator must equal the convolution of two
    single-electron structure functions up to O(beta^3) terms, ~1e-3. The
    O(beta) radiator misses the beta^2 terms and is off by ~4e-3."""
    exact = two_beam_convolution(sqrt_s, unit_bw)
    o2 = one_radiator_convolution(sqrt_s, ccs.qed_radiator_o2, unit_bw)
    assert o2 == pytest.approx(exact, rel=1.0e-3)
    if sqrt_s < 92:  # above the peak the O(beta) error changes sign and passes through 0
        o1 = one_radiator_convolution(sqrt_s, ccs.qed_radiator_o1, unit_bw)
        assert abs(o1 / exact - 1) > 2.5e-3


@pytest.mark.parametrize("name", list(ccs.RADIATOR_PARTS))
def test_convolution_substitution_matches_direct_quadrature(name):
    """convolve_isr() integrates in u = x^beta; check it against direct
    quadrature in x, with and without the acceptance cut."""
    radiator = {"o2": ccs.qed_radiator_o2, "o1": ccs.qed_radiator_o1, "simplified": ccs.qed_radiator_simplified}[name]
    with ccs._radiator(name):
        for sqrt_s in (89.45, 91.2, 93.0):
            for x_max in (ccs.ISR_X_MAX, 0.19):
                direct = one_radiator_convolution(sqrt_s, radiator, unit_bw, x_max=x_max)
                fast = ccs.convolve_isr(ccs.breit_wigner, sqrt_s, 1.0, ccs.PDG_MZ_GEV, ccs.PDG_GAMMAZ_GEV, x_max=x_max)[0]
                assert fast == pytest.approx(direct, rel=1e-7)


def test_acceptance_matches_paper_table_2():
    # m_ff^2/s > 0.01 (leptons) and s'/s > 0.01 (hadrons): x < 0.99
    assert ccs.ISR_X_MAX == pytest.approx(0.99)


def test_isr_shifts_observed_peak_upward():
    """Radiation only lowers the effective energy, so the observed peak
    sits above the true M_Z and the high-energy wing is enhanced."""
    grid = np.linspace(ccs.PDG_MZ_GEV - 2, ccs.PDG_MZ_GEV + 2, 161)
    born = ccs.breit_wigner(grid, 2.0, ccs.PDG_MZ_GEV, ccs.PDG_GAMMAZ_GEV)
    obs = ccs.breit_wigner_isr(grid, 2.0, ccs.PDG_MZ_GEV, ccs.PDG_GAMMAZ_GEV)
    assert grid[np.argmax(obs)] > grid[np.argmax(born)]
    assert obs[-1] / born[-1] > obs[0] / born[0]


# --- Propagator conventions (README 0.2) ------------------------------------


def test_running_width_is_rescaled_fixed_width():
    mZ, gZ = 91.1876, 2.4955
    k = 1 + gZ**2 / mZ**2
    s = np.linspace(85.0, 97.0, 50) ** 2
    d_run = (s - mZ**2) ** 2 + s**2 * gZ**2 / mZ**2
    d_fix = (s - (mZ / np.sqrt(k)) ** 2) ** 2 + (mZ / np.sqrt(k)) ** 2 * (gZ / np.sqrt(k)) ** 2
    np.testing.assert_allclose(d_run, k * d_fix, rtol=1e-12)


# --- gamma/Z interference and A_FB (README 0.3) ------------------------------


def test_afb_sign_flips_across_pole():
    below = sm.a_fb_true(sm.TRUE_MZ_GEV - 2.0)
    above = sm.a_fb_true(sm.TRUE_MZ_GEV + 2.0)
    assert below < 0 < above


def test_afb_symmetric_in_sin2thetaw_mirror():
    x = np.array([89.0, 91.19, 93.0])
    a = sm.a_fb_true(x, sin2thetaw=0.23)
    b = sm.a_fb_true(x, sin2thetaw=0.27)
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_gamma_ee_from_couplings_matches_pdg():
    # README 0.3: 83.4 MeV from couplings vs PDG ~84 MeV
    assert sm.gamma_ee_from_couplings() * 1000 == pytest.approx(83.4, abs=0.1)


def test_iba_peak_cross_section_matches_resonance_formula():
    # Two unrelated formulas should agree to ~1%: 1.9818 nb vs 1.9997 nb
    assert sm.sigma_peak_from_iba() == pytest.approx(sm.TRUE_SIGMA_PEAK_NB, rel=0.012)


def test_afb_fit_recovers_truth_on_asimov_data():
    """Noiseless A_FB points -> the fit must return the input sin2thetaw.
    This is the test that would have caught the 2026-09 sign error."""
    x = sm.SCAN_POINTS_GEV
    scan = pd.DataFrame(
        {
            "sqrt_s_GeV": x,
            "afb_measured": sm.a_fb_true(x),
            "afb_err": np.full(len(x), 0.01),
        }
    )
    fit = sm.fit_afb(scan, sm.TRUE_MZ_GEV, sm.TRUE_GAMMAZ_GEV)
    assert fit.values["sin2thetaw"] == pytest.approx(sm.SIN2_THETAW_EFF, abs=1e-5)
    assert fit.fval < 1e-8


def test_angular_pdf_normalized():
    for afb in (-0.3, 0.0, 0.3):
        total, _ = quad(lambda c: sm.angular_pdf(c, afb), -1, 1)
        assert total == pytest.approx(1.0, abs=1e-9)


# --- Lineshape fit on Asimov data -------------------------------------------


def test_breit_wigner_fit_recovers_truth_on_asimov_data():
    x = sm.SCAN_POINTS_GEV
    y = ccs.breit_wigner(x, 2.0, ccs.PDG_MZ_GEV, ccs.PDG_GAMMAZ_GEV)
    df = pd.DataFrame({"sqrt_s_GeV": x, "sigma_computed_nb": y, "sigma_computed_err_nb": 0.01})
    fit = ccs.fit_breit_wigner(df)
    assert fit.values["mZ"] == pytest.approx(ccs.PDG_MZ_GEV, abs=1e-4)
    assert fit.values["gammaZ"] == pytest.approx(ccs.PDG_GAMMAZ_GEV, abs=1e-4)


def test_isr_fit_recovers_truth_on_asimov_data():
    x = sm.SCAN_POINTS_GEV
    y = ccs.breit_wigner_isr(x, 2.0, ccs.PDG_MZ_GEV, ccs.PDG_GAMMAZ_GEV)
    df = pd.DataFrame({"sqrt_s_GeV": x, "sigma_computed_nb": y, "sigma_computed_err_nb": 0.01})
    fit = ccs.fit_breit_wigner_isr(df)
    assert fit.values["mZ"] == pytest.approx(ccs.PDG_MZ_GEV, abs=2e-4)
    assert fit.values["gammaZ"] == pytest.approx(ccs.PDG_GAMMAZ_GEV, abs=5e-4)


def test_afb_minos_errors_are_asymmetric_towards_the_flat_direction():
    """A_FB depends on sin2 only through v^2, flat at sin2 = 0.25, so the
    upper MINOS error (towards 0.25) must exceed the lower one."""
    x = sm.SCAN_POINTS_GEV
    scan = pd.DataFrame({"sqrt_s_GeV": x, "afb_measured": sm.a_fb_true(x), "afb_err": np.full(len(x), 0.03)})
    fit = sm.fit_afb(scan, sm.TRUE_MZ_GEV, sm.TRUE_GAMMAZ_GEV)
    lower, upper, from_minos = sm.afb_errors(fit)
    assert from_minos
    assert upper > lower > 0
    assert lower < fit.errors["sin2thetaw"] < upper


def test_asymmetric_pull_uses_the_error_facing_the_truth():
    assert sm.asymmetric_pull(1.2, 1.0, lower=0.1, upper=0.4) == pytest.approx(2.0)
    assert sm.asymmetric_pull(0.6, 1.0, lower=0.1, upper=0.4) == pytest.approx(-1.0)
