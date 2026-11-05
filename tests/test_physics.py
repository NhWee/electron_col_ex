"""Physics regression tests: formulas, normalizations and sign conventions
that have bitten this project before (or could). Everything here runs in
seconds and needs no MadGraph5."""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

import compute_cross_section as ccs
import simulate_mumu as sm


# --- ISR radiator ------------------------------------------------------------


def test_isr_beta_at_z_pole():
    # README 0.5: beta = 0.1077 at sqrt(s) = M_Z
    assert ccs.isr_beta(ccs.PDG_MZ_GEV) == pytest.approx(0.1077, abs=2e-4)


def test_full_radiator_integrates_to_one():
    """The O(alpha) leading-log radiator with the plus prescription must
    conserve cross section: int_0^1 H(x) dx = 1 exactly."""
    beta = ccs.isr_beta(ccs.PDG_MZ_GEV)
    # substitute u = x^beta to remove the x^(beta-1) endpoint singularity
    total, _ = quad(lambda x: ccs.qed_radiator_full(x, beta), 0, 1, limit=200)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_simplified_radiator_normalization_is_documented_value():
    beta = ccs.isr_beta(ccs.PDG_MZ_GEV)
    total, _ = quad(lambda x: ccs.qed_radiator_simplified(x, beta), 0, 1, limit=200)
    assert total == pytest.approx(0.951, abs=2e-3)


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
