import numpy as np
import pandas as pd
import pytest

import compute_cross_section as ccs
import extended_fits as ext


def test_relativistic_bw_peak_value_and_running_width_identity():
    mZ, gZ = 91.1876, 2.4955
    assert ext.breit_wigner_relativistic(mZ, 2.0, mZ, gZ) == pytest.approx(2.0)
    # README 0.2: running width = k * fixed width with rescaled (M, Gamma)
    k = 1 + gZ**2 / mZ**2
    s = np.linspace(88, 94, 30) ** 2
    d_run = (s - mZ**2) ** 2 + s**2 * gZ**2 / mZ**2
    m2, g2 = mZ / np.sqrt(k), gZ / np.sqrt(k)
    np.testing.assert_allclose(d_run, k * ((s - m2**2) ** 2 + m2**2 * g2**2), rtol=1e-12)


def test_iba_lineshape_norm_one_matches_independent_peak_formula():
    import simulate_mumu as sm
    assert ext.iba_lineshape(sm.TRUE_MZ_GEV, 1.0, sm.TRUE_MZ_GEV, sm.TRUE_GAMMAZ_GEV) == pytest.approx(sm.sigma_peak_from_iba(), rel=1e-9)


def test_extended_selection_has_13_points():
    pts = ext.select_extended_points(ccs.load_counts())
    assert len(pts) == 13
    assert pts["borrowed_f"].sum() == 6


def test_joint_fit_recovers_truth_on_asimov_data():
    seven = ccs.select_corrected_points(ccs.load_counts())
    data = ext.joint_dataset(seven)
    truth = {"qq": 41.0, "mumu": 2.0, "tautau": 2.0}
    asimov = data.copy()
    for ch, peak in truth.items():
        m = asimov["channel"] == ch
        asimov.loc[m, "sigma_computed_nb"] = ccs.breit_wigner_isr(asimov.loc[m, "sqrt_s_GeV"].to_numpy(), peak, ccs.PDG_MZ_GEV, ccs.PDG_GAMMAZ_GEV)
    fit = ext.joint_fit(asimov)
    assert fit.values["mZ"] == pytest.approx(ccs.PDG_MZ_GEV, abs=2e-4)
    assert fit.values["gammaZ"] == pytest.approx(ccs.PDG_GAMMAZ_GEV, abs=5e-4)
    for ch, peak in truth.items():
        assert fit.values[f"sigma_peak_{ch}"] == pytest.approx(peak, rel=1e-3)
    assert fit.fval < 1e-6
