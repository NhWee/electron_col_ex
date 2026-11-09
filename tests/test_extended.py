import numpy as np
import pytest

import compute_cross_section as ccs
import extended_fits as ext


def test_relativistic_bw_peak_value_and_running_width_identity():
    mZ, gZ = 91.1876, 2.4955
    assert ccs.breit_wigner_relativistic(mZ, 2.0, mZ, gZ) == pytest.approx(2.0)
    # README 0.2: running width = k * fixed width with rescaled (M, Gamma)
    k = 1 + gZ**2 / mZ**2
    s = np.linspace(88, 94, 30) ** 2
    d_run = (s - mZ**2) ** 2 + s**2 * gZ**2 / mZ**2
    m2, g2 = mZ / np.sqrt(k), gZ / np.sqrt(k)
    np.testing.assert_allclose(d_run, k * ((s - m2**2) ** 2 + m2**2 * g2**2), rtol=1e-12)


def test_improved_born_alpha_bookkeeping():
    """At the pole Re chi = 0, so sigma = Z term + photon term. The Z term is
    alpha-independent (G_F scheme), so switching alpha(0) -> alpha(M_Z^2)
    must change sigma by exactly the photon term's 4 pi/(3s) (a1^2 - a0^2)."""
    import electroweak as ew
    import simulate_mumu as sm
    mz, gz = sm.TRUE_MZ_GEV, sm.TRUE_GAMMAZ_GEV
    at_alpha0 = ew.born_cross_section_nb(mz, mz, gz, "mumu", alpha=ew.ALPHA_0)
    assert at_alpha0 == pytest.approx(sm.sigma_peak_from_iba(), rel=1e-9)
    at_alpha_mz = ext.iba_lineshape(mz, 1.0, mz, gz)
    photon = 4 * np.pi / (3 * mz**2) * (ew.ALPHA_MZ**2 - ew.ALPHA_0**2) * ew.GEV2_TO_NB
    assert at_alpha_mz - at_alpha0 == pytest.approx(photon, rel=1e-9)


def test_hadronic_couplings_give_r_lepton_before_qcd():
    """3 sum_q (v_q^2 + a_q^2) / (v_l^2 + a_l^2) over u, c, d, s, b is R_l
    before QCD corrections: 20.07 (x ~1.04 from QCD gives ~20.8, the PDG
    20.77 within the b-vertex and mass effects left out here). At the pole
    the photon terms pull the cross-section ratio down to 19.97."""
    import electroweak as ew
    mz, gz = 91.1876, 2.4955
    z_quarks = sum(sum(c**2 for c in ew.couplings(q, t3)) for q, t3 in ew.CHANNEL_FERMIONS["qq"])
    z_lepton = sum(c**2 for c in ew.couplings(-1.0, -0.5))
    assert 3 * z_quarks / z_lepton == pytest.approx(20.07, abs=0.01)
    ratio = ew.born_cross_section_nb(mz, mz, gz, "qq") / ew.born_cross_section_nb(mz, mz, gz, "mumu")
    assert 3 * ratio == pytest.approx(19.97, abs=0.01)


def test_fixed_to_running_width_shift():
    # README 0.2: M_run - M_fix ~ Gamma^2 / (2 M) = 34 MeV
    import electroweak as ew
    m_run, g_run = ew.fixed_to_running_width(91.1536, 2.4887)
    assert (m_run - 91.1536) * 1e3 == pytest.approx(2.4887**2 / (2 * 91.1536) * 1e3, rel=1e-3)
    assert (g_run - 2.4887) * 1e3 == pytest.approx(0.93, abs=0.05)


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
        smeared = ccs.with_beam_spread(ccs.breit_wigner_isr, asimov.loc[m, "sqrt_s_rms_GeV"].to_numpy())
        asimov.loc[m, "sigma_computed_nb"] = smeared(asimov.loc[m, "sqrt_s_GeV"].to_numpy(), peak, ccs.PDG_MZ_GEV, ccs.PDG_GAMMAZ_GEV)
    fit = ext.joint_fit(asimov)  # beam spread on by default, as in the Asimov truth
    assert fit.values["mZ"] == pytest.approx(ccs.PDG_MZ_GEV, abs=2e-4)
    assert fit.values["gammaZ"] == pytest.approx(ccs.PDG_GAMMAZ_GEV, abs=5e-4)
    for ch, peak in truth.items():
        assert fit.values[f"sigma_peak_{ch}"] == pytest.approx(peak, rel=1e-3)
    assert fit.fval < 1e-6
