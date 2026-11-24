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


def test_z_term_at_the_pole_is_the_partial_width_formula():
    """sigma0 = 12 pi Gamma_ee Gamma_ff / (M_Z^2 Gamma_Z^2), with Gamma_ee from
    the independent partial-width formula in simulate_mumu.py."""
    import electroweak as ew
    import simulate_mumu as sm
    mz, gz = sm.TRUE_MZ_GEV, sm.TRUE_GAMMAZ_GEV
    gamma_ll = sm.gamma_ee_from_couplings()
    expected = 12 * np.pi * gamma_ll**2 / (mz**2 * gz**2) * ew.GEV2_TO_NB
    assert ew.z_term_cross_section_nb(mz, mz, gz, "mumu") == pytest.approx(expected, rel=1e-12)
    assert ew.DELTA_QED == pytest.approx(1.00185, abs=1e-5)


def test_joint_pole_quantities_propagation():
    """sigma0_had and R_l from a stand-in fit with a known diagonal covariance."""
    from iminuit import Minuit
    import electroweak as ew
    centre = dict(norm_qq=3.1, norm_mumu=1.0, norm_tautau=1.02, mZ=91.15, gammaZ=2.49)
    sigma = dict(norm_qq=0.01, norm_mumu=0.005, norm_tautau=0.006, mZ=0.003, gammaZ=0.006)

    def chi2(norm_qq, norm_mumu, norm_tautau, mZ, gammaZ):
        values = dict(norm_qq=norm_qq, norm_mumu=norm_mumu, norm_tautau=norm_tautau, mZ=mZ, gammaZ=gammaZ)
        return sum(((values[k] - centre[k]) / sigma[k]) ** 2 for k in centre)

    chi2.errordef = Minuit.LEAST_SQUARES
    fit = Minuit(chi2, **{k: v * 1.001 for k, v in centre.items()})
    fit.migrad()
    fit.hesse()
    pole = ext.joint_pole_quantities(fit)
    z_qq = float(ew.z_term_cross_section_nb(91.15, 91.15, 2.49, "qq"))
    z_ll = float(ew.z_term_cross_section_nb(91.15, 91.15, 2.49, "mumu"))
    assert pole["sigma0_had"] == pytest.approx(3.1 * z_qq * ew.DELTA_QED, rel=1e-5)
    assert pole["r_lepton"] == pytest.approx(3.1 * z_qq / (1.01 * z_ll), rel=1e-5)
    # sigma0 ~ norm / Gamma^2: relative errors 0.01/3.1 and 2 x 0.006/2.49 in quadrature
    assert pole["sigma0_had_err"] / pole["sigma0_had"] == pytest.approx(np.hypot(0.01 / 3.1, 2 * 0.006 / 2.49), rel=0.02)
    # R_l: the common Gamma_Z and M_Z dependence cancels
    expected_r_err = np.sqrt((0.01 / 3.1) ** 2 + (0.005 / 2.02) ** 2 + (0.006 / 2.02) ** 2)
    assert pole["r_lepton_err"] / pole["r_lepton"] == pytest.approx(expected_r_err, rel=0.02)
