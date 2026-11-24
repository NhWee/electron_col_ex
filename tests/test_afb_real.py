"""Real-data A_FB: transcription, covariance, and the radiated A_FB model."""

import numpy as np
import pytest
from scipy.integrate import quad

import afb_real_data as afb
import compute_cross_section as ccs


@pytest.fixture(scope="module")
def data():
    return afb.load_afb()


def test_tables_22_23_have_29_points_each(data):
    assert (data.channel == "mumu").sum() == 29
    assert (data.channel == "tautau").sum() == 29
    assert (data["afb_stat_err"] > 0).all()


def test_transcription_spot_checks(data):
    # values read off hep-ex/0012018 pages 77-78
    mu = data[(data.channel == "mumu") & (data.year == 1994) & (data["sample"] == "peak(ab)")].iloc[0]
    assert (mu["sqrt_s_mean_GeV"], mu["N"], mu["afb_measured"], mu["afb_stat_err"]) == (91.2198, 70493, 0.0040, 0.0035)
    tau = data[(data.channel == "tautau") & (data.year == 1995) & (data["sample"] == "peak+2")].iloc[0]
    assert (tau["afb_corrected"], tau["afb_fit_opal"]) == (0.0939, 0.1058)


def test_covariance_is_positive_definite(data):
    cov = afb.afb_covariance(data)
    np.testing.assert_allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0
    # systematics are below 10% of the statistical errors (paper, section 9.1)
    assert (np.sqrt(np.diag(cov)) < 1.1 * data["afb_stat_err"].to_numpy()).all()


def test_isr_average_matches_quad():
    rs, x_max = 91.2, 0.25
    beta = ccs.isr_beta(rs)
    f = lambda r: 1 / r**2  # noqa: E731
    expected, _ = quad(lambda x: ccs.qed_radiator(x, beta) * f(rs * np.sqrt(1 - x)), 0, x_max, limit=200)
    assert afb.isr_average(f, rs, x_max) == pytest.approx(expected, rel=1e-6)


def test_born_limit_and_acceptance_factor():
    rs = np.array([89.4, 91.2, 93.0])
    f1, f2 = afb.f1_f2(rs, 0.2316, 91.1876, 2.4952)
    np.testing.assert_allclose(afb.afb_model(rs, 0.2316, 91.1876, 2.4952, x_max=0), 0.375 * f2 / f1)
    k = 0.95
    np.testing.assert_allclose(afb.afb_model(rs, 0.2316, 91.1876, 2.4952, k=k, x_max=0),
                               0.375 * f2 / f1 * 4 * k / (3 + k**2))


def test_pole_asymmetry_value():
    # v = -0.0368, a = -0.5: A_l = 2va/(v^2+a^2) = 0.14641 -> A_FB^0 = 3/4 A_l^2 = 0.01608
    assert afb.pole_asymmetry(0.2316) == pytest.approx(0.01608, abs=1e-5)
    assert afb.pole_asymmetry(0.25) == 0.0


def test_fit_recovers_truth_on_asimov_data(data):
    asimov = data.copy()
    for ch, k in afb.ACCEPTANCE_K.items():
        sel = asimov.channel == ch
        asimov.loc[sel, "afb_corrected"] = afb.afb_model(asimov.loc[sel, "sqrt_s_mean_GeV"].to_numpy(), 0.2316,
                                                         afb.MZ_LINESHAPE, afb.GZ_LINESHAPE, k=k)
    m = afb.fit_sin2w(asimov)
    assert m.values["sin2w"] == pytest.approx(0.2316, abs=1e-5)
    assert m.fval < 1e-6
