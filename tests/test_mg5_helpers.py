"""The parts of simulate_mumu_mg5.py that do not need MadGraph5."""

import gzip

import numpy as np
import pandas as pd
import pytest

import compute_cross_section as ccs
import simulate_mumu as sm
import simulate_mumu_mg5 as mg5


def test_boost_identity_when_at_rest():
    p = np.array([45.0, 10.0, 20.0, -5.0])
    np.testing.assert_allclose(mg5.boost_to_rest_frame(p, np.array([91.0, 0, 0, 0])), p)


def test_boost_brings_total_to_rest():
    e1 = np.array([40.0, 0, 0, 40.0])
    e2 = np.array([30.0, 0, 0, -30.0])
    tot = e1 + e2
    b1 = mg5.boost_to_rest_frame(e1, tot)
    b2 = mg5.boost_to_rest_frame(e2, tot)
    np.testing.assert_allclose((b1 + b2)[1:], 0, atol=1e-9)
    # invariant mass preserved
    assert (b1 + b2)[0] == pytest.approx(np.sqrt(tot[0] ** 2 - tot[3] ** 2))


def _lhe(tmp_path, events):
    lines = ["<LesHouchesEvents>"]
    for ev in events:
        lines.append("<event>")
        lines.append(f"{len(ev)} 0 1.0 91.0 0.0078 0.118")
        for pdg, status, px, py, pz, e in ev:
            lines.append(f"{pdg} {status} 0 0 0 0 {px} {py} {pz} {e} 0.0 0.0 9.0")
        lines.append("</event>")
    lines.append("</LesHouchesEvents>")
    path = tmp_path / "ev.lhe.gz"
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    return path


def test_parse_costheta_no_isr(tmp_path):
    # e+ along +z, e- along -z, mu- exactly along the e- direction -> cos = +1
    ev = [(-11, -1, 0, 0, 45.6, 45.6), (11, -1, 0, 0, -45.6, 45.6), (13, 1, 0, 0, -45.6, 45.6), (-13, 1, 0, 0, 45.6, 45.6)]
    c = mg5.parse_lhe_costheta(_lhe(tmp_path, [ev]))
    np.testing.assert_allclose(c, [1.0])


def test_parse_costheta_isr_boosted_frame(tmp_path):
    """Asymmetric incoming energies (after ISR): a muon transverse in the
    e+e- rest frame is not transverse in the lab, but cos(theta) must come
    out as 0 once boosted back."""
    e_plus, e_minus = 40.0, 30.0
    tot = np.array([e_plus + e_minus, 0, 0, e_plus - e_minus])
    m = np.sqrt(tot[0] ** 2 - tot[3] ** 2)
    # muon transverse in the rest frame: (m/2, m/2, 0, 0); boost to lab
    gamma, bz = tot[0] / m, tot[3] / tot[0]
    mu_lab = np.array([gamma * m / 2, m / 2, 0.0, gamma * bz * m / 2])
    ev = [
        (-11, -1, 0, 0, e_plus, e_plus),
        (11, -1, 0, 0, -e_minus, e_minus),
        (13, 1, mu_lab[1], mu_lab[2], mu_lab[3], mu_lab[0]),
        (-13, 1, -mu_lab[1], -mu_lab[2], tot[3] - mu_lab[3], tot[0] - mu_lab[0]),
    ]
    c = mg5.parse_lhe_costheta(_lhe(tmp_path, [ev]))
    np.testing.assert_allclose(c, [0.0], atol=1e-9)


def test_read_param_card_truth(tmp_path):
    """Lines in the format of MG5's sm param_card.dat (MASS block with the
    dependent W mass, DECAY line for the Z)."""
    card = tmp_path / "param_card.dat"
    card.write_text(
        "BLOCK MASS # \n"
        "      23 9.118800e+01 # MZ \n"
        "      25 1.250000e+02 # MH \n"
        "  24 8.041900e+01 # w+ : cmath.sqrt(MZ__exp__2/2. + ...) \n"
        "DECAY  23 2.441404e+00 # WZ \n"
    )
    truth = mg5.read_param_card_truth(card)
    assert truth["mZ"] == 91.188 and truth["gammaZ"] == 2.441404 and truth["mW"] == 80.419
    assert truth["sin2w_onshell"] == pytest.approx(1 - (80.419 / 91.188) ** 2)


@pytest.mark.parametrize("isr_on", [False, True])
def test_analyse_scan_runs_on_a_toy_scan(tmp_path, isr_on):
    """The analysis half of the MG5 script (also what --refit runs), fed a
    toy scan in the MG5 scan-table format instead of MadGraph5 events."""
    rng = np.random.default_rng(7)
    rows = [sm.generate_scan_point(x, 0.93, 0.0004, rng) for x in sm.SCAN_POINTS_GEV]
    scan = pd.DataFrame(rows)
    scan["sigma_mg5_true_pb"] = 1000 * ccs.breit_wigner_isr(scan["sqrt_s_GeV"].to_numpy(), sm.TRUE_SIGMA_PEAK_NB, sm.TRUE_MZ_GEV, sm.TRUE_GAMMAZ_GEV)
    truth = {"mZ": sm.TRUE_MZ_GEV, "gammaZ": sm.TRUE_GAMMAZ_GEV, "sin2w_onshell": sm.SIN2_THETAW_EFF}
    out = mg5.analyse_scan(scan, truth, isr_on=isr_on, out_dir=tmp_path)
    assert all(np.isfinite(v) for v in out.values())
    # the toy has ISR, so the ISR fit is the one that recovers the truth
    assert abs(out["pull_mZ_isr"]) < 3 < out["pull_mZ_naive"]
    tag = "_isr" if isr_on else ""
    assert (tmp_path / f"simulated_mumu_mg5{tag}_closure.png").exists()
    assert (tmp_path / f"simulated_mumu_mg5{tag}_afb.png").exists()


def test_isr_fit_of_radiation_free_data_is_biased_low():
    """README section 3 without MadGraph5: on a lineshape generated with no
    ISR (noiseless, at the param_card truth and luminosity of that section)
    the naive fit is exact and the ISR-convolved fit over-corrects, by about
    -8 sigma on M_Z and -6 sigma on Gamma_Z."""
    mz, gz = 91.1880, 2.4414
    x = sm.SCAN_POINTS_GEV
    y = ccs.breit_wigner(x, 2.0, mz, gz)
    n_selected = mg5.LUMI_PBINV_MG5 * y * 1000 * 0.9595
    scan = pd.DataFrame({"sqrt_s_GeV": x, "sigma_computed_nb": y, "sigma_computed_err_nb": y / np.sqrt(n_selected)})
    naive = ccs.fit_breit_wigner(scan)
    assert naive.values["mZ"] == pytest.approx(mz, abs=1e-4)
    assert naive.values["gammaZ"] == pytest.approx(gz, abs=1e-4)
    isr = ccs.fit_breit_wigner_isr(scan)
    assert (isr.values["mZ"] - mz) / isr.errors["mZ"] == pytest.approx(-8.3, abs=0.2)
    assert (isr.values["gammaZ"] - gz) / isr.errors["gammaZ"] == pytest.approx(-5.6, abs=0.2)


@pytest.fixture
def restore_isr_settings():
    """configure_fit_for_mg5_isr() changes module-level fit settings."""
    saved = (ccs.ISR_ALPHA, ccs.ISR_RADIATOR, ccs.ISR_X_MAX)
    yield
    ccs.ISR_ALPHA, ccs.ISR_RADIATOR, ccs.ISR_X_MAX = saved


def test_fit_radiator_is_matched_to_mg5_electron_pdf(restore_isr_settings):
    """MG5's isronlyll PDF has alpha = 0.007562397 = 1/132.23 hard-coded
    (lep_densities/isronlyll/gridpdfaux.f) and no non-log O(alpha) term, so
    with --isr the fit uses the leading-log radiator with that alpha: beta
    is 3.6% larger than with alpha(0)."""
    beta_alpha0 = ccs.isr_beta(91.188)
    mg5.configure_fit_for_mg5_isr()
    assert ccs.ISR_RADIATOR == "o2ll"
    assert 1 / ccs.ISR_ALPHA == pytest.approx(132.23, abs=0.01)
    assert ccs.ISR_X_MAX == pytest.approx(1 - 0.9**2)
    assert ccs.isr_beta(91.188) / beta_alpha0 == pytest.approx(1.0363, abs=1e-4)
    assert ccs.isr_beta(91.188) == pytest.approx(0.1116, abs=1e-4)


def test_isr_on_closure_expectation(restore_isr_settings):
    """What `--isr` should show, on a noiseless copy of it: data generated
    with MG5's radiator settings are recovered exactly by the matched fit,
    the naive fit lands ~7 sigma high on M_Z, and a fit that kept alpha(0)
    would be ~9 MeV high."""
    mz, gz = 91.1880, 2.4414
    mg5.configure_fit_for_mg5_isr()
    x = sm.SCAN_POINTS_GEV
    y = ccs.breit_wigner_isr(x, 2.0, mz, gz)
    n_selected = mg5.LUMI_PBINV_MG5 * y * 1000 * 0.96
    scan = pd.DataFrame({"sqrt_s_GeV": x, "sigma_computed_nb": y, "sigma_computed_err_nb": y / np.sqrt(n_selected)})
    matched = ccs.fit_breit_wigner_isr(scan)
    assert matched.values["mZ"] == pytest.approx(mz, abs=2e-4)
    assert matched.values["gammaZ"] == pytest.approx(gz, abs=5e-4)
    naive = ccs.fit_breit_wigner(scan)
    assert (naive.values["mZ"] - mz) / naive.errors["mZ"] == pytest.approx(7.1, abs=0.3)
    ccs.ISR_ALPHA = ccs.ALPHA_QED
    mismatched = ccs.fit_breit_wigner_isr(scan)
    assert (mismatched.values["mZ"] - mz) * 1e3 == pytest.approx(8.7, abs=1.0)



def test_refit_of_the_first_complete_isr_scan(tmp_path, restore_isr_settings):
    """The first complete --isr scan (MG5 3.5.0, 2026-11-11), as printed by
    --refit: the matched ISR fit recovers the truth within ~1 sigma, the
    naive fit is ~8 sigma high, and on MG5's own noise-free cross sections
    the improved-Born fit lands within a few MeV where the Breit-Wigner is
    ~+36 MeV off (convention and photon term)."""
    rows = [
        (88.1876, 207.6, 0.2287, 0.0282, -0.3333, 0.1161), (88.6876, 274.2, 0.2772, 0.0310, -0.0750, 0.1115),
        (89.1876, 382.0, 0.3500, 0.0348, -0.0495, 0.0994), (89.6876, 550.2, 0.5475, 0.0436, -0.0506, 0.0795),
        (90.1876, 818.0, 0.7380, 0.0506, -0.0516, 0.0684), (90.6876, 1179.0, 1.2855, 0.0668, -0.0350, 0.0519),
        (91.1876, 1454.0, 1.3964, 0.0696, 0.0124, 0.0498), (91.6876, 1360.0, 1.5073, 0.0723, 0.0667, 0.0478),
        (92.1876, 1073.0, 1.0464, 0.0603, 0.0199, 0.0575), (92.6876, 817.8, 0.9044, 0.0560, 0.1801, 0.0609),
        (93.1876, 618.9, 0.4678, 0.0403, 0.0963, 0.0857), (93.6876, 485.4, 0.4851, 0.0410, 0.2429, 0.0820),
        (94.1876, 399.9, 0.4158, 0.0380, 0.2500, 0.0884),
    ]
    scan = pd.DataFrame(rows, columns=["sqrt_s_GeV", "sigma_mg5_true_pb", "sigma_computed_nb", "sigma_computed_err_nb",
                                       "afb_measured", "afb_err"])
    truth = {"mZ": 91.1880, "gammaZ": 2.4414, "sin2w_onshell": 1 - (80.419 / 91.188) ** 2}
    mg5.configure_fit_for_mg5_isr()
    out = mg5.analyse_scan(scan, truth, isr_on=True, out_dir=tmp_path)
    assert out["pull_mZ_isr"] == pytest.approx(0.84, abs=0.05)
    assert out["pull_gammaZ_isr"] == pytest.approx(-1.22, abs=0.05)
    assert out["pull_mZ_naive"] == pytest.approx(8.16, abs=0.05)
    assert out["pull_sin2w"] == pytest.approx(0.73, abs=0.05)
    assert abs(out["pull_mZ_iba"]) < 1.5
    assert out["shape_mZ_isr_MeV"] == pytest.approx(36, abs=3)
    assert abs(out["shape_mZ_iba_MeV"]) < 8



def test_refit_of_the_isr_off_scan():
    """The ISR-off scan of 2026-11-11: the naive fit is right, the ISR fit
    over-corrects by ~-8.6 sigma, and on the noise-free cross sections the
    improved-Born model is within a few MeV."""
    rows = [
        (88.1876, 284.5, 0.3010, 0.0323, -0.2184, 0.1046), (88.6876, 383.5, 0.3806, 0.0363, -0.4182, 0.0866),
        (89.1876, 539.6, 0.5121, 0.0421, -0.2432, 0.0797), (89.6876, 793.1, 0.7958, 0.0525, -0.1478, 0.0652),
        (90.1876, 1194.0, 1.3598, 0.0687, -0.0992, 0.0502), (90.6876, 1714.0, 1.8062, 0.0792, -0.0958, 0.0436),
        (91.1876, 2022.0, 2.0484, 0.0843, 0.0000, 0.0411), (91.6876, 1749.0, 1.6920, 0.0766, 0.1370, 0.0448),
        (92.1876, 1240.0, 1.2733, 0.0664, 0.1033, 0.0518), (92.6876, 834.0, 0.8477, 0.0542, 0.1592, 0.0631),
        (93.1876, 574.6, 0.5882, 0.0451, 0.2471, 0.0743), (93.6876, 412.0, 0.4290, 0.0385, 0.2097, 0.0878),
        (94.1876, 308.7, 0.3252, 0.0336, 0.3191, 0.0977),
    ]
    scan = pd.DataFrame(rows, columns=["sqrt_s_GeV", "sigma_mg5_true_pb", "sigma_computed_nb", "sigma_computed_err_nb",
                                       "afb_measured", "afb_err"])
    truth = {"mZ": 91.1880, "gammaZ": 2.4414, "sin2w_onshell": 1 - (80.419 / 91.188) ** 2}
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        out = mg5.analyse_scan(scan, truth, isr_on=False, out_dir=Path(tmp))
    assert out["pull_mZ_naive"] == pytest.approx(0.14, abs=0.05)
    assert out["pull_mZ_isr"] == pytest.approx(-8.58, abs=0.05)
    assert out["pull_gammaZ_isr"] == pytest.approx(-5.46, abs=0.05)
    assert abs(out["shape_mZ_iba_MeV"]) < 6
