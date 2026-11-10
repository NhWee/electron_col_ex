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
