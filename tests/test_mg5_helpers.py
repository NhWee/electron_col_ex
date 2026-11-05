"""The parts of simulate_mumu_mg5.py that do not need MadGraph5."""

import gzip

import numpy as np
import pytest

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
