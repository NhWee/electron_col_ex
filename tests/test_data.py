"""Guards against transcription regressions in data/raw/ and against the
sigma = fN/L arithmetic drifting away from OPAL's published numbers."""

import pandas as pd
import pytest

import compute_cross_section as ccs


@pytest.fixture(scope="module")
def seven_points():
    return ccs.select_corrected_points(ccs.load_counts())


def test_seven_corrected_points(seven_points):
    assert len(seven_points) == 7
    assert set(seven_points["year"]) == {1993, 1994, 1995}


def test_table1_has_29_rows():
    assert len(ccs.load_counts()) == 29


@pytest.mark.parametrize("channel", list(ccs.CHANNELS))
def test_computed_matches_published_except_known_anomaly(seven_points, channel):
    comp = ccs.compute_channel(seven_points, channel)
    assert len(comp) == 7
    known_bad = (comp["year"] == 1993) & (comp["sample"] == "peak")
    clean = comp[~known_bad]
    assert (clean["diff_pct"].abs() < ccs.ANOMALY_THRESHOLD_PCT).all(), clean


def test_1993_peak_anomaly_is_track_channels_only(seven_points):
    pivot = ccs.channel_consistency_check(seven_points)
    row = pivot.loc[(1993, "peak")]
    assert abs(row["qq"]) < 1.0 and abs(row["ee"]) < 1.0
    assert row["mumu"] < -5.0 and row["tautau"] < -3.0


def test_mumu_errors_are_positive_and_small(seven_points):
    comp = ccs.compute_channel(seven_points, "mumu")
    rel = comp["sigma_computed_err_nb"] / comp["sigma_computed_nb"]
    assert (rel > 0).all() and (rel < 0.03).all()


def test_energies_are_the_channels_own_published_means(seven_points):
    """sqrt_s_GeV comes from the channel's own Table 8-11 mean (0.1 MeV
    precision); Table 1's value is that number rounded to 10 MeV."""
    for channel in ("qq", "mumu"):
        comp = ccs.compute_channel(seven_points, channel)
        published = pd.read_csv(ccs.DATA_RAW / ccs.CHANNELS[channel]["published_file"])
        merged = comp.merge(published, on=["year", "sample"])
        assert (merged["sqrt_s_GeV"] == merged["sqrt_s_mean_GeV"]).all()
        assert ((merged["sqrt_s_GeV"] - merged["sqrt_s_table1_GeV"]).abs() <= 0.005 + 1e-9).all()


def test_computed_over_measured_is_the_luminosity_ratio(seven_points):
    """N and f are shared, so computed / measured = L_channel / L_qq. For qq
    itself that leaves only Table 1's 0.1 pb^-1 rounding of L (<= 0.6% here);
    for mu+mu- at the 1993 peak it implies L_mumu = 8.15 pb^-1 against
    Table 1's L_qq = 8.8 pb^-1."""
    qq = ccs.compute_channel(seven_points, "qq")
    assert (qq["diff_pct"].abs() < 100 * 0.05 / qq["L_pbinv"]).all()
    mumu = ccs.compute_channel(seven_points, "mumu")
    row = mumu[(mumu["year"] == 1993) & (mumu["sample"] == "peak")]
    assert ccs.implied_channel_luminosity(row).iloc[0] == pytest.approx(8.15, abs=0.01)
