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
