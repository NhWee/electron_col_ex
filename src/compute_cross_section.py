"""
Compute e+e- -> f fbar cross sections from raw OPAL event counts + luminosity,
validate them against the published values, and fit the mu+mu- lineshape for
M_Z and Gamma_Z.

Also cross-checks all four final states (qq, ee, mumu, tautau) at the same
7 energy points against each other. OPAL counts each final state only while
the detectors its selection needs are working, so every channel has its own
integrated luminosity, but Table 1 lists the qq one only (paper section 8).
Since N and f are the same on both sides, computed / published-measured is
exactly L_channel / L_qq, and a point where only some channels disagree
measures that luminosity difference rather than a transcription error.

Source: OPAL, "Precise determination of the Z resonance parameters at LEP:
'Zedometry'", Eur. Phys. J. C19 (2001) 587-651, arXiv:hep-ex/0012018,
Tables 1-3 and 5-11.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.integrate import quad

from electroweak import ALPHA_0, born_cross_section_nb, fixed_to_running_width

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

# One entry per final state: which N column it uses, and which paper table
# (correction factors / published cross section) it was extracted from.
CHANNELS = {
    "qq": {
        "n_col": "N_had",
        "corrections_file": "opal_zedometry_table3_qq_corrections.csv",
        "published_file": "opal_zedometry_table8_qq_xsec_published.csv",
    },
    "ee": {
        "n_col": "N_ee",
        "corrections_file": "opal_zedometry_table5_ee_corrections.csv",
        "published_file": "opal_zedometry_table9_ee_xsec_published.csv",
    },
    "mumu": {
        "n_col": "N_mumu",
        "corrections_file": "opal_zedometry_table6_mumu_corrections.csv",
        "published_file": "opal_zedometry_table10_mumu_xsec_published.csv",
    },
    "tautau": {
        "n_col": "N_tautau",
        "corrections_file": "opal_zedometry_table7_tautau_corrections.csv",
        "published_file": "opal_zedometry_table11_tautau_xsec_published.csv",
    },
}

# Above this, a channel/point disagreement is treated as a flagged anomaly.
# computed / measured = L_channel / L_qq, which the Table 1 caption says
# varies "within about 1%" between channels.
ANOMALY_THRESHOLD_PCT = 2.0

# PDG world-average values (2024), for sanity-checking the fit result --
# not used anywhere in the fit itself.
PDG_MZ_GEV = 91.1876
PDG_MZ_ERR_GEV = 0.0021
PDG_GAMMAZ_GEV = 2.4955
PDG_GAMMAZ_ERR_GEV = 0.0023

ALPHA_QED = ALPHA_0  # alpha(0): the radiator's beta and the A_FB fits use it
M_ELECTRON_GEV = 0.000511

# Table 1 lists L to one decimal in pb^-1: a uniform rounding error.
L_ROUNDING_ERR_PBINV = 0.1 / np.sqrt(12)

# Kinematic acceptance of the published cross sections (paper Table 2):
# qq is defined for s'/s > 0.01, mu+mu- and tau+tau- for m_ff^2/s > 0.01
# (the paper notes the two definitions differ negligibly here). In the ISR
# convolution this is x = 1 - s'/s < 0.99.
SPRIME_OVER_S_MIN = 0.01

# Effect of the LEP centre-of-mass energy calibration on OPAL's own M_Z and
# Gamma_Z, quoted in section 4 of the paper. External numbers, not derived
# here; the M_Z entry is scaled by the dM_Z/dE sensitivity measured by
# refitting the selected points with all energies shifted.
ENERGY_SCALE_MZ_MEV = 1.8
ENERGY_SCALE_GAMMAZ_MEV = 1.3

# Uncertainty of the centre-of-mass energy spread (paper section 4):
# 1.1 MeV in 1993-1994 and 1.3 MeV in 1995.
BEAM_SPREAD_ERR_GEV = {1993: 0.0011, 1994: 0.0011, 1995: 0.0013}


def load_counts() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "opal_zedometry_table1_event_counts.csv")


def select_corrected_points(counts: pd.DataFrame) -> pd.DataFrame:
    """The correction-factor tables only cover 7 points: 1993/1995
    peak-2/peak/peak+2, and a single 1994 'peak' column. The 1994 'peak(ab)'
    row is the dominant-statistics sub-period that column corresponds to, so
    it gets a separate `corr_sample` join key while keeping its own `sample`
    label (needed later to match the published tables, which also use
    'peak(ab)')."""
    points_1993_1995 = counts[
        counts["sample"].isin(["peak-2", "peak", "peak+2"]) & counts["year"].isin([1993, 1995])
    ].copy()
    points_1993_1995["corr_sample"] = points_1993_1995["sample"]

    point_1994 = counts[(counts["year"] == 1994) & (counts["sample"] == "peak(ab)")].copy()
    point_1994["corr_sample"] = "peak"

    return pd.concat([points_1993_1995, point_1994], ignore_index=True)


def compute_cross_section(counts: pd.DataFrame, corrections: pd.DataFrame, n_col: str) -> pd.DataFrame:
    """sigma = f * N / L with three error sources kept separately:

    - stat:   Poisson on N
    - f_syst: the correction factor's own systematic (correlated between
              energy points of the same year, see build_covariance)
    - L_round: Table 1 quotes L to 0.1 pb^-1, so the value we read carries a
              uniform rounding error of width 0.1, i.e. sigma = 0.1/sqrt(12)

    `sigma_computed_err_nb` is their quadrature sum and is what the simple
    (diagonal) fits use."""
    merged = counts.merge(corrections, left_on=["year", "corr_sample"], right_on=["year", "sample"], suffixes=("", "_corr"))

    rows = []
    for _, row in merged.iterrows():
        n_val, f_val, l_val = row[n_col], row["total_correction_f"], row["L_pbinv"]
        f_err = f_val * row["total_df_over_f_pct"] / 100
        sigma = f_val * n_val / l_val / 1000  # pb -> nb
        stat = sigma * np.sqrt(n_val) / n_val
        f_syst = sigma * f_err / f_val
        l_round = sigma * L_ROUNDING_ERR_PBINV / l_val
        rows.append((sigma, stat, f_syst, l_round, np.sqrt(stat**2 + f_syst**2 + l_round**2)))

    (
        merged["sigma_computed_nb"],
        merged["sigma_stat_err_nb"],
        merged["sigma_fsyst_err_nb"],
        merged["sigma_lround_err_nb"],
        merged["sigma_computed_err_nb"],
    ) = map(list, zip(*rows))
    return merged


ERROR_COLUMNS = ["sigma_computed_err_nb", "sigma_stat_err_nb", "sigma_fsyst_err_nb", "sigma_lround_err_nb"]


def validate_against_published(computed: pd.DataFrame, published: pd.DataFrame, n_col: str) -> pd.DataFrame:
    """Attach OPAL's per-channel beam energies and compare with its
    published cross section.

    Energies: Table 1 rounds sqrt(s) to 10 MeV. Tables 8-11 give the
    luminosity-weighted mean sqrt(s) of each channel's own sample to 0.1 MeV
    (and its rms spread), so `sqrt_s_GeV` is replaced by that value; the
    Table 1 number is kept as `sqrt_s_table1_GeV`. These are LEP beam
    energies, an input, not a result of OPAL's cross-section analysis.

    Comparison: the published 'measured' cross section is f N / L_channel
    and still contains the beam-energy spread, like our f N / L_qq; the
    'corrected' one has the spread unfolded (paper section 8). So diff_pct
    is taken against 'measured', and computed / measured = L_channel / L_qq
    exactly."""
    comparison = computed.merge(published, on=["year", "sample"], suffixes=("", "_pub"))
    comparison["sqrt_s_table1_GeV"] = comparison["sqrt_s_GeV"]
    comparison["sqrt_s_GeV"] = comparison["sqrt_s_mean_GeV"]
    comparison["diff_pct"] = (
        (comparison["sigma_computed_nb"] - comparison["xsec_measured_nb"]) / comparison["xsec_measured_nb"] * 100
    )
    return comparison[
        ["year", "sample", "sqrt_s_GeV", "sqrt_s_table1_GeV", "sqrt_s_rms_GeV", n_col, "L_pbinv",
         "sigma_computed_nb", *ERROR_COLUMNS, "xsec_measured_nb", "xsec_corrected_nb", "diff_pct"]
    ].sort_values(["year", "sqrt_s_GeV"])


def compute_channel(seven_points: pd.DataFrame, channel: str) -> pd.DataFrame:
    spec = CHANNELS[channel]
    corrections = pd.read_csv(DATA_RAW / spec["corrections_file"])
    published = pd.read_csv(DATA_RAW / spec["published_file"])
    computed = compute_cross_section(seven_points, corrections, spec["n_col"])
    comparison = validate_against_published(computed, published, spec["n_col"])
    comparison.insert(2, "channel", channel)
    return comparison


def implied_channel_luminosity(comparison: pd.DataFrame) -> pd.Series:
    """The luminosity OPAL used for this channel, f N / sigma_measured, in
    pb^-1. Equal to L_qq * computed / measured."""
    return comparison["L_pbinv"] * comparison["sigma_computed_nb"] / comparison["xsec_measured_nb"]


def channel_consistency_check(seven_points: pd.DataFrame) -> pd.DataFrame:
    """Compute all four final states at the same 7 energy points and pivot
    into one row per point, one column per channel's diff_pct. All four use
    Table 1's (qq) luminosity, so a column that is off at one point means
    that channel's own luminosity differs there (paper section 8)."""
    all_channels = pd.concat([compute_channel(seven_points, ch) for ch in CHANNELS], ignore_index=True)
    pivot = all_channels.pivot(index=["year", "sample"], columns="channel", values="diff_pct")
    return pivot[list(CHANNELS)].sort_index(level="year")


# --- Lineshapes ----------------------------------------------------------------


def breit_wigner(sqrt_s: np.ndarray, sigma_peak: float, mZ: float, gammaZ: float) -> np.ndarray:
    """Non-relativistic Breit-Wigner lineshape in terms of E_cm = sqrt(s),
    the same simplified form used in LEP masterclass-style Z lineshape fits.
    It ignores initial-state radiation, which is why a fit of it to real
    data comes out far too high and too wide (radiative tails pull the
    effective peak up and out)."""
    half_width = gammaZ / 2
    return sigma_peak * half_width**2 / ((sqrt_s - mZ) ** 2 + half_width**2)


def breit_wigner_relativistic(sqrt_s: np.ndarray, sigma_peak: float, mZ: float, gammaZ: float) -> np.ndarray:
    """Relativistic Breit-Wigner with s-dependent (running) width, the form
    in which LEP's M_Z and Gamma_Z are defined:
        sigma(s) = sigma_peak * s Gamma^2 / ((s - M^2)^2 + s^2 Gamma^2 / M^2)
    Equals sigma_peak at s = M^2."""
    s = np.asarray(sqrt_s, dtype=float) ** 2
    return sigma_peak * s * gammaZ**2 / ((s - mZ**2) ** 2 + s**2 * gammaZ**2 / mZ**2)


# --- QED initial-state radiation -------------------------------------------------
#
# Every radiator below has the form
#
#     H(x) = beta x^(beta-1) S(beta) + R(x, beta)
#
# with x = 1 - s'/s the fraction of s lost to photons: an exponentiated
# soft+virtual part with coefficient S, and a hard part R that is at most
# logarithmic at x = 0. Keeping the two apart lets convolve_isr() integrate
# in u = x^beta, where beta x^(beta-1) dx = du turns the soft part into the
# constant S. That removes the x -> 0 singularity analytically instead of
# leaving it to the quadrature, and makes the convolution ~3x faster.


def isr_beta(sqrt_s: float) -> float:
    """Leading-log QED exponent beta = (2 alpha/pi)(ln(s/m_e^2) - 1) of the
    two-beam radiator (each beam radiates with beta/2). Controls both how
    singular the soft-photon peak at x=0 is and the overall radiation
    probability; s here is the nominal sqrt(s) of the collision, not the
    reduced one after radiating."""
    s = sqrt_s**2
    return (2 * ALPHA_QED / np.pi) * (np.log(s / M_ELECTRON_GEV**2) - 1)


def _soft_simplified(beta: float) -> float:
    return 1.0


def _hard_simplified(x: float, beta: float) -> float:
    return -0.5 * beta * x**beta


def _soft_o1(beta: float) -> float:
    return 1 + 0.75 * beta


def _hard_o1(x: float, beta: float) -> float:
    return -beta * (1 - x / 2)


def _soft_o2(beta: float) -> float:
    return 1 + 0.75 * beta + beta**2 * (9 / 32 - np.pi**2 / 12)


def _hard_o2(x: float, beta: float) -> float:
    log1mx = np.log1p(-x)
    return _hard_o1(x, beta) + beta**2 / 8 * ((2 - x) * (3 * log1mx - 4 * np.log(x)) - 4 * log1mx / x - 6 + x)


RADIATOR_PARTS = {
    "o2": (_soft_o2, _hard_o2),
    "o1": (_soft_o1, _hard_o1),
    "simplified": (_soft_simplified, _hard_simplified),
}
RADIATOR_LABELS = {
    "o2": "O(alpha^2) leading-log",
    "o1": "O(alpha) leading-log",
    "simplified": "simplified (pre-2026-11-05)",
}
RADIATOR_MATHTEXT = {"o2": r"$\mathcal{O}(\alpha^2)$", "o1": r"$\mathcal{O}(\alpha)$", "simplified": "simplified"}
# The radiator one order lower, whose difference to this one is the
# truncation systematic in error_budget().
LOWER_ORDER = {"o2": "o1", "o1": "simplified", "simplified": "o1"}


def _radiator_from_parts(name: str, x: float, beta: float) -> float:
    soft, hard = RADIATOR_PARTS[name]
    return beta * x ** (beta - 1) * soft(beta) + hard(x, beta)


def qed_radiator_o2(x: float, beta: float) -> float:
    """O(alpha^2) leading-log radiator, the default. Structure-function
    approach of Kuraev-Fadin (Sov. J. Nucl. Phys. 41 (1985) 466) and
    Nicrosini-Trentadue (Phys. Lett. B196 (1987) 551):

        H(x) = beta x^(beta-1) [1 + 3/4 beta + (9/32 - pi^2/12) beta^2]
               - beta (1 - x/2)
               + beta^2/8 [(2-x)(3 ln(1-x) - 4 ln x) - 4 ln(1-x)/x - 6 + x]

    The soft+virtual factor is the expansion of exp(beta(3/4 - gamma_E)) /
    Gamma(1 + beta). Integrates to exactly 1, and agrees with the
    convolution of two single-electron structure functions up to O(beta^3)
    terms (both checked in tests/). Leading-log only: no O(alpha) non-log
    terms and no initial-state pair emission, so not the ZFITTER/TOPAZ0
    radiator LEP used."""
    return _radiator_from_parts("o2", x, beta)


def qed_radiator_o1(x: float, beta: float) -> float:
    """O(alpha) leading-log radiator (the default from 2026-11-05 to
    2026-11-08, then called "full"):

        H(x) = beta x^(beta-1) (1 + 3/4 beta) - beta (1 - x/2)

    The first term is the exponentiated soft+virtual piece, the second the
    O(alpha) hard-photon term; together they follow from the plus-prescribed
    splitting function and integrate to exactly 1."""
    return _radiator_from_parts("o1", x, beta)


def qed_radiator_simplified(x: float, beta: float) -> float:
    """The radiator this project used until 2026-11-05: the soft-photon power
    law with the O(alpha) hard-photon correction applied multiplicatively
    and the soft+virtual (1 + 3/4 beta) factor dropped,
    beta x^(beta-1) (1 - x/2). It integrates to 0.951 rather than 1. Kept so
    that older README numbers stay reproducible."""
    return _radiator_from_parts("simplified", x, beta)


# Which radiator breit_wigner_isr() uses. The fit helpers take a
# `radiator=` argument that switches it for one call.
ISR_RADIATOR = "o2"

# Upper limit of the radiative-energy-fraction integral, from the
# acceptance definition above. Irrelevant for a pure Breit-Wigner (the
# resonance suppresses the tail) but essential once the born model has a
# 1/s' photon-exchange term, which makes x -> 1 diverge.
ISR_X_MAX = 1 - SPRIME_OVER_S_MIN


def qed_radiator(x: float, beta: float) -> float:
    """The radiator selected by ISR_RADIATOR."""
    return _radiator_from_parts(ISR_RADIATOR, x, beta)


def convolve_isr(born, sqrt_s: np.ndarray, *params, x_max: float | None = None) -> np.ndarray:
    """sigma_obs(sqrt_s) = int_0^x_max H(x) born(sqrt_s * sqrt(1-x), *params) dx.
    Radiating a fraction x always lowers the effective collision energy,
    which is what skews the observed lineshape relative to the true
    resonance. `born` is any sigma(sqrt_s, *params) model. Integrated in
    u = x^beta (see the radiator block above)."""
    if x_max is None:
        x_max = ISR_X_MAX
    soft, hard = RADIATOR_PARTS[ISR_RADIATOR]
    sqrt_s = np.atleast_1d(np.asarray(sqrt_s, dtype=float))
    observed = np.empty_like(sqrt_s)
    for i, s0 in enumerate(sqrt_s):
        beta = isr_beta(s0)
        soft_coef = soft(beta)

        def integrand(u: float, s0: float = s0, beta: float = beta, soft_coef: float = soft_coef) -> float:
            x = u ** (1 / beta)
            if x < 1e-300:  # u underflowed to x = 0: only the soft part is left
                return soft_coef * born(s0, *params)
            weight = soft_coef + hard(x, beta) * x ** (1 - beta) / beta
            return weight * born(s0 * np.sqrt(1 - x), *params)

        observed[i], _ = quad(integrand, 0, min(x_max, 1 - 1e-9) ** beta, limit=200)
    return observed


def breit_wigner_isr(sqrt_s: np.ndarray, sigma_peak: float, mZ: float, gammaZ: float) -> np.ndarray:
    """Non-relativistic Breit-Wigner convolved with the ISR radiator."""
    return convolve_isr(breit_wigner, sqrt_s, sigma_peak, mZ, gammaZ)


def breit_wigner_relativistic_isr(sqrt_s: np.ndarray, sigma_peak: float, mZ: float, gammaZ: float) -> np.ndarray:
    """Running-width Breit-Wigner convolved with the ISR radiator."""
    return convolve_isr(breit_wigner_relativistic, sqrt_s, sigma_peak, mZ, gammaZ)


def improved_born(channel: str = "mumu"):
    """sigma(sqrt_s, norm, mZ, gammaZ) = norm x the absolute improved-Born
    s-channel cross section of `channel` (electroweak.py): photon + Z +
    interference, fixed-width propagator, alpha(M_Z^2) for the photon,
    sin2(theta_W_eff) fixed to PDG. For leptons norm = 1 is the absolute
    G_F-based prediction; for qq it absorbs the colour and QCD factors
    (~3 x 1.04). Its M_Z and Gamma_Z are fixed-width parameters: convert
    with fixed_to_running_width() before comparing with PDG."""

    def model(sqrt_s: np.ndarray, norm: float, mZ: float, gammaZ: float) -> np.ndarray:
        return norm * born_cross_section_nb(sqrt_s, mZ, gammaZ, channel)

    return model


def improved_born_isr(channel: str = "mumu"):
    """improved_born(channel) convolved with the ISR radiator. The 1/s'
    photon term makes x -> 1 diverge, so here the acceptance cut
    x < ISR_X_MAX is what keeps the integral finite."""
    born = improved_born(channel)

    def model(sqrt_s: np.ndarray, norm: float, mZ: float, gammaZ: float) -> np.ndarray:
        return convolve_isr(born, sqrt_s, norm, mZ, gammaZ)

    return model


class _radiator:
    """Context manager: temporarily select a radiator form."""

    def __init__(self, name: str | None):
        self.name = name

    def __enter__(self):
        global ISR_RADIATOR
        self.previous = ISR_RADIATOR
        if self.name is not None:
            if self.name not in RADIATOR_PARTS:
                raise ValueError(f"unknown radiator {self.name!r}; choose from {list(RADIATOR_PARTS)}")
            ISR_RADIATOR = self.name

    def __exit__(self, *exc):
        global ISR_RADIATOR
        ISR_RADIATOR = self.previous


def with_beam_spread(model, sigma_e):
    """Return `model` smeared by a Gaussian centre-of-mass energy spread of
    rms sigma_e (one value per fitted point, or one for all), using 5-point
    Gauss-Hermite quadrature (identical to 15 points to 1e-8 here). The
    published 'measured' cross sections, and therefore f N / L, contain
    this spread, so the real-data fits include it. LEP's ~57 MeV is small
    against Gamma_Z but moves Gamma_Z by ~6 MeV."""
    nodes, weights = np.polynomial.hermite.hermgauss(5)
    sigma_e = np.atleast_1d(np.asarray(sigma_e, dtype=float))

    def smeared(sqrt_s: np.ndarray, sigma_peak: float, mZ: float, gammaZ: float) -> np.ndarray:
        sqrt_s = np.atleast_1d(np.asarray(sqrt_s, dtype=float))
        se = sigma_e if sigma_e.size == sqrt_s.size else np.full_like(sqrt_s, sigma_e.mean())
        out = np.zeros_like(sqrt_s)
        for node, weight in zip(nodes, weights):
            out += weight * model(sqrt_s + np.sqrt(2) * se * node, sigma_peak, mZ, gammaZ)
        return out / np.sqrt(np.pi)

    return smeared


def _least_squares_fit(comparison: pd.DataFrame, model) -> Minuit:
    least_squares = LeastSquares(
        comparison["sqrt_s_GeV"].to_numpy(),
        comparison["sigma_computed_nb"].to_numpy(),
        comparison["sigma_computed_err_nb"].to_numpy(),
        model,
    )
    minuit = Minuit(least_squares, sigma_peak=1.5, mZ=91.19, gammaZ=2.5)
    minuit.migrad()
    minuit.hesse()
    return minuit


def fit_breit_wigner(comparison: pd.DataFrame, beam_spread: bool = False) -> Minuit:
    """Least-squares fit of sigma_computed_nb vs sqrt_s_GeV to extract
    M_Z, Gamma_Z and the peak cross section directly from our own computed
    points (not the published ones). No ISR. beam_spread=True smears the
    model with each point's sqrt_s_rms_GeV."""
    model = breit_wigner
    if beam_spread:
        model = with_beam_spread(breit_wigner, comparison["sqrt_s_rms_GeV"].to_numpy())
    return _least_squares_fit(comparison, model)


def fit_breit_wigner_isr(comparison: pd.DataFrame, radiator: str | None = None, beam_spread: bool = False) -> Minuit:
    """Same as fit_breit_wigner but with the ISR-convolved model. `radiator`
    selects "o2", "o1" or "simplified" for this call only (default:
    ISR_RADIATOR)."""
    model = breit_wigner_isr
    if beam_spread:
        model = with_beam_spread(breit_wigner_isr, comparison["sqrt_s_rms_GeV"].to_numpy())
    with _radiator(radiator):
        return _least_squares_fit(comparison, model)


# --- Error budget ------------------------------------------------------------


def build_covariance(comparison: pd.DataFrame, f_correlated: bool = True) -> np.ndarray:
    """Covariance of the computed cross sections. Poisson and L-rounding
    errors are independent point to point. The correction-factor systematic
    is taken as fully correlated between energy points of the same year
    (same detector, same efficiency determination) and uncorrelated between
    years; with f_correlated=False it is diagonal, which is what the plain
    LeastSquares fits assume."""
    stat = comparison["sigma_stat_err_nb"].to_numpy()
    lround = comparison["sigma_lround_err_nb"].to_numpy()
    fsyst = comparison["sigma_fsyst_err_nb"].to_numpy()
    years = comparison["year"].to_numpy()
    cov = np.diag(stat**2 + lround**2)
    if f_correlated:
        same_year = years[:, None] == years[None, :]
        cov += np.outer(fsyst, fsyst) * same_year
    else:
        cov += np.diag(fsyst**2)
    return cov


def fit_with_covariance(comparison: pd.DataFrame, model, cov: np.ndarray, **start) -> Minuit:
    """chi2 = r^T V^-1 r with a general covariance V; otherwise identical to
    the LeastSquares fits (same model, same starting values)."""
    x = comparison["sqrt_s_GeV"].to_numpy()
    y = comparison["sigma_computed_nb"].to_numpy()
    cov_inv = np.linalg.inv(cov)

    def chi2(sigma_peak: float, mZ: float, gammaZ: float) -> float:
        r = y - model(x, sigma_peak, mZ, gammaZ)
        return float(r @ cov_inv @ r)

    chi2.errordef = Minuit.LEAST_SQUARES
    minuit = Minuit(chi2, **{"sigma_peak": 1.5, "mZ": 91.19, "gammaZ": 2.5, **start})
    minuit.migrad()
    minuit.hesse()
    return minuit


def _refit_errors(comparison: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Copy of `comparison` whose sigma_computed_err_nb is the quadrature
    sum of only the named error columns."""
    out = comparison.copy()
    out["sigma_computed_err_nb"] = np.sqrt(sum(out[c].to_numpy() ** 2 for c in columns))
    return out


# Kinds of error-budget row that enter the systematic total.
SYSTEMATIC_KINDS = ("model", "data", "external")


def error_budget(clean: pd.DataFrame, radiator: str | None = None) -> pd.DataFrame:
    """Decompose the uncertainty of the baseline fit (ISR-convolved
    Breit-Wigner smeared by the measured beam-energy spread) on M_Z and
    Gamma_Z. Row kinds:

      input        propagated through the fit (stat, L rounding, f, f correlation)
      input-shift  central-value move from correlating f; not added to totals
      model        shift between two defensible treatments (radiator order,
                   beam-spread uncertainty)
      data         shift from the per-channel luminosity Table 1 does not list
      external     LEP energy calibration, from the paper
      convention   running-width minus baseline propagator: a definition, not
                   an uncertainty; add it to compare with PDG. Not in totals
      omission     physics the Breit-Wigner leaves out (photon exchange and
                   gamma/Z interference), measured by the improved-Born fit.
                   A correction, not an uncertainty. Not in totals
      applied      size of a correction already in the baseline. Not in totals

    The quoted baseline is the covariance fit (f correlated within a year).
    Baseline + convention + omission is the complete-model result in LEP's
    running-width convention, the one to compare with PDG."""
    rms = clean["sqrt_s_rms_GeV"].to_numpy()
    diagonal = build_covariance(clean, f_correlated=False)

    with _radiator(radiator):
        rows = []

        def add(label: str, kind: str, d_mz: float, d_gz: float, fit: Minuit | None = None) -> None:
            rows.append(dict(source=label, kind=kind, mZ_MeV=d_mz * 1e3, gammaZ_MeV=d_gz * 1e3,
                             mZ_GeV=fit.values["mZ"] if fit else np.nan,
                             gammaZ_GeV=fit.values["gammaZ"] if fit else np.nan,
                             chi2=fit.fval if fit else np.nan))

        def shift(fit: Minuit, reference: Minuit) -> tuple[float, float]:
            return fit.values["mZ"] - reference.values["mZ"], fit.values["gammaZ"] - reference.values["gammaZ"]

        def added_in_quadrature(fit: Minuit, previous: Minuit) -> tuple[float, float]:
            return tuple(np.sqrt(max(fit.errors[p] ** 2 - previous.errors[p] ** 2, 0)) for p in ("mZ", "gammaZ"))

        # 1. propagated input errors, added one at a time
        fit_stat = fit_breit_wigner_isr(_refit_errors(clean, ["sigma_stat_err_nb"]), beam_spread=True)
        add("statistical (Poisson on N)", "input", fit_stat.errors["mZ"], fit_stat.errors["gammaZ"], fit_stat)
        fit_stat_l = fit_breit_wigner_isr(_refit_errors(clean, ["sigma_stat_err_nb", "sigma_lround_err_nb"]), beam_spread=True)
        add("+ L rounding (0.1 pb^-1 granularity)", "input", *added_in_quadrature(fit_stat_l, fit_stat), fit_stat_l)
        fit_diag = fit_breit_wigner_isr(clean, beam_spread=True)
        add("+ f systematic, uncorrelated (= default fit)", "input", *added_in_quadrature(fit_diag, fit_stat_l), fit_diag)

        # 2. f systematic correlated within a year: full covariance fit
        fit_cov = fit_with_covariance(clean, with_beam_spread(breit_wigner_isr, rms), build_covariance(clean, f_correlated=True))
        add("f correlated within year (covariance fit): shift", "input-shift", *shift(fit_cov, fit_diag), fit_cov)
        add("f correlated within year (covariance fit): error", "input", fit_cov.errors["mZ"], fit_cov.errors["gammaZ"], fit_cov)

        # 3. model: radiator truncation, this order minus one order lower
        lower = LOWER_ORDER[ISR_RADIATOR]
        fit_lower = fit_breit_wigner_isr(clean, radiator=lower, beam_spread=True)
        add(f"radiator truncation ({ISR_RADIATOR} - {lower})", "model", *shift(fit_diag, fit_lower), fit_lower)

        # 4. model: uncertainty of the beam-energy spread itself
        rms_up = rms + clean["year"].map(BEAM_SPREAD_ERR_GEV).to_numpy()
        fit_rms = fit_with_covariance(clean, with_beam_spread(breit_wigner_isr, rms_up), diagonal)
        add("beam-energy spread uncertainty (rms + 1.1/1.3 MeV)", "model", *shift(fit_rms, fit_diag), fit_rms)

        # 5. data: Table 1 gives the qq luminosity only. OPAL's measured
        # cross section is f N / L_mumu, so fitting it instead (same errors
        # in relative terms) measures what the per-channel luminosity does.
        own_lumi = clean.copy()
        scale = own_lumi["xsec_measured_nb"] / own_lumi["sigma_computed_nb"]
        for column in ["sigma_computed_nb", *ERROR_COLUMNS]:
            own_lumi[column] = own_lumi[column] * scale
        fit_lumi = fit_breit_wigner_isr(own_lumi, beam_spread=True)
        add("channel luminosity (L_mumu from OPAL's measured sigma vs L_qq)", "data", *shift(fit_lumi, fit_diag), fit_lumi)

        # 6. external: energy scale. Measure dM_Z/dE by shifting all points.
        delta = 0.010  # GeV
        shifted = clean.copy()
        shifted["sqrt_s_GeV"] = shifted["sqrt_s_GeV"] + delta
        fit_shift = fit_breit_wigner_isr(shifted, beam_spread=True)
        d_mz_de = (fit_shift.values["mZ"] - fit_diag.values["mZ"]) / delta
        add(f"LEP energy calibration (OPAL sec. 4; dM_Z/dE={d_mz_de:.2f})", "external",
            ENERGY_SCALE_MZ_MEV / 1e3 * abs(d_mz_de), ENERGY_SCALE_GAMMAZ_MEV / 1e3)

        # 7. convention: LEP defines M_Z, Gamma_Z with the running-width propagator
        fit_run = fit_with_covariance(clean, with_beam_spread(breit_wigner_relativistic_isr, rms), diagonal)
        add("propagator convention (running width - baseline)", "convention", *shift(fit_run, fit_diag), fit_run)

        # 8. omission: photon exchange and gamma/Z interference, i.e. the
        # improved-Born fit (fixed width, converted) minus the running-width BW
        fit_iba = fit_with_covariance(clean, with_beam_spread(improved_born_isr("mumu"), rms), diagonal, sigma_peak=1.0)
        mz_iba, gz_iba = fixed_to_running_width(fit_iba.values["mZ"], fit_iba.values["gammaZ"])
        add("photon exchange + gamma/Z interference (improved Born - running BW)", "omission",
            mz_iba - fit_run.values["mZ"], gz_iba - fit_run.values["gammaZ"], fit_iba)

        # informational: the beam-spread smearing already in the baseline
        fit_unsmeared = fit_breit_wigner_isr(clean)
        add("beam-energy spread smearing (with - without)", "applied", *shift(fit_diag, fit_unsmeared), fit_unsmeared)

    budget = pd.DataFrame(rows)
    syst_rows = budget["kind"].isin(SYSTEMATIC_KINDS)
    convention = budget[budget["kind"] == "convention"].iloc[0]
    omission = budget[budget["kind"] == "omission"].iloc[0]
    budget.attrs["summary"] = dict(
        radiator=ISR_RADIATOR if radiator is None else radiator,
        mZ=fit_cov.values["mZ"], mZ_fit=fit_cov.errors["mZ"] * 1e3,
        mZ_syst=np.sqrt((budget.loc[syst_rows, "mZ_MeV"] ** 2).sum()),
        gammaZ=fit_cov.values["gammaZ"], gammaZ_fit=fit_cov.errors["gammaZ"] * 1e3,
        gammaZ_syst=np.sqrt((budget.loc[syst_rows, "gammaZ_MeV"] ** 2).sum()),
        mZ_stat=fit_stat.errors["mZ"] * 1e3, gammaZ_stat=fit_stat.errors["gammaZ"] * 1e3,
        mZ_running=fit_cov.values["mZ"] + convention["mZ_MeV"] / 1e3,
        gammaZ_running=fit_cov.values["gammaZ"] + convention["gammaZ_MeV"] / 1e3,
        mZ_complete=fit_cov.values["mZ"] + (convention["mZ_MeV"] + omission["mZ_MeV"]) / 1e3,
        gammaZ_complete=fit_cov.values["gammaZ"] + (convention["gammaZ_MeV"] + omission["gammaZ_MeV"]) / 1e3,
        iba_norm=fit_iba.values["sigma_peak"], iba_norm_err=fit_iba.errors["sigma_peak"],
        chi2=fit_cov.fval, ndof=len(clean) - 3,
    )
    return budget


def print_error_budget(budget: pd.DataFrame) -> None:
    print(f"{'source':66s} {'kind':12s} {'dM_Z (MeV)':>11s} {'dG_Z (MeV)':>11s}")
    for _, r in budget.iterrows():
        print(f"{r['source']:66s} {r['kind']:12s} {r['mZ_MeV']:11.2f} {r['gammaZ_MeV']:11.2f}")
    s = budget.attrs["summary"]
    print(f"\n  covariance fit: chi2/ndof = {s['chi2']:.2f}/{s['ndof']}")
    print(
        f"  M_Z     = {s['mZ']:.4f} +/- {s['mZ_fit'] / 1e3:.4f} (fit: stat {s['mZ_stat']:.1f} MeV + f + L) "
        f"+/- {s['mZ_syst'] / 1e3:.4f} (syst: model + data + energy scale) GeV"
    )
    print(
        f"  Gamma_Z = {s['gammaZ']:.4f} +/- {s['gammaZ_fit'] / 1e3:.4f} (fit: stat {s['gammaZ_stat']:.1f} MeV + f + L) "
        f"+/- {s['gammaZ_syst'] / 1e3:.4f} (syst: model + data + energy scale) GeV"
    )
    for label, suffix in [("Breit-Wigner, running-width convention", "running"),
                          ("complete Born model (+ photon exchange, interference)", "complete")]:
        print(f"  {label}:")
        for name, key, fit_err, syst, pdg, pdg_err in [
            ("M_Z", "mZ", s["mZ_fit"], s["mZ_syst"], PDG_MZ_GEV, PDG_MZ_ERR_GEV),
            ("Gamma_Z", "gammaZ", s["gammaZ_fit"], s["gammaZ_syst"], PDG_GAMMAZ_GEV, PDG_GAMMAZ_ERR_GEV),
        ]:
            value = s[f"{key}_{suffix}"]
            total = np.hypot(np.hypot(fit_err, syst) / 1e3, pdg_err)
            print(f"    {name:7s} = {value:.4f} GeV, {(value - pdg) / total:+.2f} sigma from PDG (fit + syst + PDG errors)")
    print(f"  improved-Born normalisation = {s['iba_norm']:.4f} +/- {s['iba_norm_err']:.4f} (1 = absolute G_F prediction)")
    print("  Rows of kind input-shift, convention, omission and applied are not added to any total.")


def print_fit_result(minuit: Minuit) -> None:
    print(f"chi2/ndof = {minuit.fval:.2f}/{minuit.ndof:.0f}   (Migrad valid: {minuit.valid})")
    for name, pdg_value, pdg_err in [
        ("mZ", PDG_MZ_GEV, PDG_MZ_ERR_GEV),
        ("gammaZ", PDG_GAMMAZ_GEV, PDG_GAMMAZ_ERR_GEV),
    ]:
        fitted = minuit.values[name]
        error = minuit.errors[name]
        print(f"  {name:10s} = {fitted:.4f} +/- {error:.4f} GeV   (PDG: {pdg_value:.4f} +/- {pdg_err:.4f} GeV)")
    print(f"  {'sigma_peak':10s} = {minuit.values['sigma_peak']:.4f} +/- {minuit.errors['sigma_peak']:.4f} nb")
    corr = minuit.covariance.correlation()
    print(f"  correlation(mZ, gammaZ) = {corr['mZ', 'gammaZ']:+.2f}, "
          f"correlation(mZ, sigma_peak) = {corr['mZ', 'sigma_peak']:+.2f}")


def plot(
    comparison: pd.DataFrame,
    out_path: Path,
    fit: Minuit | None = None,
    model=breit_wigner,
    fit_label: str = "Breit-Wigner fit",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.errorbar(
        comparison["sqrt_s_GeV"],
        comparison["sigma_computed_nb"],
        yerr=comparison["sigma_computed_err_nb"],
        fmt="o",
        color=BLUE,
        markersize=8,
        markeredgewidth=0,
        elinewidth=2,
        capsize=3,
        label="Computed from N, L, f (this script)",
        zorder=3,
    )
    ax.plot(
        comparison["sqrt_s_GeV"],
        comparison["xsec_measured_nb"],
        "s",
        color=ORANGE,
        markersize=6,
        markeredgewidth=0,
        label="Published (OPAL, measured)",
        zorder=2,
    )

    if fit is not None:
        x_smooth = np.linspace(comparison["sqrt_s_GeV"].min() - 0.3, comparison["sqrt_s_GeV"].max() + 0.3, 200)
        y_smooth = model(x_smooth, *fit.values)
        ax.plot(
            x_smooth,
            y_smooth,
            "-",
            color=AQUA,
            linewidth=2,
            label=fit_label,
            zorder=1,
        )

    ax.set_xlabel(r"$\sqrt{s}$ (GeV)", color=INK)
    ax.set_ylabel(r"$\sigma(e^+e^- \to \mu^+\mu^-)$ (nb)", color=INK)
    ax.set_title("OPAL: cross section computed from raw counts vs. published", color=INK, loc="left")

    ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)

    legend = ax.legend(frameon=False, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main() -> None:
    counts = load_counts()
    seven_points = select_corrected_points(counts)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.width", 250)

    mumu = compute_channel(seven_points, "mumu")
    print("=== mu+mu- cross section: computed vs published (OPAL 'measured' column) ===")
    print(mumu.to_string(index=False))

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    mumu.to_csv(DATA_PROCESSED / "mumu_cross_section_computed.csv", index=False)
    print(f"\nSaved computed table to {DATA_PROCESSED / 'mumu_cross_section_computed.csv'}")

    print("\n=== Breit-Wigner fit, no ISR, all 7 points (beam spread included in every real-data fit) ===")
    fit_all = fit_breit_wigner(mumu, beam_spread=True)
    print_fit_result(fit_all)

    # The 1993 peak point is low only in the track-based channels (see the
    # cross-channel check below): Table 1's qq luminosity is larger than
    # the luminosity OPAL's own mu+mu- selection used there.
    anomalous = mumu[mumu["diff_pct"].abs() > ANOMALY_THRESHOLD_PCT]
    clean = mumu[mumu["diff_pct"].abs() <= ANOMALY_THRESHOLD_PCT]
    print(f"\n=== Breit-Wigner fit, no ISR, excluding {len(anomalous)} flagged point(s) ===")
    flagged = anomalous[["year", "sample", "L_pbinv", "diff_pct"]].assign(
        L_mumu_implied_pbinv=implied_channel_luminosity(anomalous))
    print(flagged.to_string(index=False))
    fit_clean = fit_breit_wigner(clean, beam_spread=True)
    print_fit_result(fit_clean)
    print(
        "  note: chi2/ndof looks great mainly because only 3 distinct sqrt_s\n"
        "  clusters (peak-2/peak/peak+2) feed a 3-parameter fit -- it is an\n"
        "  almost-exact fit, not a strong overconstraint."
    )

    print(f"\n=== Same clean points, ISR-convolved Breit-Wigner ({RADIATOR_LABELS[ISR_RADIATOR]} radiator) ===")
    fit_isr = fit_breit_wigner_isr(clean, beam_spread=True)
    print_fit_result(fit_isr)
    shift_mZ = (fit_isr.values["mZ"] - fit_clean.values["mZ"]) * 1000
    shift_gammaZ = (fit_isr.values["gammaZ"] - fit_clean.values["gammaZ"]) * 1000
    pull_mZ = (fit_isr.values["mZ"] - PDG_MZ_GEV) / fit_isr.errors["mZ"]
    pull_gammaZ = (fit_isr.values["gammaZ"] - PDG_GAMMAZ_GEV) / fit_isr.errors["gammaZ"]
    print(f"  shift vs. non-ISR fit: mZ {shift_mZ:+.1f} MeV, gammaZ {shift_gammaZ:+.1f} MeV")
    print(f"  pull vs. PDG (fit error only): mZ {pull_mZ:+.2f} sigma, gammaZ {pull_gammaZ:+.2f} sigma")
    print(
        "  Still only 6 points in 3 distinct sqrt_s clusters: treat this as the\n"
        "  direction and size of the ISR effect, not as a LEP-precision measurement."
    )

    print("\n=== Radiator order check: same fit with each radiator ===")
    fits_by_radiator = {name: (fit_isr if name == ISR_RADIATOR else fit_breit_wigner_isr(clean, radiator=name, beam_spread=True))
                        for name in RADIATOR_PARTS}
    for name, fit in fits_by_radiator.items():
        print(f"  {RADIATOR_LABELS[name]:30s} chi2 {fit.fval:5.2f}   mZ {fit.values['mZ']:.4f}   "
              f"gammaZ {fit.values['gammaZ']:.4f}   sigma_peak {fit.values['sigma_peak']:.4f}")
    for higher, lower in [("o1", "simplified"), ("o2", "o1")]:
        d_m = (fits_by_radiator[higher].values["mZ"] - fits_by_radiator[lower].values["mZ"]) * 1000
        d_g = (fits_by_radiator[higher].values["gammaZ"] - fits_by_radiator[lower].values["gammaZ"]) * 1000
        print(f"  {higher} - {lower}: mZ {d_m:+.1f} MeV, gammaZ {d_g:+.1f} MeV")
    print("  The last difference is the radiator-truncation systematic in the error budget.")

    print("\n=== Error budget (6 clean points) ===")
    budget = error_budget(clean)
    print_error_budget(budget)
    budget.to_csv(DATA_PROCESSED / "error_budget.csv", index=False)
    print(f"\nSaved error budget to {DATA_PROCESSED / 'error_budget.csv'}")

    plot(
        mumu,
        DATA_PROCESSED / "mumu_cross_section.png",
        fit=fit_isr,
        model=with_beam_spread(breit_wigner_isr, clean["sqrt_s_rms_GeV"].mean()),
        fit_label=f"ISR-convolved Breit-Wigner fit, {RADIATOR_MATHTEXT[ISR_RADIATOR]} radiator",
    )

    print("\n=== Cross-channel consistency check (% diff from OPAL 'measured', by channel) ===")
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    consistency = channel_consistency_check(seven_points)
    print(consistency.to_string())
    print("  computed / measured = L_channel / L_qq, since N and f are the same on both sides.")

    anomalies = consistency[(consistency.abs() > ANOMALY_THRESHOLD_PCT).any(axis=1)]
    if not anomalies.empty:
        print(f"\nPoints with a channel off by more than {ANOMALY_THRESHOLD_PCT}%:")
        print(anomalies.to_string())

    consistency.to_csv(DATA_PROCESSED / "channel_consistency_check.csv")
    print(f"\nSaved consistency check to {DATA_PROCESSED / 'channel_consistency_check.csv'}")


if __name__ == "__main__":
    main()
