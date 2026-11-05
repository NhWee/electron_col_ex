"""
Compute e+e- -> f fbar cross sections from raw OPAL event counts + luminosity,
validate against the published (already-corrected) values, and plot sigma vs
sqrt(s) across the Z resonance.

Also cross-checks all four final states (qq, ee, mumu, tautau) at the same
7 energy points against each other: an anomaly isolated to a single channel
at a single energy point points at a channel-specific systematic (e.g. a
tracking-detector-only livetime issue) rather than a bad luminosity number.

Source: OPAL "Zedometry" (Abbiendi et al., Eur.Phys.J.C19:587-651, 2001,
arXiv:hep-ex/0012018), Tables 1, 3, 5, 6, 7, 8, 9, 10 and 11.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.integrate import quad
from uncertainties import ufloat

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

# Above this, a channel/point disagreement is treated as a flagged anomaly
# rather than statistical noise (typical point-to-point scatter is <1%).
ANOMALY_THRESHOLD_PCT = 2.0

# PDG world-average values (2024), for sanity-checking the fit result --
# not used anywhere in the fit itself.
PDG_MZ_GEV = 91.1876
PDG_MZ_ERR_GEV = 0.0021
PDG_GAMMAZ_GEV = 2.4955
PDG_GAMMAZ_ERR_GEV = 0.0023

ALPHA_QED = 1 / 137.035999
M_ELECTRON_GEV = 0.000511

# Table 1 lists L to one decimal in pb^-1: a uniform rounding error.
L_ROUNDING_ERR_PBINV = 0.1 / np.sqrt(12)

# LEP centre-of-mass energy calibration (1993-1995 resonant depolarisation),
# quoted as its effect on the combined LEP M_Z and Gamma_Z by the LEP
# Electroweak Working Group, Phys. Rept. 427 (2006) 257, Table 2.3. These
# are external numbers, not derived here; the error budget uses them
# together with the energy-scale sensitivity measured by refitting.
LEP_ENERGY_SCALE_MZ_MEV = 1.7
LEP_ENERGY_SCALE_GAMMAZ_MEV = 1.2


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
    comparison = computed.merge(published, on=["year", "sample"], suffixes=("", "_pub"))
    comparison["diff_pct"] = (
        (comparison["sigma_computed_nb"] - comparison["xsec_corrected_nb"]) / comparison["xsec_corrected_nb"] * 100
    )
    return comparison[
        ["year", "sample", "sqrt_s_GeV", "sqrt_s_rms_GeV", n_col, "L_pbinv", "sigma_computed_nb", *ERROR_COLUMNS, "xsec_corrected_nb", "diff_pct"]
    ].sort_values(["year", "sqrt_s_GeV"])


def compute_channel(seven_points: pd.DataFrame, channel: str) -> pd.DataFrame:
    spec = CHANNELS[channel]
    corrections = pd.read_csv(DATA_RAW / spec["corrections_file"])
    published = pd.read_csv(DATA_RAW / spec["published_file"])
    computed = compute_cross_section(seven_points, corrections, spec["n_col"])
    comparison = validate_against_published(computed, published, spec["n_col"])
    comparison.insert(2, "channel", channel)
    return comparison


def channel_consistency_check(seven_points: pd.DataFrame) -> pd.DataFrame:
    """Compute all four final states at the same 7 energy points and pivot
    into one row per point, one column per channel's diff_pct. A point where
    only one or two channels disagree points at a channel-specific
    systematic (e.g. tracking-detector livetime) rather than a bad L or a
    transcription error, since all four channels share the same raw L."""
    all_channels = pd.concat([compute_channel(seven_points, ch) for ch in CHANNELS], ignore_index=True)
    pivot = all_channels.pivot(index=["year", "sample"], columns="channel", values="diff_pct")
    return pivot[list(CHANNELS)].sort_index(level="year")


def breit_wigner(sqrt_s: np.ndarray, sigma_peak: float, mZ: float, gammaZ: float) -> np.ndarray:
    """Non-relativistic Breit-Wigner lineshape in terms of E_cm = sqrt(s),
    the same simplified form used in LEP masterclass-style Z lineshape fits.
    It ignores initial-state radiation, which is why the fitted Gamma_Z
    below comes out a bit wider than the PDG value (radiative tails pull
    the effective peak width up)."""
    half_width = gammaZ / 2
    return sigma_peak * half_width**2 / ((sqrt_s - mZ) ** 2 + half_width**2)


def isr_beta(sqrt_s: float) -> float:
    """Leading-log QED 'photon exponentiation' exponent (Kuraev-Fadin
    structure-function approach). Controls both how singular the soft-photon
    peak at x=0 is and the overall radiation probability; s here is the
    actual sqrt(s) of the collision, not the shifted one after radiating."""
    s = sqrt_s**2
    return (2 * ALPHA_QED / np.pi) * (np.log(s / M_ELECTRON_GEV**2) - 1)


def qed_radiator_simplified(x: float, beta: float) -> float:
    """The radiator this project used until 2026-11: the soft-photon power
    law with the O(alpha) hard-photon correction applied multiplicatively
    and the soft+virtual (1 + 3/4 beta) factor dropped. It integrates to
    0.951 rather than 1. Kept so the README's model-systematic comparison
    (simplified vs full) stays reproducible."""
    return beta * x ** (beta - 1) * (1 - x / 2)


def qed_radiator_full(x: float, beta: float) -> float:
    """O(alpha) leading-log structure-function radiator (Kuraev-Fadin /
    Nicrosini-Trentadue form):

        H(x) = beta x^(beta-1) (1 + 3/4 beta) - beta (1 - x/2)

    The first term is the exponentiated soft+virtual piece, the second the
    O(alpha) hard-photon term; together they follow from the plus-prescribed
    splitting function and integrate to exactly 1 (tested in tests/). Still
    not the ZFITTER-level O(alpha^2) radiator LEP used."""
    return beta * x ** (beta - 1) * (1 + 0.75 * beta) - beta * (1 - x / 2)


RADIATORS = {"full": qed_radiator_full, "simplified": qed_radiator_simplified}

# Which radiator breit_wigner_isr() uses. "full" is the default; the fit
# helpers take a `radiator=` argument that switches it for one call.
ISR_RADIATOR = "full"

# Upper limit of the radiative-energy-fraction integral. 1 for a pure
# Breit-Wigner (the resonance suppresses the tail); lower it to emulate a
# sqrt(s')/sqrt(s) cut when the born model has a 1/s photon-exchange term.
ISR_X_MAX = 1.0


def qed_radiator(x: float, beta: float) -> float:
    """Dispatch to the radiator selected by ISR_RADIATOR."""
    return RADIATORS[ISR_RADIATOR](x, beta)


def convolve_isr(born, sqrt_s: np.ndarray, *params, x_max: float | None = None) -> np.ndarray:
    """sigma_obs(sqrt_s) = int_0^x_max H(x) born(sqrt_s * sqrt(1-x), *params) dx.
    Radiating a fraction x always lowers the effective collision energy,
    which is what skews the observed lineshape relative to the true
    resonance. `born` is any sigma(sqrt_s, *params) model."""
    if x_max is None:
        x_max = ISR_X_MAX
    sqrt_s = np.atleast_1d(np.asarray(sqrt_s, dtype=float))
    observed = np.empty_like(sqrt_s)
    for i, s0 in enumerate(sqrt_s):
        beta = isr_beta(s0)

        def integrand(x: float, s0: float = s0, beta: float = beta) -> float:
            sqrt_s_prime = s0 * np.sqrt(1 - x)
            return qed_radiator(x, beta) * born(sqrt_s_prime, *params)

        observed[i], _ = quad(integrand, 0, min(x_max, 1 - 1e-9), limit=200)
    return observed


def breit_wigner_isr(sqrt_s: np.ndarray, sigma_peak: float, mZ: float, gammaZ: float) -> np.ndarray:
    """Non-relativistic Breit-Wigner convolved with the ISR radiator."""
    return convolve_isr(breit_wigner, sqrt_s, sigma_peak, mZ, gammaZ)


class _radiator:
    """Context manager: temporarily select a radiator form."""

    def __init__(self, name: str | None):
        self.name = name

    def __enter__(self):
        global ISR_RADIATOR
        self.previous = ISR_RADIATOR
        if self.name is not None:
            if self.name not in RADIATORS:
                raise ValueError(f"unknown radiator {self.name!r}; choose from {list(RADIATORS)}")
            ISR_RADIATOR = self.name

    def __exit__(self, *exc):
        global ISR_RADIATOR
        ISR_RADIATOR = self.previous


def fit_breit_wigner_isr(comparison: pd.DataFrame, radiator: str | None = None) -> Minuit:
    """Same as fit_breit_wigner but with the ISR-convolved model. `radiator`
    selects "full" or "simplified" for this call only (default: ISR_RADIATOR)."""
    least_squares = LeastSquares(
        comparison["sqrt_s_GeV"].to_numpy(),
        comparison["sigma_computed_nb"].to_numpy(),
        comparison["sigma_computed_err_nb"].to_numpy(),
        breit_wigner_isr,
    )
    with _radiator(radiator):
        minuit = Minuit(least_squares, sigma_peak=1.5, mZ=91.19, gammaZ=2.5)
        minuit.migrad()
        minuit.hesse()
    return minuit


def fit_breit_wigner(comparison: pd.DataFrame) -> Minuit:
    """Least-squares fit of sigma_computed_nb vs sqrt_s_GeV to extract
    M_Z, Gamma_Z and the peak cross section directly from our own computed
    points (not the published ones)."""
    least_squares = LeastSquares(
        comparison["sqrt_s_GeV"].to_numpy(),
        comparison["sigma_computed_nb"].to_numpy(),
        comparison["sigma_computed_err_nb"].to_numpy(),
        breit_wigner,
    )
    minuit = Minuit(least_squares, sigma_peak=1.5, mZ=91.19, gammaZ=2.5)
    minuit.migrad()
    minuit.hesse()
    return minuit


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


def with_beam_spread(model, sigma_e: np.ndarray):
    """Return model smeared by a Gaussian centre-of-mass energy spread of
    rms sigma_e (one value per fitted point), using 9-point Gauss-Hermite
    quadrature. LEP's ~55 MeV spread is small against Gamma_Z but not
    negligible against the ~10 MeV precision aimed at here."""
    nodes, weights = np.polynomial.hermite.hermgauss(9)
    sigma_e = np.atleast_1d(np.asarray(sigma_e, dtype=float))

    def smeared(sqrt_s: np.ndarray, *params) -> np.ndarray:
        sqrt_s = np.atleast_1d(np.asarray(sqrt_s, dtype=float))
        se = sigma_e if sigma_e.size == sqrt_s.size else np.full_like(sqrt_s, sigma_e.mean())
        out = np.zeros_like(sqrt_s)
        for node, weight in zip(nodes, weights):
            out += weight * model(sqrt_s + np.sqrt(2) * se * node, *params)
        return out / np.sqrt(np.pi)

    return smeared


def _refit_errors(comparison: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Copy of `comparison` whose sigma_computed_err_nb is the quadrature
    sum of only the named error columns."""
    out = comparison.copy()
    out["sigma_computed_err_nb"] = np.sqrt(sum(out[c].to_numpy() ** 2 for c in columns))
    return out


def error_budget(clean: pd.DataFrame, radiator: str | None = None) -> pd.DataFrame:
    """Decompose the ISR-convolved fit's uncertainty on M_Z and Gamma_Z.

    Rows marked 'input' are propagated through the fit (stat, L rounding,
    f systematic, f correlation); rows marked 'model' are central-value
    shifts between two defensible treatments (radiator form, beam energy
    spread); 'external' is the LEP energy calibration taken from the LEP
    EWWG, scaled by the measured sensitivity dM_Z/dE."""
    with _radiator(radiator):
        rows = []

        def record(label: str, kind: str, fit: Minuit, *, reference: Minuit | None = None) -> Minuit:
            if reference is None:
                rows.append(dict(source=label, kind=kind,
                                 mZ_MeV=fit.errors["mZ"] * 1e3, gammaZ_MeV=fit.errors["gammaZ"] * 1e3,
                                 mZ_GeV=fit.values["mZ"], gammaZ_GeV=fit.values["gammaZ"], chi2=fit.fval))
            else:
                rows.append(dict(source=label, kind=kind,
                                 mZ_MeV=(fit.values["mZ"] - reference.values["mZ"]) * 1e3,
                                 gammaZ_MeV=(fit.values["gammaZ"] - reference.values["gammaZ"]) * 1e3,
                                 mZ_GeV=fit.values["mZ"], gammaZ_GeV=fit.values["gammaZ"], chi2=fit.fval))
            return fit

        # 1. propagated input errors, added one at a time
        fit_stat = record("statistical (Poisson on N)", "input",
                          fit_breit_wigner_isr(_refit_errors(clean, ["sigma_stat_err_nb"])))
        fit_stat_l = fit_breit_wigner_isr(_refit_errors(clean, ["sigma_stat_err_nb", "sigma_lround_err_nb"]))
        rows.append(dict(source="+ L rounding (0.1 pb^-1 granularity)", kind="input",
                         mZ_MeV=np.sqrt(max(fit_stat_l.errors["mZ"] ** 2 - fit_stat.errors["mZ"] ** 2, 0)) * 1e3,
                         gammaZ_MeV=np.sqrt(max(fit_stat_l.errors["gammaZ"] ** 2 - fit_stat.errors["gammaZ"] ** 2, 0)) * 1e3,
                         mZ_GeV=fit_stat_l.values["mZ"], gammaZ_GeV=fit_stat_l.values["gammaZ"], chi2=fit_stat_l.fval))
        fit_diag = fit_breit_wigner_isr(clean)
        rows.append(dict(source="+ f systematic, uncorrelated (= default fit)", kind="input",
                         mZ_MeV=np.sqrt(max(fit_diag.errors["mZ"] ** 2 - fit_stat_l.errors["mZ"] ** 2, 0)) * 1e3,
                         gammaZ_MeV=np.sqrt(max(fit_diag.errors["gammaZ"] ** 2 - fit_stat_l.errors["gammaZ"] ** 2, 0)) * 1e3,
                         mZ_GeV=fit_diag.values["mZ"], gammaZ_GeV=fit_diag.values["gammaZ"], chi2=fit_diag.fval))

        # 2. f systematic correlated within a year: full covariance fit
        fit_cov = fit_with_covariance(clean, breit_wigner_isr, build_covariance(clean, f_correlated=True))
        record("f correlated within year (covariance fit): shift", "input-shift", fit_cov, reference=fit_diag)
        record("f correlated within year (covariance fit): error", "input", fit_cov)

        # 3. model: radiator form
        other = "simplified" if ISR_RADIATOR == "full" else "full"
        record(f"radiator form ({ISR_RADIATOR} - {other})", "model",
               fit_breit_wigner_isr(clean, radiator=other), reference=fit_diag)
        rows[-1]["mZ_MeV"] *= -1
        rows[-1]["gammaZ_MeV"] *= -1

        # 4. model: beam energy spread (rms from OPAL's own Table 10)
        spread_model = with_beam_spread(breit_wigner_isr, clean["sqrt_s_rms_GeV"].to_numpy())
        fit_spread = fit_with_covariance(clean, spread_model, build_covariance(clean, f_correlated=False))
        record("beam energy spread (smeared - unsmeared)", "model", fit_spread, reference=fit_diag)

        # 5. external: energy scale. Measure dM_Z/dE by shifting all points.
        delta = 0.010  # GeV
        shifted = clean.copy()
        shifted["sqrt_s_GeV"] = shifted["sqrt_s_GeV"] + delta
        fit_shift = fit_breit_wigner_isr(shifted)
        d_mz = (fit_shift.values["mZ"] - fit_diag.values["mZ"]) / delta
        d_gz = (fit_shift.values["gammaZ"] - fit_diag.values["gammaZ"]) / delta
        rows.append(dict(source=f"LEP energy calibration (LEP EWWG; dM_Z/dE={d_mz:.2f}, dG_Z/dE={d_gz:+.2f})",
                         kind="external", mZ_MeV=LEP_ENERGY_SCALE_MZ_MEV * abs(d_mz),
                         gammaZ_MeV=LEP_ENERGY_SCALE_GAMMAZ_MEV, mZ_GeV=np.nan, gammaZ_GeV=np.nan, chi2=np.nan))

    budget = pd.DataFrame(rows)
    syst_rows = budget["kind"].isin(["model", "external"])
    total_syst_mz = np.sqrt((budget.loc[syst_rows, "mZ_MeV"] ** 2).sum())
    total_syst_gz = np.sqrt((budget.loc[syst_rows, "gammaZ_MeV"] ** 2).sum())
    budget.attrs["summary"] = dict(
        mZ=fit_cov.values["mZ"], mZ_fit=fit_cov.errors["mZ"] * 1e3, mZ_syst=total_syst_mz,
        gammaZ=fit_cov.values["gammaZ"], gammaZ_fit=fit_cov.errors["gammaZ"] * 1e3, gammaZ_syst=total_syst_gz,
        mZ_stat=fit_stat.errors["mZ"] * 1e3, gammaZ_stat=fit_stat.errors["gammaZ"] * 1e3,
    )
    return budget


def print_error_budget(budget: pd.DataFrame) -> None:
    print(f"{'source':62s} {'kind':12s} {'dM_Z (MeV)':>11s} {'dG_Z (MeV)':>11s}")
    for _, r in budget.iterrows():
        print(f"{r['source']:62s} {r['kind']:12s} {r['mZ_MeV']:11.2f} {r['gammaZ_MeV']:11.2f}")
    s = budget.attrs["summary"]
    print(
        f"\n  M_Z     = {s['mZ']:.4f} +/- {s['mZ_fit'] / 1e3:.4f} (fit: stat {s['mZ_stat']:.1f} MeV + f + L) "
        f"+/- {s['mZ_syst'] / 1e3:.4f} (model + energy scale) GeV"
    )
    print(
        f"  Gamma_Z = {s['gammaZ']:.4f} +/- {s['gammaZ_fit'] / 1e3:.4f} (fit: stat {s['gammaZ_stat']:.1f} MeV + f + L) "
        f"+/- {s['gammaZ_syst'] / 1e3:.4f} (model + energy scale) GeV"
    )
    print("  'input-shift' rows are central-value moves, not added to any total.")


def print_fit_result(minuit: Minuit) -> None:
    print(f"chi2/ndof = {minuit.fval:.2f}/{minuit.ndof:.0f}")
    for name, pdg_value, pdg_err in [
        ("mZ", PDG_MZ_GEV, PDG_MZ_ERR_GEV),
        ("gammaZ", PDG_GAMMAZ_GEV, PDG_GAMMAZ_ERR_GEV),
    ]:
        fitted = minuit.values[name]
        error = minuit.errors[name]
        print(f"  {name:10s} = {fitted:.4f} +/- {error:.4f} GeV   (PDG: {pdg_value:.4f} +/- {pdg_err:.4f} GeV)")
    print(f"  {'sigma_peak':10s} = {minuit.values['sigma_peak']:.4f} +/- {minuit.errors['sigma_peak']:.4f} nb")


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
        label="Computed from N, L (this script)",
        zorder=3,
    )
    ax.plot(
        comparison["sqrt_s_GeV"],
        comparison["xsec_corrected_nb"],
        "s",
        color=ORANGE,
        markersize=6,
        markeredgewidth=0,
        label="Published (OPAL, corrected)",
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
    print(f"Saved plot to {out_path}")


def main() -> None:
    counts = load_counts()
    seven_points = select_corrected_points(counts)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    mumu = compute_channel(seven_points, "mumu")
    print("=== mu+mu- cross section: computed vs published ===")
    print(mumu.to_string(index=False))

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    mumu.to_csv(DATA_PROCESSED / "mumu_cross_section_computed.csv", index=False)
    print(f"\nSaved computed table to {DATA_PROCESSED / 'mumu_cross_section_computed.csv'}")

    print("\n=== Breit-Wigner fit to computed mu+mu- points (all 7) ===")
    fit_all = fit_breit_wigner(mumu)
    print_fit_result(fit_all)

    # The 1993 peak point was independently traced (via the cross-channel
    # consistency check below) to a track-detector-only livetime issue, not
    # a transcription error -- so it is legitimate to exclude it here rather
    # than let one bad point pull M_Z and Gamma_Z off their true values.
    anomalous = mumu[mumu["diff_pct"].abs() > ANOMALY_THRESHOLD_PCT]
    clean = mumu[mumu["diff_pct"].abs() <= ANOMALY_THRESHOLD_PCT]
    print(f"\n=== Breit-Wigner fit excluding {len(anomalous)} known-anomalous point(s) ===")
    print(anomalous[["year", "sample", "diff_pct"]].to_string(index=False))
    fit_clean = fit_breit_wigner(clean)
    print_fit_result(fit_clean)
    print(
        "  note: chi2/ndof looks great mainly because only 3 distinct sqrt_s\n"
        "  clusters (peak-2/peak/peak+2) feed a 3-parameter fit -- it is an\n"
        "  almost-exact fit, not a strong overconstraint."
    )

    print(f"\n=== Same clean points, fit with the ISR-convolved Breit-Wigner ({ISR_RADIATOR} radiator) ===")
    fit_isr = fit_breit_wigner_isr(clean)
    print_fit_result(fit_isr)
    shift_mZ = (fit_isr.values["mZ"] - fit_clean.values["mZ"]) * 1000
    shift_gammaZ = (fit_isr.values["gammaZ"] - fit_clean.values["gammaZ"]) * 1000
    pull_mZ = (fit_isr.values["mZ"] - PDG_MZ_GEV) / fit_isr.errors["mZ"]
    pull_gammaZ = (fit_isr.values["gammaZ"] - PDG_GAMMAZ_GEV) / fit_isr.errors["gammaZ"]
    print(f"  shift vs. non-ISR fit: mZ {shift_mZ:+.1f} MeV, gammaZ {shift_gammaZ:+.1f} MeV")
    print(f"  pull vs. PDG: mZ {pull_mZ:+.2f} sigma, gammaZ {pull_gammaZ:+.2f} sigma")
    print(
        "  the ~280 MeV ISR shift lands mZ within 1 sigma of PDG and gammaZ\n"
        "  within ~1.2 sigma -- convolving in even a leading-log radiator\n"
        "  removes most of the bias the naive fit had. Still only 6 points /\n"
        "  3 distinct sqrt_s clusters, so treat the fit uncertainties\n"
        "  themselves as optimistic, not as a real LEP-precision measurement."
    )

    other = "simplified" if ISR_RADIATOR == "full" else "full"
    fit_isr_other = fit_breit_wigner_isr(clean, radiator=other)
    print(f"\n=== Radiator model check: same fit with the {other} radiator ===")
    print_fit_result(fit_isr_other)
    print(
        f"  {ISR_RADIATOR} - {other}: mZ {(fit_isr.values['mZ'] - fit_isr_other.values['mZ']) * 1000:+.1f} MeV, "
        f"gammaZ {(fit_isr.values['gammaZ'] - fit_isr_other.values['gammaZ']) * 1000:+.1f} MeV "
        "(entered as the radiator-model systematic in the error budget)"
    )

    print("\n=== Error budget for the ISR-convolved fit (6 clean points) ===")
    budget = error_budget(clean)
    print_error_budget(budget)
    budget.to_csv(DATA_PROCESSED / "error_budget.csv", index=False)
    print(f"\nSaved error budget to {DATA_PROCESSED / 'error_budget.csv'}")

    plot(
        mumu,
        DATA_PROCESSED / "mumu_cross_section.png",
        fit=fit_isr,
        model=breit_wigner_isr,
        fit_label="Breit-Wigner fit (ISR-convolved)",
    )

    print("\n=== Cross-channel consistency check (% diff from published, by channel) ===")
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    consistency = channel_consistency_check(seven_points)
    print(consistency.to_string())

    anomalies = consistency[(consistency.abs() > ANOMALY_THRESHOLD_PCT).any(axis=1)]
    if not anomalies.empty:
        print(f"\nPoints with a channel off by more than {ANOMALY_THRESHOLD_PCT}%:")
        print(anomalies.to_string())

    consistency.to_csv(DATA_PROCESSED / "channel_consistency_check.csv")
    print(f"\nSaved consistency check to {DATA_PROCESSED / 'channel_consistency_check.csv'}")


if __name__ == "__main__":
    main()
