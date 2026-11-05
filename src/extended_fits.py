"""
Three extensions of the mu+mu- lineshape measurement in
`compute_cross_section.py`, each answering one item of ROADMAP.md:

P1-1  More energy points. OPAL only tabulates correction factors f for 7
      of the 29 rows of Table 1, but the mu+mu- f barely depends on energy
      (1993: 1.0922 / 1.0920 / 1.0927 across peak-2 / peak / peak+2). The
      near-peak 1993-1995 rows without their own f (prescan, 1994 peak(c)
      and peak(d)) are given the same year's peak f, with half the spread
      of that year's f values added as a "borrowed f" systematic.

P1-3  LEP-convention lineshape models. The relativistic running-width
      Breit-Wigner (the propagator LEP's M_Z is defined with), and the full
      improved-Born photon + Z + interference cross section with the ISR
      integral cut at OPAL's s'/s acceptance, both convolved with the same
      radiator as the baseline fit.

P1-4  Joint fit of qq, mu+mu- and tau+tau- with common M_Z and Gamma_Z and
      one free peak cross section per channel, using a covariance in which
      the luminosity rounding is shared by all channels at a point and the
      f systematic is shared within a (channel, year). e+e- is left out:
      its t-channel contribution is not a Breit-Wigner. The 1993 peak point
      is dropped for the two track-based channels only (README section 1).

    python src/extended_fits.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit

import compute_cross_section as ccs
from compute_cross_section import (
    ANOMALY_THRESHOLD_PCT,
    AQUA,
    BLUE,
    CHANNELS,
    DATA_PROCESSED,
    DATA_RAW,
    ERROR_COLUMNS,
    GRIDLINE,
    INK,
    MUTED,
    ORANGE,
    PDG_GAMMAZ_GEV,
    PDG_MZ_GEV,
    SURFACE,
    breit_wigner,
    breit_wigner_isr,
    compute_cross_section,
    convolve_isr,
    fit_breit_wigner_isr,
    fit_with_covariance,
    load_counts,
    select_corrected_points,
)
from simulate_mumu import GEV2_TO_NB, SIN2_THETAW_EFF, iba_f1_f2

# PDG 2024 for comparison only
PDG_R_LEPTON = 20.767
PDG_SIGMA0_HAD_NB = 41.481

# OPAL defines its leptonic cross sections within an s'/s acceptance so that
# radiative-return events (a hard ISR photon bringing the collision back
# down to the Z) are excluded. The value below is the cut as recalled from
# the OPAL lineshape paper; it could not be re-checked against the PDF from
# this environment and should be verified (ROADMAP P1-3).
SPRIME_OVER_S_MIN = 0.7225
X_MAX_FROM_CUT = 1 - SPRIME_OVER_S_MIN


# --- P1-1: extended point selection ------------------------------------------

EXTRA_SAMPLES = {
    1993: ["prescan(a)", "prescan(b)"],
    1994: ["peak(c)", "peak(d)"],
    1995: ["prescan(a)", "prescan(b)"],
}


def select_extended_points(counts: pd.DataFrame) -> pd.DataFrame:
    """The 7 points with their own f, plus the near-peak 1993-1995 rows that
    borrow the same year's 'peak' f. Column `borrowed_f` marks the latter."""
    own = select_corrected_points(counts)
    own["borrowed_f"] = False
    extras = []
    for year, samples in EXTRA_SAMPLES.items():
        rows = counts[(counts["year"] == year) & counts["sample"].isin(samples)].copy()
        rows["corr_sample"] = "peak"
        rows["borrowed_f"] = True
        extras.append(rows)
    return pd.concat([own, *extras], ignore_index=True)


def borrowed_f_relative_error(corrections: pd.DataFrame) -> dict[int, float]:
    """Half the spread of a year's tabulated f, relative to its mean: the
    systematic assigned to a point that uses a neighbour's f."""
    out = {}
    for year, grp in corrections.groupby("year"):
        f = grp["total_correction_f"]
        out[int(year)] = float((f.max() - f.min()) / 2 / f.mean()) if len(f) > 1 else 0.0
    return out


def compute_channel_extended(points: pd.DataFrame, channel: str) -> pd.DataFrame:
    spec = CHANNELS[channel]
    corrections = pd.read_csv(DATA_RAW / spec["corrections_file"])
    published = pd.read_csv(DATA_RAW / spec["published_file"])
    computed = compute_cross_section(points, corrections, spec["n_col"])

    # the 1994 'peak' f has only one row, so 1994 borrows get the mean
    # of the other years' relative spreads
    rel = borrowed_f_relative_error(corrections)
    fallback = np.mean([v for v in rel.values() if v > 0]) if any(v > 0 for v in rel.values()) else 0.0
    rel_err = computed["year"].map(lambda y: rel.get(int(y), 0.0) or fallback).to_numpy()
    computed["sigma_borrow_err_nb"] = np.where(computed["borrowed_f"], computed["sigma_computed_nb"] * rel_err, 0.0)
    computed["sigma_computed_err_nb"] = np.sqrt(computed["sigma_computed_err_nb"] ** 2 + computed["sigma_borrow_err_nb"] ** 2)

    out = computed.merge(
        published[["year", "sample", "sqrt_s_rms_GeV", "xsec_corrected_nb"]], on=["year", "sample"], how="left"
    )
    out["diff_pct"] = (out["sigma_computed_nb"] - out["xsec_corrected_nb"]) / out["xsec_corrected_nb"] * 100
    out.insert(2, "channel", channel)
    cols = ["year", "sample", "channel", "sqrt_s_GeV", "sqrt_s_rms_GeV", spec["n_col"], "L_pbinv", "borrowed_f",
            "sigma_computed_nb", *ERROR_COLUMNS, "sigma_borrow_err_nb", "xsec_corrected_nb", "diff_pct"]
    return out[cols].sort_values(["year", "sqrt_s_GeV"]).reset_index(drop=True)


# --- P1-3: LEP-convention lineshape models ------------------------------------


def breit_wigner_relativistic(sqrt_s: np.ndarray, sigma_peak: float, mZ: float, gammaZ: float) -> np.ndarray:
    """Relativistic Breit-Wigner with s-dependent (running) width, the form
    in which LEP's M_Z and Gamma_Z are defined:
        sigma(s) = sigma_peak * s Gamma^2 / ((s - M^2)^2 + s^2 Gamma^2 / M^2)
    Equals sigma_peak at s = M^2."""
    s = np.asarray(sqrt_s, dtype=float) ** 2
    return sigma_peak * s * gammaZ**2 / ((s - mZ**2) ** 2 + s**2 * gammaZ**2 / mZ**2)


def breit_wigner_relativistic_isr(sqrt_s, sigma_peak, mZ, gammaZ):
    return convolve_isr(breit_wigner_relativistic, sqrt_s, sigma_peak, mZ, gammaZ)


def iba_lineshape(sqrt_s: np.ndarray, sigma_peak: float, mZ: float, gammaZ: float) -> np.ndarray:
    """Improved-Born photon + Z + interference total cross section (README
    0.3) with sin2(theta_W_eff) fixed to PDG. Here `sigma_peak` is a
    dimensionless normalisation multiplying the absolute G_F-based
    prediction, so a fit returning sigma_peak ~ 1 is itself a check of the
    absolute normalisation of the computed cross sections."""
    sqrt_s = np.asarray(sqrt_s, dtype=float)
    f1, _ = iba_f1_f2(sqrt_s, SIN2_THETAW_EFF, mZ, gammaZ)
    s = sqrt_s**2
    return sigma_peak * (ccs.ALPHA_QED**2 / (4 * s)) * f1 * (16 * np.pi / 3) * GEV2_TO_NB


def iba_lineshape_isr(sqrt_s, sigma_peak, mZ, gammaZ):
    # the 1/s photon term makes x -> 1 diverge, so the s'/s cut is essential
    return convolve_isr(iba_lineshape, sqrt_s, sigma_peak, mZ, gammaZ, x_max=X_MAX_FROM_CUT)


LINESHAPE_MODELS = {
    "non-relativistic BW (baseline)": (breit_wigner_isr, dict(sigma_peak=2.0)),
    "relativistic running-width BW": (breit_wigner_relativistic_isr, dict(sigma_peak=2.0)),
    f"IBA gamma+Z+interf., s'/s > {SPRIME_OVER_S_MIN}": (iba_lineshape_isr, dict(sigma_peak=1.0)),
}


def compare_lineshape_models(clean: pd.DataFrame) -> pd.DataFrame:
    cov = ccs.build_covariance(clean, f_correlated=True)
    rows = []
    for name, (model, start) in LINESHAPE_MODELS.items():
        fit = fit_with_covariance(clean, model, cov, **start)
        rows.append(dict(model=name, chi2=fit.fval, ndof=len(clean) - 3,
                         mZ_GeV=fit.values["mZ"], mZ_err_GeV=fit.errors["mZ"],
                         gammaZ_GeV=fit.values["gammaZ"], gammaZ_err_GeV=fit.errors["gammaZ"],
                         norm=fit.values["sigma_peak"], norm_err=fit.errors["sigma_peak"],
                         mZ_pull_PDG=(fit.values["mZ"] - PDG_MZ_GEV) / fit.errors["mZ"],
                         gammaZ_pull_PDG=(fit.values["gammaZ"] - PDG_GAMMAZ_GEV) / fit.errors["gammaZ"]))
    return pd.DataFrame(rows)


# --- P1-4: joint multi-channel fit --------------------------------------------

JOINT_CHANNELS = ["qq", "mumu", "tautau"]
TRACK_CHANNELS = {"mumu", "tautau"}


def joint_dataset(seven_points: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for ch in JOINT_CHANNELS:
        t = ccs.compute_channel(seven_points, ch)
        if ch in TRACK_CHANNELS:
            t = t[t["diff_pct"].abs() <= ANOMALY_THRESHOLD_PCT]
        tables.append(t)
    return pd.concat(tables, ignore_index=True)


def joint_covariance(data: pd.DataFrame) -> np.ndarray:
    stat = data["sigma_stat_err_nb"].to_numpy()
    lr = data["sigma_lround_err_nb"].to_numpy()
    fs = data["sigma_fsyst_err_nb"].to_numpy()
    point = (data["year"].astype(str) + data["sample"]).to_numpy()
    chyear = (data["channel"] + data["year"].astype(str)).to_numpy()
    cov = np.diag(stat**2)
    cov += np.outer(lr, lr) * (point[:, None] == point[None, :])  # same L for every channel at a point
    cov += np.outer(fs, fs) * (chyear[:, None] == chyear[None, :])  # f shared within channel-year
    return cov


def joint_fit(data: pd.DataFrame, model=breit_wigner_isr, beam_spread: bool = False) -> Minuit:
    """beam_spread=True smears the model with the mean centre-of-mass energy
    spread of the fitted points (OPAL Table 8-11 rms, ~57 MeV); at the
    joint fit's few-MeV precision this is no longer negligible."""
    x = data["sqrt_s_GeV"].to_numpy()
    y = data["sigma_computed_nb"].to_numpy()
    cov_inv = np.linalg.inv(joint_covariance(data))
    if beam_spread:
        model = ccs.with_beam_spread(model, data["sqrt_s_rms_GeV"].mean())
    masks = [(data["channel"] == ch).to_numpy() for ch in JOINT_CHANNELS]
    names = [f"sigma_peak_{ch}" for ch in JOINT_CHANNELS] + ["mZ", "gammaZ"]

    def chi2(*params):
        peaks, mZ, gammaZ = params[:-2], params[-2], params[-1]
        pred = np.empty_like(y)
        for mask, peak in zip(masks, peaks):
            pred[mask] = model(x[mask], peak, mZ, gammaZ)
        r = y - pred
        return float(r @ cov_inv @ r)

    chi2.errordef = Minuit.LEAST_SQUARES
    starts = [float(data.loc[m, "sigma_computed_nb"].max()) * 1.3 for m in masks] + [91.19, 2.5]
    minuit = Minuit(chi2, *starts, name=names)
    minuit.migrad()
    minuit.hesse()
    return minuit


def print_joint_fit(fit: Minuit, n_points: int) -> None:
    print(f"chi2/ndof = {fit.fval:.2f}/{n_points - fit.npar}")
    for name in fit.parameters:
        unit = "GeV" if name in ("mZ", "gammaZ") else "nb"
        print(f"  {name:18s} = {fit.values[name]:.4f} +/- {fit.errors[name]:.4f} {unit}")
    pull_mZ = (fit.values["mZ"] - PDG_MZ_GEV) / fit.errors["mZ"]
    pull_gZ = (fit.values["gammaZ"] - PDG_GAMMAZ_GEV) / fit.errors["gammaZ"]
    print(f"  pull vs. PDG: mZ {pull_mZ:+.2f} sigma, gammaZ {pull_gZ:+.2f} sigma")
    # derived quantities with simple error propagation from the covariance
    cov = np.array(fit.covariance)
    idx = {n: i for i, n in enumerate(fit.parameters)}
    had, mu, tau = (fit.values[f"sigma_peak_{c}"] for c in JOINT_CHANNELS)
    lep = (mu + tau) / 2
    g = np.zeros(fit.npar)
    g[idx["sigma_peak_qq"]] = 1 / lep
    g[idx["sigma_peak_mumu"]] = -had / lep**2 / 2
    g[idx["sigma_peak_tautau"]] = -had / lep**2 / 2
    r_l, r_l_err = had / lep, np.sqrt(g @ cov @ g)
    ratio = mu / tau
    ratio_err = ratio * np.sqrt((fit.errors["sigma_peak_mumu"] / mu) ** 2 + (fit.errors["sigma_peak_tautau"] / tau) ** 2)
    print(f"  R_l = sigma_had / sigma_l (born, at peak) = {r_l:.2f} +/- {r_l_err:.2f}   (PDG R_l = {PDG_R_LEPTON})")
    print(f"  sigma_mumu / sigma_tautau at peak = {ratio:.4f} +/- {ratio_err:.4f}   (lepton universality: 1)")
    print(f"  sigma0_had = {had:.2f} +/- {fit.errors['sigma_peak_qq']:.2f} nb   (PDG {PDG_SIGMA0_HAD_NB})")


# --- plots --------------------------------------------------------------------


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def plot_joint(data: pd.DataFrame, fit: Minuit, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor=SURFACE)
    x = np.linspace(89.0, 93.5, 200)
    for ax, ch in zip(axes, JOINT_CHANNELS):
        d = data[data["channel"] == ch]
        _style(ax)
        ax.errorbar(d["sqrt_s_GeV"], d["sigma_computed_nb"], yerr=d["sigma_computed_err_nb"], fmt="o", color=BLUE,
                    markersize=7, markeredgewidth=0, elinewidth=2, capsize=3, zorder=3, label="computed from N, L, f")
        ax.plot(x, breit_wigner_isr(x, fit.values[f"sigma_peak_{ch}"], fit.values["mZ"], fit.values["gammaZ"]),
                color=AQUA, lw=2, zorder=1, label="joint ISR-convolved fit")
        ax.set_title(ch, color=INK, loc="left")
        ax.set_xlabel(r"$\sqrt{s}$ (GeV)", color=INK)
    axes[0].set_ylabel(r"$\sigma$ (nb)", color=INK)
    leg = axes[0].legend(frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.suptitle(f"Joint fit: $M_Z$ = {fit.values['mZ']:.4f} ± {fit.errors['mZ']:.4f} GeV, "
                 f"$\\Gamma_Z$ = {fit.values['gammaZ']:.4f} ± {fit.errors['gammaZ']:.4f} GeV", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def plot_extended(ext: pd.DataFrame, fit: Minuit, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    _style(ax)
    own, borrowed = ext[~ext["borrowed_f"]], ext[ext["borrowed_f"]]
    ax.errorbar(own["sqrt_s_GeV"], own["sigma_computed_nb"], yerr=own["sigma_computed_err_nb"], fmt="o", color=BLUE,
                markersize=8, markeredgewidth=0, elinewidth=2, capsize=3, zorder=3, label="own correction factor (7 points)")
    ax.errorbar(borrowed["sqrt_s_GeV"], borrowed["sigma_computed_nb"], yerr=borrowed["sigma_computed_err_nb"], fmt="D",
                color=ORANGE, markersize=7, markeredgewidth=0, elinewidth=2, capsize=3, zorder=3, label="borrowed f (6 points)")
    x = np.linspace(89.0, 93.5, 200)
    ax.plot(x, breit_wigner_isr(x, *fit.values), color=AQUA, lw=2, zorder=1, label="ISR-convolved fit, 12 points")
    ax.set_xlabel(r"$\sqrt{s}$ (GeV)", color=INK)
    ax.set_ylabel(r"$\sigma(e^+e^- \to \mu^+\mu^-)$ (nb)", color=INK)
    ax.set_title("Extended mu+mu- lineshape (1993-1995 near-peak points added)", color=INK, loc="left")
    leg = ax.legend(frameon=False, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


# --- main ---------------------------------------------------------------------


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 200)
    counts = load_counts()
    seven = select_corrected_points(counts)
    mumu = ccs.compute_channel(seven, "mumu")
    clean = mumu[mumu["diff_pct"].abs() <= ANOMALY_THRESHOLD_PCT]

    # ---- P1-1
    print("=== P1-1: mu+mu- with borrowed correction factors ===")
    ext = compute_channel_extended(select_extended_points(counts), "mumu")
    print(ext.to_string(index=False))
    ext.to_csv(DATA_PROCESSED / "mumu_cross_section_extended.csv", index=False)
    ext_clean = ext[~((ext["year"] == 1993) & (ext["sample"] == "peak"))]
    ext_no93 = ext_clean[~(ext_clean["borrowed_f"] & (ext_clean["year"] == 1993))]
    fits_ext = {
        "6 own-f points (baseline)": fit_breit_wigner_isr(clean),
        f"{len(ext_clean)} points incl. borrowed f": fit_breit_wigner_isr(ext_clean),
        f"{len(ext_no93)} points, borrowed f but no 1993 prescan": fit_breit_wigner_isr(ext_no93),
    }
    print("\nISR-convolved fits (diagonal errors):")
    for label, fit in fits_ext.items():
        print(f"  {label:48s} chi2/ndof {fit.fval:6.2f}/{fit.ndof:.0f}   mZ {fit.values['mZ']:.4f} +/- {fit.errors['mZ']:.4f}"
              f"   gammaZ {fit.values['gammaZ']:.4f} +/- {fit.errors['gammaZ']:.4f}")
    print("  The 1993 prescan points have no published cross section to validate against and\n"
          "  share the run period of the anomalous 1993 peak point, hence the third row.")
    plot_extended(ext_clean, fits_ext[f"{len(ext_clean)} points incl. borrowed f"], DATA_PROCESSED / "mumu_extended_lineshape.png")

    # ---- P1-3
    print("\n=== P1-3: lineshape model comparison on the 6 clean points (covariance fit) ===")
    models = compare_lineshape_models(clean)
    print(models.to_string(index=False))
    models.to_csv(DATA_PROCESSED / "lineshape_model_comparison.csv", index=False)
    base, rel = models.iloc[0], models.iloc[1]
    print(f"  running-width - non-relativistic: mZ {(rel['mZ_GeV'] - base['mZ_GeV']) * 1e3:+.1f} MeV, "
          f"gammaZ {(rel['gammaZ_GeV'] - base['gammaZ_GeV']) * 1e3:+.1f} MeV")
    iba = models.iloc[2]
    print(f"  IBA model normalisation = {iba['norm']:.4f} +/- {iba['norm_err']:.4f} (1 = absolute G_F prediction)")

    # ---- P1-4
    print("\n=== P1-4: joint qq + mumu + tautau fit, common mZ and gammaZ ===")
    data = joint_dataset(seven)
    print(data[["year", "sample", "channel", "sqrt_s_GeV", "sigma_computed_nb", "sigma_computed_err_nb", "diff_pct"]].to_string(index=False))
    fit_joint = joint_fit(data)
    print("\n-- without beam energy spread --")
    print_joint_fit(fit_joint, len(data))
    fit_joint_spread = joint_fit(data, beam_spread=True)
    print(f"\n-- with Gaussian beam energy spread (rms {data['sqrt_s_rms_GeV'].mean() * 1e3:.0f} MeV) --")
    print_joint_fit(fit_joint_spread, len(data))
    print(
        "  At this precision the fit error is smaller than the model systematics of the\n"
        "  error budget (radiator form, beam spread, propagator convention), so the pulls\n"
        "  vs. PDG measure the model, not the data. See README."
    )
    plot_joint(data, fit_joint, DATA_PROCESSED / "joint_fit.png")
    pd.DataFrame({
        "parameter": list(fit_joint.parameters),
        "value": list(fit_joint.values), "error": list(fit_joint.errors),
        "value_beam_spread": list(fit_joint_spread.values), "error_beam_spread": list(fit_joint_spread.errors),
    }).to_csv(DATA_PROCESSED / "joint_fit.csv", index=False)


if __name__ == "__main__":
    main()
