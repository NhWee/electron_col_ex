"""
Three extensions of the mu+mu- lineshape measurement in
`compute_cross_section.py`, each answering one item of ROADMAP.md. All fits
use the baseline's radiator, beam-energy spread and per-channel energies.

P1-1  More energy points. OPAL only tabulates correction factors f for 7
      of the 29 rows of Table 1, but the mu+mu- f barely depends on energy
      (1993: 1.0922 / 1.0920 / 1.0927 across peak-2 / peak / peak+2). The
      near-peak 1993-1995 rows without their own f (prescan, 1994 peak(c)
      and peak(d)) are given the same year's peak f, with half the spread
      of that year's f values added as a "borrowed f" systematic.

P1-3  LEP-convention lineshape models. The relativistic running-width
      Breit-Wigner (the propagator LEP's M_Z is defined with), and the full
      improved-Born photon + Z + interference cross section with the ISR
      integral cut at the paper's acceptance (m_ff^2/s > 0.01, Table 2).
      The improved-Born model uses a fixed-width propagator; its M_Z and
      Gamma_Z are also converted to the running-width convention.

P1-4  Joint fit of qq, mu+mu- and tau+tau- with common M_Z and Gamma_Z and
      one free peak cross section per channel, using a covariance in which
      the luminosity rounding is shared by all channels at a point and the
      f systematic is shared within a (channel, year). e+e- is left out:
      its t-channel contribution is not a Breit-Wigner. The 1993 peak point
      is dropped for the two track-based channels only (README section 1).
      Done twice: with the Breit-Wigner, and with the improved-Born
      photon + Z + interference cross section of each channel (quark
      couplings for qq), converted to the running-width convention.

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
    PDG_GAMMAZ_ERR_GEV,
    PDG_GAMMAZ_GEV,
    PDG_MZ_ERR_GEV,
    PDG_MZ_GEV,
    SURFACE,
    breit_wigner_isr,
    breit_wigner_relativistic_isr,
    compute_cross_section,
    fit_breit_wigner_isr,
    fit_with_covariance,
    improved_born,
    improved_born_isr,
    load_counts,
    select_corrected_points,
    with_beam_spread,
)
from electroweak import fixed_to_running_width, z_term_cross_section_nb

# PDG 2024 for comparison only
PDG_R_LEPTON = 20.767
PDG_SIGMA0_HAD_NB = 41.481



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
        published[["year", "sample", "sqrt_s_mean_GeV", "sqrt_s_rms_GeV", "xsec_measured_nb"]],
        on=["year", "sample"], how="left",
    )
    # Same energies as compute_channel(): the channel's own mean sqrt(s)
    # where OPAL publishes one. The 1993 and 1995 prescans have no published
    # cross section, so they keep Table 1's 10 MeV-rounded sqrt(s) and get
    # their year's mean beam spread (all 1993-1995 spreads lie within
    # 54.6-56.7 MeV, paper section 4).
    out["sqrt_s_table1_GeV"] = out["sqrt_s_GeV"]
    out["sqrt_s_GeV"] = out["sqrt_s_mean_GeV"].fillna(out["sqrt_s_GeV"])
    out["sqrt_s_rms_GeV"] = out["sqrt_s_rms_GeV"].fillna(out.groupby("year")["sqrt_s_rms_GeV"].transform("mean"))
    out["diff_pct"] = (out["sigma_computed_nb"] - out["xsec_measured_nb"]) / out["xsec_measured_nb"] * 100
    out.insert(2, "channel", channel)
    cols = ["year", "sample", "channel", "sqrt_s_GeV", "sqrt_s_table1_GeV", "sqrt_s_rms_GeV", spec["n_col"], "L_pbinv",
            "borrowed_f", "sigma_computed_nb", *ERROR_COLUMNS, "sigma_borrow_err_nb", "xsec_measured_nb", "diff_pct"]
    return out[cols].sort_values(["year", "sqrt_s_GeV"]).reset_index(drop=True)


# --- P1-3: LEP-convention lineshape models ------------------------------------


# Improved-Born mu+mu- cross section (photon + Z + interference, fixed width,
# alpha(M_Z^2) photon, sin2(theta_W_eff) fixed to PDG), see electroweak.py.
# Its normalisation parameter is 1 for the absolute G_F-based prediction, so
# a fit returning ~1 checks the absolute normalisation of the computed cross
# sections. The ISR integral stops at the paper's acceptance, x < 0.99.
iba_lineshape = improved_born("mumu")
iba_lineshape_isr = improved_born_isr("mumu")


LINESHAPE_MODELS = {
    # name: (ISR-convolved model, start values, propagator convention)
    "non-relativistic BW (baseline)": (breit_wigner_isr, dict(sigma_peak=2.0), "non-relativistic"),
    "relativistic running-width BW": (breit_wigner_relativistic_isr, dict(sigma_peak=2.0), "running"),
    "improved Born gamma+Z+interference": (iba_lineshape_isr, dict(sigma_peak=1.0), "fixed"),
}


def compare_lineshape_models(clean: pd.DataFrame) -> pd.DataFrame:
    cov = ccs.build_covariance(clean, f_correlated=True)
    rms = clean["sqrt_s_rms_GeV"].to_numpy()
    rows = []
    for name, (model, start, convention) in LINESHAPE_MODELS.items():
        fit = fit_with_covariance(clean, with_beam_spread(model, rms), cov, **start)
        mZ, gZ = fit.values["mZ"], fit.values["gammaZ"]
        if convention == "running":
            mZ_run, gZ_run = mZ, gZ
        elif convention == "fixed":
            mZ_run, gZ_run = fixed_to_running_width(mZ, gZ)
        else:
            mZ_run, gZ_run = np.nan, np.nan
        rows.append(dict(model=name, convention=convention, chi2=fit.fval, ndof=len(clean) - 3, valid=fit.valid,
                         mZ_GeV=mZ, mZ_err_GeV=fit.errors["mZ"], gammaZ_GeV=gZ, gammaZ_err_GeV=fit.errors["gammaZ"],
                         norm=fit.values["sigma_peak"], norm_err=fit.errors["sigma_peak"],
                         mZ_running_GeV=mZ_run, gammaZ_running_GeV=gZ_run,
                         mZ_running_pull_PDG=(mZ_run - PDG_MZ_GEV) / np.hypot(fit.errors["mZ"], PDG_MZ_ERR_GEV),
                         gammaZ_running_pull_PDG=(gZ_run - PDG_GAMMAZ_GEV) / np.hypot(fit.errors["gammaZ"], PDG_GAMMAZ_ERR_GEV)))
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


def joint_fit(data: pd.DataFrame, model=breit_wigner_isr, beam_spread: bool = True,
              prefix: str = "sigma_peak", start_norms: dict | None = None) -> Minuit:
    """`model` is one sigma(sqrt_s, norm, mZ, gammaZ) for all channels, or a
    dict {channel: model}. beam_spread=True (the default, as in the
    baseline) smears each point's model with its own centre-of-mass energy
    spread (OPAL Tables 8-11, ~57 MeV); at the joint fit's few-MeV
    precision it matters. The per-channel normalisations are called
    f"{prefix}_{channel}"; start_norms overrides their starting values."""
    x = data["sqrt_s_GeV"].to_numpy()
    y = data["sigma_computed_nb"].to_numpy()
    cov_inv = np.linalg.inv(joint_covariance(data))
    masks = [(data["channel"] == ch).to_numpy() for ch in JOINT_CHANNELS]
    per_channel = [model[ch] if isinstance(model, dict) else model for ch in JOINT_CHANNELS]
    if beam_spread:
        rms = data["sqrt_s_rms_GeV"].to_numpy()
        models = [with_beam_spread(m, rms[mask]) for m, mask in zip(per_channel, masks)]
    else:
        models = per_channel
    names = [f"{prefix}_{ch}" for ch in JOINT_CHANNELS] + ["mZ", "gammaZ"]

    def chi2(*params):
        peaks, mZ, gammaZ = params[:-2], params[-2], params[-1]
        pred = np.empty_like(y)
        for mask, peak, channel_model in zip(masks, peaks, models):
            pred[mask] = channel_model(x[mask], peak, mZ, gammaZ)
        r = y - pred
        return float(r @ cov_inv @ r)

    chi2.errordef = Minuit.LEAST_SQUARES
    if start_norms is None:
        starts = [float(data.loc[m, "sigma_computed_nb"].max()) * 1.3 for m in masks]
    else:
        starts = [start_norms[ch] for ch in JOINT_CHANNELS]
    starts += [91.19, 2.5]
    minuit = Minuit(chi2, *starts, name=names)
    minuit.migrad()
    minuit.hesse()
    return minuit


def print_joint_fit(fit: Minuit, n_points: int) -> None:
    print(f"chi2/ndof = {fit.fval:.2f}/{n_points - fit.npar}   (Migrad valid: {fit.valid})")
    for name in fit.parameters:
        unit = "GeV" if name in ("mZ", "gammaZ") else "nb"
        print(f"  {name:18s} = {fit.values[name]:.4f} +/- {fit.errors[name]:.4f} {unit}")
    pull_mZ = (fit.values["mZ"] - PDG_MZ_GEV) / fit.errors["mZ"]
    pull_gZ = (fit.values["gammaZ"] - PDG_GAMMAZ_GEV) / fit.errors["gammaZ"]
    print(f"  pull vs. PDG (fit error only): mZ {pull_mZ:+.2f} sigma, gammaZ {pull_gZ:+.2f} sigma")
    print(f"  correlation(mZ, gammaZ) = {fit.covariance.correlation()['mZ', 'gammaZ']:+.2f}")
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


def _propagate(fit: Minuit, function, step: float = 1e-6) -> tuple[float, float]:
    """value and error of function(values dict), from the fit covariance by
    numerical gradient."""
    names = list(fit.parameters)
    values = dict(zip(names, fit.values))
    central = function(values)
    grad = np.zeros(len(names))
    for i, name in enumerate(names):
        shifted = dict(values)
        h = step * max(abs(values[name]), 1.0)
        shifted[name] += h
        grad[i] = (function(shifted) - central) / h
    return central, float(np.sqrt(grad @ np.array(fit.covariance) @ grad))


def joint_pole_quantities(fit: Minuit) -> dict:
    """sigma0_had (Z term of the qq model at the pole, times its fitted
    normalisation) and R_l = Gamma_had / Gamma_ll (the ratio of the qq and
    mean-lepton Z terms at the pole), with errors from the full covariance.
    Both lean on the Standard Model couplings with one sin2(theta_W_eff) to
    split each channel's cross section into Z, photon and interference."""

    def sigma0_had(v):
        return v["norm_qq"] * float(z_term_cross_section_nb(v["mZ"], v["mZ"], v["gammaZ"], "qq"))

    def r_lepton(v):
        lepton = (v["norm_mumu"] + v["norm_tautau"]) / 2 * float(z_term_cross_section_nb(v["mZ"], v["mZ"], v["gammaZ"], "mumu"))
        return sigma0_had(v) / lepton

    s0, s0_err = _propagate(fit, sigma0_had)
    r, r_err = _propagate(fit, r_lepton)
    return dict(sigma0_had=s0, sigma0_had_err=s0_err, r_lepton=r, r_lepton_err=r_err)


def print_joint_improved_born(fit: Minuit, n_points: int) -> tuple[float, float]:
    """Print the improved-Born joint fit; returns (M_Z, Gamma_Z) converted
    to the running-width convention."""
    print(f"chi2/ndof = {fit.fval:.2f}/{n_points - fit.npar}   (Migrad valid: {fit.valid})")
    for name in fit.parameters:
        unit = "GeV" if name in ("mZ", "gammaZ") else ""
        print(f"  {name:18s} = {fit.values[name]:.4f} +/- {fit.errors[name]:.4f} {unit}")
    mz_run, gz_run = fixed_to_running_width(fit.values["mZ"], fit.values["gammaZ"])
    pull_m = (mz_run - PDG_MZ_GEV) / np.hypot(fit.errors["mZ"], PDG_MZ_ERR_GEV)
    pull_g = (gz_run - PDG_GAMMAZ_GEV) / np.hypot(fit.errors["gammaZ"], PDG_GAMMAZ_ERR_GEV)
    print(f"  running-width convention: mZ = {mz_run:.4f} GeV, gammaZ = {gz_run:.4f} GeV")
    print(f"  pull vs. PDG (fit + PDG errors): mZ {pull_m:+.2f} sigma, gammaZ {pull_g:+.2f} sigma")
    lep = (fit.values["norm_mumu"] + fit.values["norm_tautau"]) / 2
    print(f"  norm_mumu / norm_tautau = {fit.values['norm_mumu'] / fit.values['norm_tautau']:.4f}   (lepton universality: 1)")
    print(f"  norm_qq / (3 x mean lepton norm) = {fit.values['norm_qq'] / (3 * lep):.4f}   "
          "(colour-stripped hadronic factor, ~1 + alpha_s/pi ~ 1.04 expected)")
    pole = joint_pole_quantities(fit)
    print(f"  sigma0_had = {pole['sigma0_had']:.2f} +/- {pole['sigma0_had_err']:.2f} nb   (PDG {PDG_SIGMA0_HAD_NB})")
    print(f"  R_l = Gamma_had / Gamma_ll = {pole['r_lepton']:.2f} +/- {pole['r_lepton_err']:.2f}   (PDG R_l = {PDG_R_LEPTON})")
    return mz_run, gz_run


# --- plots --------------------------------------------------------------------


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def plot_joint(data: pd.DataFrame, fit: Minuit, out_path: Path, model=breit_wigner_isr,
               prefix: str = "sigma_peak", title: str | None = None) -> None:
    """`model` and `prefix` as in joint_fit()."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor=SURFACE)
    x = np.linspace(89.0, 93.5, 200)
    rms = data["sqrt_s_rms_GeV"].mean()
    for ax, ch in zip(axes, JOINT_CHANNELS):
        channel_model = with_beam_spread(model[ch] if isinstance(model, dict) else model, rms)
        d = data[data["channel"] == ch]
        _style(ax)
        ax.errorbar(d["sqrt_s_GeV"], d["sigma_computed_nb"], yerr=d["sigma_computed_err_nb"], fmt="o", color=BLUE,
                    markersize=7, markeredgewidth=0, elinewidth=2, capsize=3, zorder=3, label="computed from N, L, f")
        ax.plot(x, channel_model(x, fit.values[f"{prefix}_{ch}"], fit.values["mZ"], fit.values["gammaZ"]),
                color=AQUA, lw=2, zorder=1, label="joint ISR-convolved fit")
        ax.set_title(ch, color=INK, loc="left")
        ax.set_xlabel(r"$\sqrt{s}$ (GeV)", color=INK)
    axes[0].set_ylabel(r"$\sigma$ (nb)", color=INK)
    leg = axes[0].legend(frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK)
    if title is None:
        title = (f"Joint fit: $M_Z$ = {fit.values['mZ']:.4f} ± {fit.errors['mZ']:.4f} GeV, "
                 f"$\\Gamma_Z$ = {fit.values['gammaZ']:.4f} ± {fit.errors['gammaZ']:.4f} GeV")
    fig.suptitle(title, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def plot_extended(ext: pd.DataFrame, fit: Minuit, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    _style(ax)
    own, borrowed = ext[~ext["borrowed_f"]], ext[ext["borrowed_f"]]
    ax.errorbar(own["sqrt_s_GeV"], own["sigma_computed_nb"], yerr=own["sigma_computed_err_nb"], fmt="o", color=BLUE,
                markersize=8, markeredgewidth=0, elinewidth=2, capsize=3, zorder=3,
                label=f"own correction factor ({len(own)} points)")
    ax.errorbar(borrowed["sqrt_s_GeV"], borrowed["sigma_computed_nb"], yerr=borrowed["sigma_computed_err_nb"], fmt="D",
                color=ORANGE, markersize=7, markeredgewidth=0, elinewidth=2, capsize=3, zorder=3,
                label=f"borrowed f ({len(borrowed)} points)")
    x = np.linspace(89.0, 93.5, 200)
    model = with_beam_spread(breit_wigner_isr, ext["sqrt_s_rms_GeV"].mean())
    ax.plot(x, model(x, *fit.values), color=AQUA, lw=2, zorder=1, label=f"ISR-convolved fit, {len(ext)} points")
    ax.set_xlabel(r"$\sqrt{s}$ (GeV)", color=INK)
    ax.set_ylabel(r"$\sigma(e^+e^- \to \mu^+\mu^-)$ (nb)", color=INK)
    ax.set_title("Extended mu+mu- lineshape (1993-1995 near-peak points added)", color=INK, loc="left")
    leg = ax.legend(frameon=False, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


# --- main ---------------------------------------------------------------------


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 250)
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
        "6 own-f points (baseline)": fit_breit_wigner_isr(clean, beam_spread=True),
        f"{len(ext_clean)} points incl. borrowed f": fit_breit_wigner_isr(ext_clean, beam_spread=True),
        f"{len(ext_no93)} points, borrowed f but no 1993 prescan": fit_breit_wigner_isr(ext_no93, beam_spread=True),
    }
    print("\nISR-convolved fits (diagonal errors, beam spread included):")
    for label, fit in fits_ext.items():
        print(f"  {label:48s} chi2/ndof {fit.fval:6.2f}/{fit.ndof:.0f}   mZ {fit.values['mZ']:.4f} +/- {fit.errors['mZ']:.4f}"
              f"   gammaZ {fit.values['gammaZ']:.4f} +/- {fit.errors['gammaZ']:.4f}")
    print("  The 1993 prescan points have no published cross section to validate against and\n"
          "  share the run period of the flagged 1993 peak point, hence the third row.")
    plot_extended(ext_clean, fits_ext[f"{len(ext_clean)} points incl. borrowed f"], DATA_PROCESSED / "mumu_extended_lineshape.png")

    # ---- P1-3
    print(f"\n=== P1-3: lineshape model comparison on the 6 clean points "
          f"(covariance fit, s'/s > {ccs.SPRIME_OVER_S_MIN} acceptance) ===")
    models = compare_lineshape_models(clean)
    print(models.to_string(index=False))
    models.to_csv(DATA_PROCESSED / "lineshape_model_comparison.csv", index=False)
    base, rel, iba = models.iloc[0], models.iloc[1], models.iloc[2]
    print(f"  running-width - non-relativistic: mZ {(rel['mZ_GeV'] - base['mZ_GeV']) * 1e3:+.1f} MeV, "
          f"gammaZ {(rel['gammaZ_GeV'] - base['gammaZ_GeV']) * 1e3:+.1f} MeV")
    print(f"  improved Born, fixed width:     mZ {iba['mZ_GeV']:.4f}, gammaZ {iba['gammaZ_GeV']:.4f} GeV")
    print(f"  ... in the running-width convention: mZ {iba['mZ_running_GeV']:.4f}, gammaZ {iba['gammaZ_running_GeV']:.4f} GeV "
          f"({iba['mZ_running_pull_PDG']:+.2f} / {iba['gammaZ_running_pull_PDG']:+.2f} sigma from PDG, fit error)")
    print(f"  photon exchange + interference, both in the running-width convention: "
          f"mZ {(iba['mZ_running_GeV'] - rel['mZ_GeV']) * 1e3:+.1f} MeV, gammaZ {(iba['gammaZ_running_GeV'] - rel['gammaZ_GeV']) * 1e3:+.1f} MeV")
    print(f"  improved-Born normalisation = {iba['norm']:.4f} +/- {iba['norm_err']:.4f} (1 = absolute G_F prediction)")

    # ---- P1-4
    print("\n=== P1-4: joint qq + mumu + tautau fit, common mZ and gammaZ ===")
    data = joint_dataset(seven)
    print(data[["year", "sample", "channel", "sqrt_s_GeV", "sigma_computed_nb", "sigma_computed_err_nb", "diff_pct"]].to_string(index=False))
    fit_joint = joint_fit(data)
    print(f"\n-- with the per-point beam energy spread (rms {data['sqrt_s_rms_GeV'].mean() * 1e3:.0f} MeV on average), as the baseline --")
    print_joint_fit(fit_joint, len(data))
    fit_joint_nospread = joint_fit(data, beam_spread=False)
    print("\n-- without beam energy spread, for comparison --")
    print_joint_fit(fit_joint_nospread, len(data))
    fit_joint_running = joint_fit(data, model=breit_wigner_relativistic_isr)
    print("\n-- running-width Breit-Wigner (LEP convention), with beam spread --")
    print_joint_fit(fit_joint_running, len(data))
    fit_joint_iba = joint_fit(data, model={ch: improved_born_isr(ch) for ch in JOINT_CHANNELS}, prefix="norm",
                              start_norms={"qq": 3.1, "mumu": 1.0, "tautau": 1.0})
    print("\n-- improved Born: photon + Z + interference per channel, with beam spread --")
    mz_iba, gz_iba = print_joint_improved_born(fit_joint_iba, len(data))
    print(f"  photon exchange + interference vs. the running-width BW: mZ {(mz_iba - fit_joint_running.values['mZ']) * 1e3:+.1f} MeV, "
          f"gammaZ {(gz_iba - fit_joint_running.values['gammaZ']) * 1e3:+.1f} MeV")
    lower = ccs.LOWER_ORDER[ccs.ISR_RADIATOR]
    with ccs._radiator(lower):
        fit_joint_iba_lower = joint_fit(data, model={ch: improved_born_isr(ch) for ch in JOINT_CHANNELS}, prefix="norm",
                                        start_norms={"qq": 3.1, "mumu": 1.0, "tautau": 1.0})
    mz_low, gz_low = fixed_to_running_width(fit_joint_iba_lower.values["mZ"], fit_joint_iba_lower.values["gammaZ"])
    print(f"  radiator truncation ({ccs.ISR_RADIATOR} - {lower}) for this fit: mZ {(mz_iba - mz_low) * 1e3:+.1f} MeV, "
          f"gammaZ {(gz_iba - gz_low) * 1e3:+.1f} MeV")
    print(
        "  The Breit-Wigner fits above leave out the photon-exchange term, which is\n"
        "  what most of their pull vs. PDG on Gamma_Z is. Their pulls use the fit error\n"
        "  only, which is smaller than the systematics of the error budget. See README."
    )
    plot_joint(data, fit_joint_iba, DATA_PROCESSED / "joint_fit.png",
               model={ch: improved_born_isr(ch) for ch in JOINT_CHANNELS}, prefix="norm",
               title=(r"Joint $\gamma/Z$ fit (improved Born, ISR, beam spread), running-width convention: "
                      f"$M_Z$ = {mz_iba:.4f} ± {fit_joint_iba.errors['mZ']:.4f} GeV, "
                      f"$\\Gamma_Z$ = {gz_iba:.4f} ± {fit_joint_iba.errors['gammaZ']:.4f} GeV"))
    pd.DataFrame({
        "parameter": list(fit_joint.parameters),
        "value": list(fit_joint.values), "error": list(fit_joint.errors),
        "value_no_beam_spread": list(fit_joint_nospread.values), "error_no_beam_spread": list(fit_joint_nospread.errors),
        "value_running_width": list(fit_joint_running.values), "error_running_width": list(fit_joint_running.errors),
        "value_improved_born": list(fit_joint_iba.values), "error_improved_born": list(fit_joint_iba.errors),
    }).to_csv(DATA_PROCESSED / "joint_fit.csv", index=False)


if __name__ == "__main__":
    main()
