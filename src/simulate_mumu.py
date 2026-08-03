"""
Simulation-based closure test for the e+e- -> mu+mu- extraction pipeline.

`compute_cross_section.py` computes sigma = f*N/L from OPAL's own published
N, L and f tables -- real arithmetic and real error propagation, but every
input number was already OPAL's own analysis output. This script instead
generates a toy Monte Carlo sample from a *known* truth model (photon+Z
resonance, convolved with the same leading-log ISR radiator already
implemented, with a physically motivated peak cross section derived from the
standard resonance formula rather than an arbitrary guess), reconstructs
sigma(sqrt_s) with purely statistical error bars from the simulated event
counts, and reuses the exact same fit machinery (`fit_breit_wigner`,
`fit_breit_wigner_isr`) to check whether it recovers the exact input truth --
a real closure test of the extraction pipeline, not just "close to PDG."

The detector acceptance correction mirrors OPAL's own `f` factor: it is
derived once from a dedicated large simulation sample (never from the
per-point data itself), exactly like OPAL's published correction tables --
except here we compute it ourselves instead of reading it from a paper.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares
from uncertainties import ufloat

from compute_cross_section import (
    ALPHA_QED,
    AQUA,
    BLUE,
    DATA_PROCESSED,
    GRIDLINE,
    INK,
    MUTED,
    ORANGE,
    PDG_GAMMAZ_GEV,
    PDG_MZ_GEV,
    SURFACE,
    breit_wigner,
    breit_wigner_isr,
    fit_breit_wigner,
    fit_breit_wigner_isr,
    print_fit_result,
)

YELLOW = "#eda100"

RNG = np.random.default_rng(20260801)

# --- Truth model -------------------------------------------------------
# Generate assuming the PDG values are exactly true, so the fit results
# below can be compared to an exact known answer instead of an external
# reference with its own uncertainty.
TRUE_MZ_GEV = PDG_MZ_GEV
TRUE_GAMMAZ_GEV = PDG_GAMMAZ_GEV

# Peak cross section from the standard resonance formula
# sigma_peak = (12*pi/M_Z^2) * Gamma_ee*Gamma_mumu / Gamma_Z^2, using PDG's
# leptonic partial width (~84 MeV, lepton universality) -- not an arbitrary
# number, and it should land close to the ~2.0 nb LEP actually measured.
GEV2_TO_NB = 389379.0  # (hbar*c)^2, CODATA, converted from mb to nb
PDG_GAMMA_LL_GEV = 0.08399
TRUE_SIGMA_PEAK_NB = (
    12 * np.pi / TRUE_MZ_GEV**2 * PDG_GAMMA_LL_GEV**2 / TRUE_GAMMAZ_GEV**2 * GEV2_TO_NB
)

# --- Angular distribution / gamma-Z interference -------------------------
# Standard "improved Born approximation" (IBA) formula (e.g. Renton,
# Electroweak Interactions): dsigma/dOmega ~ (1+cos^2 theta)*F1(s) +
# cos(theta)*F2(s), with F1, F2 built from the photon/Z propagator ratio
# chi(s) and standard lepton vector/axial couplings. A_FB(s) = (3/8)*F2/F1
# is the standard result of integrating forward vs. backward hemispheres.
# This captures the real gamma/Z interference (sign flip below/above the
# peak) that the constant-A_FB approximation could not -- but note it is a
# *different* level of approximation than the non-relativistic fixed-width
# breit_wigner() used for the rate/lineshape side of this script (that one
# ignores the photon propagator and interference entirely). Two different
# approximations for two different observables, not a bug.
SIN2_THETAW_EFF = 0.23155
V_LEPTON = -0.5 + 2 * SIN2_THETAW_EFF
A_LEPTON = -0.5
Q_LEPTON = -1
G_FERMI_GEV2 = 1.16637e-5


def chi(sqrt_s: np.ndarray, mZ: float, gammaZ: float) -> np.ndarray:
    s = np.asarray(sqrt_s, dtype=float) ** 2
    kappa0 = G_FERMI_GEV2 * mZ**2 / (2 * np.sqrt(2) * np.pi * ALPHA_QED)
    return kappa0 * s / (s - mZ**2 + 1j * mZ * gammaZ)


def iba_f1_f2(sqrt_s: np.ndarray, sin2thetaw: float, mZ: float, gammaZ: float) -> tuple[np.ndarray, np.ndarray]:
    v = -0.5 + 2 * sin2thetaw
    a = -0.5
    c = chi(sqrt_s, mZ, gammaZ)
    f1 = Q_LEPTON**4 - 2 * Q_LEPTON**2 * v * v * np.real(c) + (v**2 + a**2) ** 2 * np.abs(c) ** 2
    f2 = -4 * Q_LEPTON**2 * a * a * np.real(c) + 8 * (v * a) ** 2 * np.abs(c) ** 2
    return f1, f2


def a_fb_true(
    sqrt_s: np.ndarray,
    sin2thetaw: float = SIN2_THETAW_EFF,
    mZ: float = TRUE_MZ_GEV,
    gammaZ: float = TRUE_GAMMAZ_GEV,
) -> np.ndarray:
    f1, f2 = iba_f1_f2(sqrt_s, sin2thetaw, mZ, gammaZ)
    return 0.375 * f2 / f1


def gamma_ee_from_couplings(sin2thetaw: float = SIN2_THETAW_EFF, mZ: float = TRUE_MZ_GEV) -> float:
    """Sanity check on the coupling convention and G_F usage: compare
    against PDG's ~84 MeV leptonic partial width, independent of anything
    else in this script."""
    v = -0.5 + 2 * sin2thetaw
    a = -0.5
    return G_FERMI_GEV2 * mZ**3 / (6 * np.sqrt(2) * np.pi) * (v**2 + a**2)


def sigma_peak_from_iba(sin2thetaw: float = SIN2_THETAW_EFF, mZ: float = TRUE_MZ_GEV, gammaZ: float = TRUE_GAMMAZ_GEV) -> float:
    """Second, independent sanity check: integrate F1(s) at s=mZ^2 over
    solid angle and compare to TRUE_SIGMA_PEAK_NB (derived from the
    unrelated 12*pi*Gamma_ee*Gamma_mumu/(mZ^2*Gamma_Z^2) formula). Two
    different formulas agreeing is evidence the IBA normalization is right."""
    f1, _ = iba_f1_f2(mZ, sin2thetaw, mZ, gammaZ)
    s = mZ**2
    sigma_gev2 = (ALPHA_QED**2 / (4 * s)) * f1 * (16 * np.pi / 3)
    return sigma_gev2 * GEV2_TO_NB


# --- Scan + toy detector ---------------------------------------------------
SCAN_POINTS_GEV = np.arange(TRUE_MZ_GEV - 3.0, TRUE_MZ_GEV + 3.01, 0.5)
LUMI_PBINV = 2.0  # toy per-point luminosity, not tied to a real run
ACCEPTANCE_COSTHETA_MAX = 0.95
EFFICIENCY_SAMPLE_SIZE = 500_000


def angular_pdf(costheta: np.ndarray, a_fb: float) -> np.ndarray:
    return 0.375 * ((1 + costheta**2) + (8 / 3) * a_fb * costheta)


def sample_costheta(n: int, a_fb: float, rng: np.random.Generator) -> np.ndarray:
    """Rejection sampling of angular_pdf on [-1, 1]; peak density is at
    costheta = +-1, giving an envelope of 0.375*(2 + (8/3)*|a_fb|)."""
    envelope = 0.375 * (2 + (8 / 3) * abs(a_fb))
    accepted = np.empty(0)
    while accepted.size < n:
        batch = n - accepted.size
        candidates = rng.uniform(-1, 1, size=batch * 2)
        u = rng.uniform(0, envelope, size=batch * 2)
        accepted = np.concatenate([accepted, candidates[u < angular_pdf(candidates, a_fb)]])
    return accepted[:n]


def measure_acceptance(rng: np.random.Generator) -> tuple[float, float]:
    """A_hat and its binomial statistical error, from a dedicated simulation
    sample -- plays exactly the role OPAL's f correction factor plays in
    compute_cross_section.py, except derived here rather than read from a
    paper. a_fb=0 here: the |cos(theta)| cut is symmetric, so A_hat does not
    depend on A_FB (proven and documented in the README)."""
    costheta = sample_costheta(EFFICIENCY_SAMPLE_SIZE, 0.0, rng)
    passed = np.abs(costheta) < ACCEPTANCE_COSTHETA_MAX
    a_hat = passed.mean()
    a_hat_err = np.sqrt(a_hat * (1 - a_hat) / EFFICIENCY_SAMPLE_SIZE)
    return a_hat, a_hat_err


def generate_scan_point(sqrt_s: float, a_hat: float, a_hat_err: float, rng: np.random.Generator) -> dict:
    """Draw a toy event sample at one nominal scan energy and reconstruct
    sigma purely from N_selected -- N_gen (the true production count) is
    used only to build the toy universe and is never looked at again after
    this point, matching how a real analysis never observes it either.

    Also splits the accepted events into forward/backward counts to measure
    A_FB(sqrt_s) -- this rides along on exactly the same simulated sample,
    it doesn't change anything about how sigma is computed above (the
    |cos(theta)| cut is symmetric, so A_FB has zero effect on N_selected)."""
    sigma_true_isr = breit_wigner_isr(sqrt_s, TRUE_SIGMA_PEAK_NB, TRUE_MZ_GEV, TRUE_GAMMAZ_GEV)[0]
    n_gen = rng.poisson(LUMI_PBINV * sigma_true_isr * 1000)  # nb -> pb

    a_fb = a_fb_true(sqrt_s)
    costheta = sample_costheta(int(n_gen), a_fb, rng)
    accepted = costheta[np.abs(costheta) < ACCEPTANCE_COSTHETA_MAX]
    n_selected = accepted.size
    n_forward = int(np.sum(accepted > 0))
    n_backward = n_selected - n_forward

    f_corr = ufloat(1 / a_hat, a_hat_err / a_hat**2)
    sigma = f_corr * ufloat(n_selected, np.sqrt(n_selected)) / LUMI_PBINV / 1000  # pb -> nb

    afb_measured = (n_forward - n_backward) / n_selected
    afb_err = np.sqrt(max(1 - afb_measured**2, 0.0) / n_selected)

    return {
        "sqrt_s_GeV": sqrt_s,
        "n_generated": int(n_gen),
        "n_selected": n_selected,
        "sigma_computed_nb": sigma.nominal_value,
        "sigma_computed_err_nb": sigma.std_dev,
        "n_forward": n_forward,
        "n_backward": n_backward,
        "afb_measured": afb_measured,
        "afb_err": afb_err,
        "afb_true": a_fb,
    }


def run_scan(rng: np.random.Generator) -> pd.DataFrame:
    a_hat, a_hat_err = measure_acceptance(rng)
    print(
        f"Acceptance A_hat = {a_hat:.4f} +/- {a_hat_err:.4f} "
        f"(from a {EFFICIENCY_SAMPLE_SIZE}-event dedicated simulation sample, "
        f"|cos(theta)| < {ACCEPTANCE_COSTHETA_MAX} cut)"
    )
    rows = [generate_scan_point(sqrt_s, a_hat, a_hat_err, rng) for sqrt_s in SCAN_POINTS_GEV]
    return pd.DataFrame(rows)


def fit_afb(scan: pd.DataFrame, mZ: float, gammaZ: float) -> Minuit:
    """Second fit, mirroring the real two-step LEP procedure: mZ and
    gammaZ come in fixed from the lineshape fit (fit_breit_wigner_isr), and
    sin2(theta_W_eff) is the one free parameter extracted from the measured
    A_FB(s) points."""

    def model(sqrt_s: np.ndarray, sin2thetaw: float) -> np.ndarray:
        return a_fb_true(sqrt_s, sin2thetaw=sin2thetaw, mZ=mZ, gammaZ=gammaZ)

    least_squares = LeastSquares(
        scan["sqrt_s_GeV"].to_numpy(),
        scan["afb_measured"].to_numpy(),
        scan["afb_err"].to_numpy(),
        model,
    )
    minuit = Minuit(least_squares, sin2thetaw=0.25)
    minuit.migrad()
    minuit.hesse()
    return minuit


def plot_afb(
    scan: pd.DataFrame,
    fit_result: Minuit,
    mZ: float,
    gammaZ: float,
    out_path: Path,
    *,
    sin2w_true: float = SIN2_THETAW_EFF,
    mZ_true: float = TRUE_MZ_GEV,
    gammaZ_true: float = TRUE_GAMMAZ_GEV,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    x_smooth = np.linspace(scan["sqrt_s_GeV"].min() - 0.3, scan["sqrt_s_GeV"].max() + 0.3, 200)

    ax.axhline(0, color=MUTED, linewidth=1, zorder=0)
    ax.plot(
        x_smooth,
        a_fb_true(x_smooth, sin2thetaw=sin2w_true, mZ=mZ_true, gammaZ=gammaZ_true),
        "-",
        color=YELLOW,
        linewidth=2,
        label="True A_FB(s) (gamma/Z interference)",
        zorder=1,
    )
    ax.plot(
        x_smooth,
        a_fb_true(x_smooth, sin2thetaw=fit_result.values["sin2thetaw"], mZ=mZ, gammaZ=gammaZ),
        "-",
        color=AQUA,
        linewidth=2,
        label="Fitted A_FB(s)",
        zorder=2,
    )
    ax.errorbar(
        scan["sqrt_s_GeV"],
        scan["afb_measured"],
        yerr=scan["afb_err"],
        fmt="o",
        color=BLUE,
        markersize=7,
        markeredgewidth=0,
        elinewidth=2,
        capsize=3,
        label="Simulated measurement (F/B counting)",
        zorder=3,
    )

    ax.set_xlabel(r"$\sqrt{s}$ (GeV)", color=INK)
    ax.set_ylabel(r"$A_{FB}(e^+e^- \to \mu^+\mu^-)$", color=INK)
    ax.set_title("Forward-backward asymmetry across the Z peak", color=INK, loc="left")

    ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)

    legend = ax.legend(frameon=False, loc="upper right", fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def plot_closure(
    scan: pd.DataFrame,
    fit_plain,
    fit_isr,
    out_path: Path,
    *,
    sigma_peak_true: float = TRUE_SIGMA_PEAK_NB,
    mZ_true: float = TRUE_MZ_GEV,
    gammaZ_true: float = TRUE_GAMMAZ_GEV,
    true_model=breit_wigner_isr,
    true_label: str = "True generating curve (ISR-convolved)",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    x_smooth = np.linspace(scan["sqrt_s_GeV"].min() - 0.3, scan["sqrt_s_GeV"].max() + 0.3, 200)

    ax.plot(
        x_smooth,
        true_model(x_smooth, sigma_peak_true, mZ_true, gammaZ_true),
        "-",
        color=YELLOW,
        linewidth=2,
        label=true_label,
        zorder=1,
    )
    ax.plot(
        x_smooth,
        breit_wigner(x_smooth, *fit_plain.values),
        "--",
        color=ORANGE,
        linewidth=2,
        label="Naive Breit-Wigner fit (no ISR)",
        zorder=2,
    )
    ax.plot(
        x_smooth,
        breit_wigner_isr(x_smooth, *fit_isr.values),
        "-",
        color=AQUA,
        linewidth=2,
        label="Breit-Wigner fit (ISR-convolved)",
        zorder=2,
    )
    ax.errorbar(
        scan["sqrt_s_GeV"],
        scan["sigma_computed_nb"],
        yerr=scan["sigma_computed_err_nb"],
        fmt="o",
        color=BLUE,
        markersize=7,
        markeredgewidth=0,
        elinewidth=2,
        capsize=3,
        label="Simulated measurement (this script)",
        zorder=3,
    )

    ax.set_xlabel(r"$\sqrt{s}$ (GeV)", color=INK)
    ax.set_ylabel(r"$\sigma(e^+e^- \to \mu^+\mu^-)$ (nb)", color=INK)
    ax.set_title("Closure test: simulated data vs. true generating model", color=INK, loc="left")

    ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)

    legend = ax.legend(frameon=False, loc="upper right", fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main() -> None:
    print(f"True input: mZ={TRUE_MZ_GEV:.4f} GeV, gammaZ={TRUE_GAMMAZ_GEV:.4f} GeV, "
          f"sigma_peak={TRUE_SIGMA_PEAK_NB:.4f} nb (from 12*pi/mZ^2 * Gamma_ll^2/Gamma_Z^2 "
          f"-- should land close to the ~2.0 nb LEP actually measured)")

    scan = run_scan(RNG)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n=== Simulated mu+mu- cross section scan ===")
    print(scan.to_string(index=False))

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    scan.to_csv(DATA_PROCESSED / "simulated_mumu_scan.csv", index=False)
    print(f"\nSaved scan table to {DATA_PROCESSED / 'simulated_mumu_scan.csv'}")

    print("\n=== Naive Breit-Wigner fit to simulated points ===")
    fit_plain = fit_breit_wigner(scan)
    print_fit_result(fit_plain)
    pull_mZ_plain = (fit_plain.values["mZ"] - TRUE_MZ_GEV) / fit_plain.errors["mZ"]
    pull_gammaZ_plain = (fit_plain.values["gammaZ"] - TRUE_GAMMAZ_GEV) / fit_plain.errors["gammaZ"]
    print(f"  pull vs. TRUE input: mZ {pull_mZ_plain:+.2f} sigma, gammaZ {pull_gammaZ_plain:+.2f} sigma")

    print("\n=== ISR-convolved Breit-Wigner fit to simulated points ===")
    fit_isr = fit_breit_wigner_isr(scan)
    print_fit_result(fit_isr)
    pull_mZ_isr = (fit_isr.values["mZ"] - TRUE_MZ_GEV) / fit_isr.errors["mZ"]
    pull_gammaZ_isr = (fit_isr.values["gammaZ"] - TRUE_GAMMAZ_GEV) / fit_isr.errors["gammaZ"]
    print(f"  pull vs. TRUE input: mZ {pull_mZ_isr:+.2f} sigma, gammaZ {pull_gammaZ_isr:+.2f} sigma")

    plot_closure(
        scan,
        fit_plain,
        fit_isr,
        DATA_PROCESSED / "simulated_mumu_closure.png",
        sigma_peak_true=TRUE_SIGMA_PEAK_NB,
        mZ_true=TRUE_MZ_GEV,
        gammaZ_true=TRUE_GAMMAZ_GEV,
    )

    print("\n=== Sanity checks on the gamma/Z interference (IBA) formula ===")
    gamma_ee_check = gamma_ee_from_couplings()
    print(f"  Gamma_ee from couplings = {gamma_ee_check * 1000:.2f} MeV (PDG: ~84 MeV)")
    sigma_peak_check = sigma_peak_from_iba()
    print(
        f"  sigma_peak from IBA F1(s=mZ^2) = {sigma_peak_check:.4f} nb "
        f"(should be close to TRUE_SIGMA_PEAK_NB = {TRUE_SIGMA_PEAK_NB:.4f} nb, "
        f"an independently-derived number)"
    )

    print("\n=== A_FB(s) fit (mZ, gammaZ fixed from the ISR-convolved lineshape fit above) ===")
    fit_afb_result = fit_afb(scan, fit_isr.values["mZ"], fit_isr.values["gammaZ"])
    print(f"chi2/ndof = {fit_afb_result.fval:.2f}/{fit_afb_result.ndof:.0f}")
    fitted_sin2 = fit_afb_result.values["sin2thetaw"]
    fitted_sin2_err = fit_afb_result.errors["sin2thetaw"]
    pull_sin2 = (fitted_sin2 - SIN2_THETAW_EFF) / fitted_sin2_err
    print(f"  sin2(theta_W_eff) = {fitted_sin2:.5f} +/- {fitted_sin2_err:.5f}   (true input: {SIN2_THETAW_EFF:.5f})")
    print(f"  pull vs. TRUE input: {pull_sin2:+.2f} sigma")

    plot_afb(
        scan,
        fit_afb_result,
        fit_isr.values["mZ"],
        fit_isr.values["gammaZ"],
        DATA_PROCESSED / "simulated_mumu_afb.png",
        sin2w_true=SIN2_THETAW_EFF,
        mZ_true=TRUE_MZ_GEV,
        gammaZ_true=TRUE_GAMMAZ_GEV,
    )


if __name__ == "__main__":
    main()
