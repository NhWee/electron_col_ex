"""
sin2(theta_W_eff) from OPAL's measured lepton forward-backward asymmetries.

The Zedometry paper does not only tabulate N, L and f: Tables 22 and 23 list
the measured e+e- -> mu+mu- and tau+tau- forward-backward asymmetries at all
29 energy points of 1990-1995, with their statistical errors, and Table 25
gives the systematic covariance. Those are transcribed into data/raw/ and
fitted here with the same improved-Born A_FB formula the closure tests use
(simulate_mumu.py), so this is the real-data counterpart of the toy-MC and
MadGraph5 A_FB results.

What the model contains, and what it does not:

- gamma + Z exchange and their interference, improved Born, with a running
  Z width (the same mass definition as the lineshape fit) and alpha(M_Z) in
  the photon/Z propagator ratio (alpha(0) as a variation).
- Initial-state radiation, folded into the forward-backward and total cross
  sections separately with the O(alpha) leading-log radiator of
  compute_cross_section.py, up to a maximum radiated energy fraction x_max.
  OPAL's asymmetry acceptance cuts on the acollinearity (< 15 deg), not on
  s'/s; x_max = 0.25 (s'/s > 0.75) is used as the equivalent, and varied.
- OPAL's acceptance |cos theta_l-| < k (k = 0.95 for mu, 0.90 for tau),
  through their Equation 5: A(|cos| < k) = A(full) * 4k / (3 + k^2).
- NOT included: initial-final state interference (OPAL leaves it in the
  measured values), final-state radiation, and the ISR boost on the angle.
  How much these matter is estimated by comparing this model with OPAL's own
  ZFITTER-based fit values, which Tables 22 and 23 also list.

M_Z and Gamma_Z are fixed from this project's lineshape fit, mirroring the
two-step procedure of the closure tests, and sin2(theta_W_eff) is the only
free parameter (lepton universality: the same value for e, mu and tau).

    python src/afb_real_data.py

Reference: OPAL quotes sin2(theta_lept_eff) = 0.2325 +- 0.0010 and
A_FB^(0,l) = 0.0145 +- 0.0017 from all three lepton species (hep-ex/0012018,
Eq. 30 and abstract).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit

from compute_cross_section import (
    AQUA,
    BLUE,
    DATA_PROCESSED,
    DATA_RAW,
    GRIDLINE,
    INK,
    MUTED,
    ORANGE,
    SURFACE,
    isr_beta,
)

ALPHA_0 = 1 / 137.035999
ALPHA_MZ = 1 / 128.95  # on-shell alpha(M_Z^2), the improved-Born choice
G_FERMI_GEV2 = 1.1663788e-5

# This project's ISR-convolved mu+mu- lineshape fit (README section 1) ...
MZ_LINESHAPE, GZ_LINESHAPE = 91.1865, 2.5287
# ... and the joint qq+mumu+tautau fit with beam spread (README section 4.3).
MZ_JOINT, GZ_JOINT = 91.1974, 2.5144

OPAL_SIN2W_EFF, OPAL_SIN2W_EFF_ERR = 0.2325, 0.0010
OPAL_A0FB_LEPT, OPAL_A0FB_LEPT_ERR = 0.0145, 0.0017

ACCEPTANCE_K = {"mumu": 0.95, "tautau": 0.90}
X_MAX_DEFAULT = 0.25

SYST_CATEGORIES = ["1990", "1991", "1992", "1993 peak-2", "1993 peak", "1993 peak+2",
                   "1994 peak", "1995 peak-2", "1995 peak", "1995 peak+2"]

# Gauss-Legendre nodes for the radiator integrals (see isr_average).
_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)


# --- data -------------------------------------------------------------------


def syst_category(year: int, sample: str) -> str:
    """Map a Table 22/23 row onto a Table 25 systematic category. 1990-1992
    and 1994 have one category per year; the 1993/1995 prescans run at the
    peak energy and are given that year's peak category."""
    if year in (1990, 1991, 1992):
        return str(year)
    if year == 1994:
        return "1994 peak"
    base = sample if sample in ("peak-2", "peak", "peak+2") else "peak"
    return f"{year} {base}"


def load_afb() -> pd.DataFrame:
    frames = []
    for tab, ch in [(22, "mumu"), (23, "tautau")]:
        d = pd.read_csv(DATA_RAW / f"opal_zedometry_table{tab}_{ch}_afb.csv")
        d.insert(0, "channel", ch)
        d["syst_category"] = [syst_category(y, s) for y, s in zip(d["year"], d["sample"])]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def afb_covariance(data: pd.DataFrame) -> np.ndarray:
    """Statistical errors (diagonal) plus Table 25's systematic covariance.
    Table 25 gives signed square roots of covariance elements in units of
    1e-3; there are no correlations between species."""
    syst = pd.read_csv(DATA_RAW / "opal_zedometry_table25_lepton_afb_syst.csv")
    cov = np.diag(data["afb_stat_err"].to_numpy() ** 2)
    for ch, g in syst.groupby("species"):
        root = g.set_index("category")[SYST_CATEGORIES].astype(float)
        idx = np.flatnonzero((data["channel"] == ch).to_numpy())
        cats = data["syst_category"].to_numpy()[idx]
        r = root.loc[cats, cats].to_numpy()
        cov[np.ix_(idx, idx)] += np.sign(r) * (r * 1e-3) ** 2
    return cov


# --- model ------------------------------------------------------------------


def f1_f2(sqrt_s: np.ndarray, sin2w: float, mZ: float, gZ: float, alpha: float = ALPHA_MZ):
    """Improved-Born F1, F2 (README section 0.3) with a running-width Z
    propagator: chi(s) = kappa s / (s - mZ^2 + i s gZ / mZ)."""
    s = np.asarray(sqrt_s, dtype=float) ** 2
    kappa = G_FERMI_GEV2 * mZ**2 / (2 * np.sqrt(2) * np.pi * alpha)
    chi = kappa * s / (s - mZ**2 + 1j * s * gZ / mZ)
    v, a, q = -0.5 + 2 * sin2w, -0.5, -1.0
    f1 = q**2 - 2 * q * v * v * chi.real + (v**2 + a**2) ** 2 * np.abs(chi) ** 2
    f2 = -4 * q * a * a * chi.real + 8 * (v * a) ** 2 * np.abs(chi) ** 2
    return f1, f2


def isr_average(func, sqrt_s: float, x_max: float) -> float:
    """int_0^x_max H(x) func(s(1-x)) dx with the full O(alpha) leading-log
    radiator H(x) = beta x^(beta-1)(1 + 3beta/4) - beta(1 - x/2). The soft
    term is integrated in u = x^beta, which removes its endpoint singularity
    (beta x^(beta-1) dx = du), so plain Gauss-Legendre is accurate."""
    beta = isr_beta(sqrt_s)
    u_max = x_max**beta
    u = 0.5 * u_max * (_GL_X + 1)
    x_soft = u ** (1 / beta)
    soft = (1 + 0.75 * beta) * 0.5 * u_max * np.sum(_GL_W * func(sqrt_s * np.sqrt(1 - x_soft)))
    x = 0.5 * x_max * (_GL_X + 1)
    hard = -beta * 0.5 * x_max * np.sum(_GL_W * (1 - x / 2) * func(sqrt_s * np.sqrt(1 - x)))
    return soft + hard


def afb_model(sqrt_s, sin2w, mZ, gZ, *, k: float = 1.0, x_max: float = X_MAX_DEFAULT,
              alpha: float = ALPHA_MZ) -> np.ndarray:
    """Forward-backward asymmetry within |cos theta| < k. With x_max = 0 it
    is the Born value 3/8 F2/F1; otherwise the forward-backward and total
    cross sections (each proportional to F/s') are radiated separately and
    their ratio taken."""
    sqrt_s = np.atleast_1d(np.asarray(sqrt_s, dtype=float))
    accept = 4 * k / (3 + k**2)
    if x_max <= 0:
        f1, f2 = f1_f2(sqrt_s, sin2w, mZ, gZ, alpha)
        return accept * 0.375 * f2 / f1
    out = np.empty_like(sqrt_s)
    for i, rs in enumerate(sqrt_s):
        num = isr_average(lambda r: f1_f2(r, sin2w, mZ, gZ, alpha)[1] / r**2, rs, x_max)
        den = isr_average(lambda r: f1_f2(r, sin2w, mZ, gZ, alpha)[0] / r**2, rs, x_max)
        out[i] = accept * 0.375 * num / den
    return out


def pole_asymmetry(sin2w: float) -> float:
    """A_FB^(0,l) = 3/4 A_l^2 with A_l = 2 v a / (v^2 + a^2)."""
    v, a = -0.5 + 2 * sin2w, -0.5
    a_l = 2 * v * a / (v**2 + a**2)
    return 0.75 * a_l**2


# --- fit --------------------------------------------------------------------


def missing_effects_correction(data: pd.DataFrame, x_max: float = X_MAX_DEFAULT, alpha: float = ALPHA_MZ) -> np.ndarray:
    """Per point: OPAL's ZFITTER-based fit value minus this model evaluated at
    OPAL's own fitted parameters (sin2w_eff = 0.2325, M_Z = 91.1852,
    Gamma_Z = 2.4948). This is the size of what the model leaves out
    (initial-final interference, FSR, the exact E > 6 GeV / acollinearity
    acceptance, higher-order ISR). Adding it to the model is a calibration
    against ZFITTER, so the result that uses it is no longer independent of
    OPAL's theory code; it is reported as a variation, not the baseline."""
    out = np.empty(len(data))
    for i, (_, r) in enumerate(data.iterrows()):
        out[i] = r["afb_fit_opal"] - afb_model(r["sqrt_s_mean_GeV"], OPAL_SIN2W_EFF, 91.1852, 2.4948,
                                               k=ACCEPTANCE_K[r["channel"]], x_max=x_max, alpha=alpha)[0]
    return out


def fit_sin2w(data: pd.DataFrame, *, mZ: float = MZ_LINESHAPE, gZ: float = GZ_LINESHAPE,
              x_max: float = X_MAX_DEFAULT, alpha: float = ALPHA_MZ, minos: bool = False,
              free_mz: bool = False, zfitter_correction: bool = False) -> Minuit:
    """chi^2 fit of sin2(theta_W_eff) to the beam-spread-corrected asymmetries
    with the full stat + syst covariance. A_FB depends on sin2w only through
    v^2, so the fit is started below 0.25 and bounded to the physical
    branch (the pull study showed the symmetric error is unreliable there:
    MINOS errors are available for that reason)."""
    x = data["sqrt_s_mean_GeV"].to_numpy()
    y = data["afb_corrected"].to_numpy()
    ks = data["channel"].map(ACCEPTANCE_K).to_numpy()
    cov_inv = np.linalg.inv(afb_covariance(data))
    shift = missing_effects_correction(data, x_max, alpha) if zfitter_correction else np.zeros_like(y)

    def predict(s2w, m):
        out = np.empty_like(y)
        for k in np.unique(ks):
            sel = ks == k
            out[sel] = afb_model(x[sel], s2w, m, gZ, k=k, x_max=x_max, alpha=alpha)
        return out + shift

    if free_mz:
        def chi2(sin2w, mZ_fit):
            r = y - predict(sin2w, mZ_fit)
            return float(r @ cov_inv @ r)
        m = Minuit(chi2, sin2w=0.232, mZ_fit=mZ)
    else:
        def chi2(sin2w):
            r = y - predict(sin2w, mZ)
            return float(r @ cov_inv @ r)
        m = Minuit(chi2, sin2w=0.232)
    m.errordef = Minuit.LEAST_SQUARES
    m.limits["sin2w"] = (0.15, 0.2499)
    m.migrad()
    m.hesse()
    if minos:
        m.minos("sin2w")
    return m


def describe(label: str, m: Minuit, n: int) -> dict:
    s, e = m.values["sin2w"], m.errors["sin2w"]
    row = dict(fit=label, sin2w=s, err=e, chi2=m.fval, ndof=n - m.nfit,
               pull_vs_opal=(s - OPAL_SIN2W_EFF) / np.hypot(e, OPAL_SIN2W_EFF_ERR),
               a0fb=pole_asymmetry(s))
    if m.merrors and "sin2w" in m.merrors:
        row["err_lo"], row["err_hi"] = m.merrors["sin2w"].lower, m.merrors["sin2w"].upper
    if "mZ_fit" in m.parameters:
        row["mZ_fit"], row["mZ_fit_err"] = m.values["mZ_fit"], m.errors["mZ_fit"]
    return row


def model_vs_opal_fit(data: pd.DataFrame) -> pd.DataFrame:
    """Our model at OPAL's own sin2w and M_Z, against the 'fit' column of
    Tables 22/23 (OPAL's ZFITTER-based 9-parameter fit, which includes
    initial-final interference and their exact acceptance). The difference
    is a model-level systematic independent of the data's fluctuations."""
    main = data[data["year"].isin([1993, 1995]) & data["sample"].isin(["peak-2", "peak", "peak+2"])].copy()
    pred = []
    for _, r in main.iterrows():
        pred.append(afb_model(r["sqrt_s_mean_GeV"], OPAL_SIN2W_EFF, 91.1852, 2.4948,
                              k=ACCEPTANCE_K[r["channel"]])[0])
    main["afb_this_model"] = pred
    main["model_minus_opal_fit"] = main["afb_this_model"] - main["afb_fit_opal"]
    return main[["channel", "year", "sample", "sqrt_s_mean_GeV", "afb_fit_opal", "afb_this_model", "model_minus_opal_fit"]]


# --- plot -------------------------------------------------------------------


def plot(data: pd.DataFrame, m: Minuit, m_born: Minuit, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.linspace(88.0, 94.5, 160)
    s2w = m.values["sin2w"]
    ax.axhline(0, color=MUTED, lw=1)
    ax.plot(x, afb_model(x, s2w, MZ_LINESHAPE, GZ_LINESHAPE, k=0.95), color=AQUA, lw=2,
            label=f"fit with ISR, sin²θ_W = {s2w:.4f}")
    ax.plot(x, afb_model(x, m_born.values["sin2w"], MZ_LINESHAPE, GZ_LINESHAPE, k=0.95, x_max=0), "--",
            color=MUTED, lw=1.5, label=f"Born fit (no ISR): hits the 0.25 bound, χ²/ndf = {m_born.fval:.0f}/{len(data) - 1}")
    for ch, col, mk, dx, tab in [("mumu", BLUE, "o", -0.03, 22), ("tautau", ORANGE, "s", 0.03, 23)]:
        d = data[data["channel"] == ch]
        ax.errorbar(d["sqrt_s_mean_GeV"] + dx, d["afb_corrected"], yerr=d["afb_stat_err"], fmt=mk, color=col,
                    ms=5, elinewidth=1.2, capsize=0, label=f"OPAL {ch} (Table {tab})")
    ax.set_xlabel(r"$\sqrt{s}$ (GeV)", color=INK)
    ax.set_ylabel(r"$A_{FB}$", color=INK)
    ax.set_title("OPAL lepton forward-backward asymmetry, 1990-1995", color=INK, loc="left")
    ax.grid(True, color=GRIDLINE, lw=1, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED)
    leg = ax.legend(frameon=False, loc="upper left", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


# --- main -------------------------------------------------------------------


def main() -> None:
    data = load_afb()
    n_all = len(data)
    print(f"Loaded {n_all} OPAL asymmetry points ({(data.channel == 'mumu').sum()} mumu, "
          f"{(data.channel == 'tautau').sum()} tautau), 1990-1995")

    print("\n=== Baseline: mumu + tautau, ISR (x_max = 0.25), alpha(M_Z), M_Z/Gamma_Z from the lineshape fit ===")
    base = fit_sin2w(data, minos=True)
    b = describe("baseline", base, n_all)
    print(f"chi2/ndof = {b['chi2']:.1f}/{b['ndof']}")
    print(f"  sin2(theta_W_eff) = {b['sin2w']:.5f} +/- {b['err']:.5f}  (MINOS {b['err_lo']:+.5f} / {b['err_hi']:+.5f})")
    print(f"  A_FB^(0,l) = {b['a0fb']:.4f}   (OPAL: {OPAL_A0FB_LEPT} +/- {OPAL_A0FB_LEPT_ERR})")
    print(f"  vs OPAL sin2(theta_lept_eff) = {OPAL_SIN2W_EFF} +/- {OPAL_SIN2W_EFF_ERR}: {b['pull_vs_opal']:+.2f} sigma")

    rows = [b]
    variations = [
        ("mumu only", dict(), data[data.channel == "mumu"]),
        ("tautau only", dict(), data[data.channel == "tautau"]),
        ("Born, no ISR", dict(x_max=0.0), data),
        ("ISR x_max = 0.15", dict(x_max=0.15), data),
        ("ISR x_max = 0.40", dict(x_max=0.40), data),
        ("alpha(0) in chi(s)", dict(alpha=ALPHA_0), data),
        ("M_Z, Gamma_Z from joint fit", dict(mZ=MZ_JOINT, gZ=GZ_JOINT), data),
        ("M_Z free in the A_FB fit", dict(free_mz=True), data),
        ("+ ZFITTER-calibrated missing effects", dict(zfitter_correction=True), data),
    ]
    fits = {"baseline": base}
    for label, kw, d in variations:
        m = fit_sin2w(d, **kw)
        fits[label] = m
        rows.append(describe(label, m, len(d)))
    table = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:.5f}")
    pd.set_option("display.width", 160)
    print("\n=== Variations ===")
    print(table[["fit", "sin2w", "err", "chi2", "ndof", "a0fb", "pull_vs_opal"]].to_string(index=False))
    print("  'Born, no ISR' and 'M_Z free' run into the sin2w = 0.25 bound: without ISR no value\n"
          "  describes the data (see chi2), and A_FB alone cannot separate M_Z from sin2w (the\n"
          "  zero crossing moves with both). Neither is a measurement.")

    model_syst = {
        "ISR cut x_max 0.15-0.40": max(abs(table.set_index("fit").loc[l, "sin2w"] - b["sin2w"])
                                       for l in ["ISR x_max = 0.15", "ISR x_max = 0.40"]),
        "alpha(0) vs alpha(M_Z)": abs(table.set_index("fit").loc["alpha(0) in chi(s)", "sin2w"] - b["sin2w"]),
        "M_Z, Gamma_Z input": abs(table.set_index("fit").loc["M_Z, Gamma_Z from joint fit", "sin2w"] - b["sin2w"]),
        "missing effects (vs ZFITTER)": abs(table.set_index("fit").loc["+ ZFITTER-calibrated missing effects", "sin2w"] - b["sin2w"]),
    }
    total = float(np.sqrt(sum(v**2 for v in model_syst.values())))
    print("\n=== Model systematics on sin2(theta_W_eff) (shift of the central value) ===")
    for k, v in model_syst.items():
        print(f"  {k:30s} {v:.5f}")
    print(f"  {'total (quadrature)':30s} {total:.5f}")
    print(f"\n  sin2(theta_W_eff) = {b['sin2w']:.4f} +/- {b['err']:.4f} (stat+syst) +/- {total:.4f} (model)"
          f"   vs OPAL {OPAL_SIN2W_EFF} +/- {OPAL_SIN2W_EFF_ERR}: "
          f"{(b['sin2w'] - OPAL_SIN2W_EFF) / np.sqrt(b['err']**2 + total**2 + OPAL_SIN2W_EFF_ERR**2):+.2f} sigma")
    calib = table.set_index("fit").loc["+ ZFITTER-calibrated missing effects"]
    print(f"  with the ZFITTER calibration: {calib['sin2w']:.4f} +/- {calib['err']:.4f}, "
          f"{calib['pull_vs_opal']:+.2f} sigma from OPAL")

    print("\n=== This model vs OPAL's own ZFITTER-based fit values (same sin2w, M_Z) ===")
    cmp_ = model_vs_opal_fit(data)
    print(cmp_.to_string(index=False))
    print("  These differences (initial-final interference, FSR, exact acceptance) are absent from\n"
          "  this model; they are comparable to the statistical error on a single peak point.")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    table.to_csv(DATA_PROCESSED / "afb_real_data_fits.csv", index=False)
    cmp_.to_csv(DATA_PROCESSED / "afb_model_vs_opal_fit.csv", index=False)
    print(f"\nSaved tables to {DATA_PROCESSED / 'afb_real_data_fits.csv'} and {DATA_PROCESSED / 'afb_model_vs_opal_fit.csv'}")
    plot(data, base, fits["Born, no ISR"], DATA_PROCESSED / "afb_real_data.png")


if __name__ == "__main__":
    main()
