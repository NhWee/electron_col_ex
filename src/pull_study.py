"""
Repeated-toy coverage test for the closure pipeline in `simulate_mumu.py`.

A single toy experiment (one seed) can only say "the fit landed within 1
sigma this time". Repeating it N times with fresh Poisson noise and fresh
angular samples tests the two things that one run cannot:

- bias:     is the mean pull (fit - truth) / fit_error compatible with 0?
- coverage: is the pull width compatible with 1, i.e. does the fit's own
            error estimate match the actual scatter of the result?

The sin2(theta_W) pull is computed twice: with the symmetric Hesse error,
and with the MINOS error on the side facing the truth. A_FB depends on
sin2(theta_W) only through v^2, so its chi2 is not parabolic and the
Hesse width can be wrong; the MINOS pull is the one to judge coverage by.

The truth model, acceptance correction and all fit functions are the ones
`simulate_mumu.py` uses; only the random seed changes between toys. The
acceptance A_hat is measured once from a dedicated sample, as a real
analysis would, and reused for every toy.

    python src/pull_study.py            # 500 toys, ~5 minutes
    python src/pull_study.py --n-toys 50
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from compute_cross_section import (
    AQUA,
    BLUE,
    DATA_PROCESSED,
    GRIDLINE,
    INK,
    MUTED,
    ORANGE,
    SURFACE,
    fit_breit_wigner,
    fit_breit_wigner_isr,
)
from simulate_mumu import (
    SCAN_POINTS_GEV,
    SIN2_THETAW_EFF,
    TRUE_GAMMAZ_GEV,
    TRUE_MZ_GEV,
    afb_errors,
    asymmetric_pull,
    fit_afb,
    generate_scan_point,
    measure_acceptance,
)

PULL_COLUMNS = {
    "pull_mZ_naive": ("naive fit", r"$M_Z$"),
    "pull_gammaZ_naive": ("naive fit", r"$\Gamma_Z$"),
    "pull_mZ_isr": ("ISR-convolved fit", r"$M_Z$"),
    "pull_gammaZ_isr": ("ISR-convolved fit", r"$\Gamma_Z$"),
    "pull_sin2w": (r"$A_{FB}$ fit, Hesse error", r"$\sin^2\theta_W^{\rm eff}$"),
    "pull_sin2w_minos": (r"$A_{FB}$ fit, MINOS error", r"$\sin^2\theta_W^{\rm eff}$"),
}


def one_toy(seed: int, a_hat: float, a_hat_err: float) -> dict:
    rng = np.random.default_rng(seed)
    scan = pd.DataFrame([generate_scan_point(x, a_hat, a_hat_err, rng) for x in SCAN_POINTS_GEV])

    naive = fit_breit_wigner(scan)
    isr = fit_breit_wigner_isr(scan)
    afb = fit_afb(scan, isr.values["mZ"], isr.values["gammaZ"])
    sin2w = afb.values["sin2thetaw"]
    lower, upper, from_minos = afb_errors(afb)

    return {
        "seed": seed,
        "mZ_naive": naive.values["mZ"], "mZ_naive_err": naive.errors["mZ"],
        "gammaZ_naive": naive.values["gammaZ"], "gammaZ_naive_err": naive.errors["gammaZ"],
        "chi2_naive": naive.fval,
        "mZ_isr": isr.values["mZ"], "mZ_isr_err": isr.errors["mZ"],
        "gammaZ_isr": isr.values["gammaZ"], "gammaZ_isr_err": isr.errors["gammaZ"],
        "chi2_isr": isr.fval, "isr_valid": bool(isr.valid),
        "sin2w": sin2w, "sin2w_err": afb.errors["sin2thetaw"],
        "sin2w_err_lower": lower, "sin2w_err_upper": upper, "minos_valid": from_minos,
        "chi2_afb": afb.fval, "afb_valid": bool(afb.valid),
        "pull_mZ_naive": (naive.values["mZ"] - TRUE_MZ_GEV) / naive.errors["mZ"],
        "pull_gammaZ_naive": (naive.values["gammaZ"] - TRUE_GAMMAZ_GEV) / naive.errors["gammaZ"],
        "pull_mZ_isr": (isr.values["mZ"] - TRUE_MZ_GEV) / isr.errors["mZ"],
        "pull_gammaZ_isr": (isr.values["gammaZ"] - TRUE_GAMMAZ_GEV) / isr.errors["gammaZ"],
        "pull_sin2w": (sin2w - SIN2_THETAW_EFF) / afb.errors["sin2thetaw"],
        "pull_sin2w_minos": asymmetric_pull(sin2w, SIN2_THETAW_EFF, lower, upper),
    }


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(results)
    for col, (fit_name, param) in PULL_COLUMNS.items():
        p = results[col].to_numpy()
        mean, std = p.mean(), p.std(ddof=1)
        rows.append(
            dict(
                column=col,
                fit=fit_name,
                parameter=param,
                n=n,
                mean=mean,
                mean_err=std / np.sqrt(n),
                width=std,
                width_err=std / np.sqrt(2 * (n - 1)),
                frac_within_1sigma=np.mean(np.abs(p) < 1),
            )
        )
    return pd.DataFrame(rows)


def plot_pulls(results: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE)
    # (main column, comparison column, x label, main label, comparison label)
    panels = [
        ("pull_mZ_isr", "pull_mZ_naive", r"$M_Z$", "ISR-convolved fit", "naive fit (clipped at ±5)"),
        ("pull_gammaZ_isr", "pull_gammaZ_naive", r"$\Gamma_Z$", "ISR-convolved fit", "naive fit (clipped at ±5)"),
        ("pull_sin2w_minos", "pull_sin2w", r"$\sin^2\theta_W^{\rm eff}$", r"$A_{FB}$ fit, MINOS error", "same, Hesse error"),
    ]
    bins = np.linspace(-5, 5, 41)
    x = np.linspace(-5, 5, 300)
    for ax, (main_col, other_col, label, main_label, other_label) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.hist(results[main_col], bins=bins, color=BLUE, alpha=0.85, label=main_label)
        clipped = np.clip(results[other_col], -4.9, 4.9)
        ax.hist(clipped, bins=bins, color=ORANGE, alpha=0.5, label=other_label)
        row = summary[summary["column"] == main_col].iloc[0]
        n = len(results)
        ax.plot(x, n * (bins[1] - bins[0]) * norm.pdf(x, row["mean"], row["width"]), color=AQUA, lw=2,
                label=f"Gaussian: μ={row['mean']:+.2f}±{row['mean_err']:.2f}, σ={row['width']:.2f}±{row['width_err']:.2f}")
        ax.plot(x, n * (bins[1] - bins[0]) * norm.pdf(x), color=MUTED, lw=1, ls="--", label="N(0,1)")
        ax.set_xlabel(f"pull of {label}", color=INK)
        ax.grid(True, color=GRIDLINE, lw=1, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
        leg = ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        for t in leg.get_texts():
            t.set_color(INK)
    axes[0].set_ylabel("toy experiments", color=INK)
    fig.suptitle(f"Pull distributions from {len(results)} toy experiments (13 points, 2 pb$^{{-1}}$ each)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-toys", type=int, default=500)
    parser.add_argument("--first-seed", type=int, default=1_000)
    args = parser.parse_args()

    a_hat, a_hat_err = measure_acceptance(np.random.default_rng(20260801))
    print(f"Acceptance A_hat = {a_hat:.4f} +/- {a_hat_err:.4f} (measured once, reused for every toy)")

    results = []
    for i in range(args.n_toys):
        results.append(one_toy(args.first_seed + i, a_hat, a_hat_err))
        if (i + 1) % 50 == 0 or i + 1 == args.n_toys:
            print(f"  {i + 1}/{args.n_toys} toys done")
    results = pd.DataFrame(results)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    results.to_csv(DATA_PROCESSED / "pull_study_toys.csv", index=False)

    n_bad = int((~results["isr_valid"]).sum() + (~results["afb_valid"]).sum())
    if n_bad:
        print(f"WARNING: {n_bad} fit(s) reported as invalid by Minuit; they are kept in the table.")
    n_no_minos = int((~results["minos_valid"]).sum())
    if n_no_minos:
        print(f"Note: MINOS found no upper limit in {n_no_minos} toy(s) (the 1 sigma interval reaches\n"
              f"  sin2 = 0.25 and the mirror solution); that side falls back to the Hesse error.")

    summary = summarize(results)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("\n=== Pull summary (fit - truth) / fit error ===")
    print(summary.to_string(index=False))
    print(
        "\n  A correct, unbiased fit with correct errors has mean 0 and width 1\n"
        "  (68.3% within |pull| < 1). The naive fit is expected to fail this\n"
        "  badly -- it ignores ISR -- and is shown as the control. The\n"
        "  sin2(theta_W) chi2 is not parabolic, so compare its Hesse pull with\n"
        "  the MINOS one."
    )
    summary.to_csv(DATA_PROCESSED / "pull_study_summary.csv", index=False)
    print(f"\nSaved tables to {DATA_PROCESSED / 'pull_study_toys.csv'} and {DATA_PROCESSED / 'pull_study_summary.csv'}")

    plot_pulls(results, summary, DATA_PROCESSED / "pull_study.png")


if __name__ == "__main__":
    main()
