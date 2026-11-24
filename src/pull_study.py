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
    python src/pull_study.py --replot      # new summary/plot from the saved toys
    python src/pull_study.py --expected-errors   # check of the chi2 bias, see below

A least-squares fit that takes each point's error from its own observed
count is slightly biased: a downward fluctuation gets a smaller error and so
more weight. --expected-errors gives every point the error expected from the
truth model instead, which no real analysis can do, to show how much of the
small Gamma_Z bias is this effect. Its outputs carry the suffix
_expected_errors and are not used by the README figures.
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
    breit_wigner_isr,
    fit_breit_wigner,
    fit_breit_wigner_isr,
)
from simulate_mumu import (
    LUMI_PBINV,
    SCAN_POINTS_GEV,
    SIN2_THETAW_EFF,
    TRUE_GAMMAZ_GEV,
    TRUE_MZ_GEV,
    TRUE_SIGMA_PEAK_NB,
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


def expected_point_errors(a_hat: float) -> np.ndarray:
    """Cross-section error of each scan point if its selected-event count
    were exactly the truth model's expectation (--expected-errors)."""
    sigma_true = breit_wigner_isr(SCAN_POINTS_GEV, TRUE_SIGMA_PEAK_NB, TRUE_MZ_GEV, TRUE_GAMMAZ_GEV)
    n_selected = LUMI_PBINV * sigma_true * 1000 * a_hat
    return np.sqrt(n_selected) / a_hat / LUMI_PBINV / 1000


def one_toy(seed: int, a_hat: float, a_hat_err: float, expected_err: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(seed)
    scan = pd.DataFrame([generate_scan_point(x, a_hat, a_hat_err, rng) for x in SCAN_POINTS_GEV])
    if expected_err is not None:
        scan["sigma_computed_err_nb"] = expected_err

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
                frac_above_1sigma=np.mean(p > 1),
                frac_below_minus_1sigma=np.mean(p < -1),
            )
        )
    return pd.DataFrame(rows)


def plot_pulls(results: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE)
    # (main column, comparison column, x label, main label, comparison label).
    # The naive fit's pulls (+10 to +18) are off this axis: they are quoted
    # in a note rather than drawn as a clipped bar that would flatten the rest.
    panels = [
        ("pull_mZ_isr", "pull_mZ_naive", r"$M_Z$", "ISR-convolved fit", None),
        ("pull_gammaZ_isr", "pull_gammaZ_naive", r"$\Gamma_Z$", "ISR-convolved fit", None),
        ("pull_sin2w_minos", "pull_sin2w", r"$\sin^2\theta_W^{\rm eff}$", r"$A_{FB}$ fit, MINOS error", "same, Hesse error"),
    ]
    bins = np.linspace(-5, 5, 41)
    x = np.linspace(-5, 5, 300)
    for ax, (main_col, other_col, label, main_label, other_label) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.hist(results[main_col], bins=bins, color=BLUE, alpha=0.85, label=main_label)
        if other_label is not None:
            ax.hist(results[other_col], bins=bins, color=ORANGE, alpha=0.5, label=other_label)
        else:
            naive = results[other_col]
            ax.text(0.98, 0.97, f"naive fit (no ISR):\nmean pull {naive.mean():+.1f},\n"
                    f"{np.mean(naive > 5):.0%} of toys beyond +5", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, color=ORANGE)
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
    parser.add_argument("--replot", action="store_true",
                        help="redo the summary and plot from the saved pull_study_toys.csv, without new toys")
    parser.add_argument("--expected-errors", action="store_true",
                        help="use the truth model's expected error for every cross-section point (chi2-bias check)")
    args = parser.parse_args()

    suffix = "_expected_errors" if args.expected_errors else ""
    toys_csv = DATA_PROCESSED / f"pull_study_toys{suffix}.csv"
    if args.replot:
        results = pd.read_csv(toys_csv)
        print(f"Re-summarising {len(results)} toys from {toys_csv}")
    else:
        a_hat, a_hat_err = measure_acceptance(np.random.default_rng(20260801))
        print(f"Acceptance A_hat = {a_hat:.4f} +/- {a_hat_err:.4f} (measured once, reused for every toy)")
        expected_err = expected_point_errors(a_hat) if args.expected_errors else None
        if args.expected_errors:
            print("Cross-section errors: expected from the truth model, not from each toy's own counts.")

        results = []
        for i in range(args.n_toys):
            results.append(one_toy(args.first_seed + i, a_hat, a_hat_err, expected_err))
            if (i + 1) % 50 == 0 or i + 1 == args.n_toys:
                print(f"  {i + 1}/{args.n_toys} toys done")
        results = pd.DataFrame(results)

        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        results.to_csv(toys_csv, index=False)

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
        "  (68.3% within |pull| < 1, 15.9% beyond on each side). The naive fit is expected to fail this\n"
        "  badly -- it ignores ISR -- and is shown as the control. The\n"
        "  sin2(theta_W) chi2 is not parabolic, so compare its Hesse pull with\n"
        "  the MINOS one."
    )
    for name, column, truth in [("M_Z", "mZ_isr", TRUE_MZ_GEV), ("Gamma_Z", "gammaZ_isr", TRUE_GAMMAZ_GEV)]:
        bias = (results[column] - truth) * 1e3
        print(f"  ISR-convolved fit, {name}: mean (fit - truth) = {bias.mean():+.1f} +/- {bias.std(ddof=1) / np.sqrt(len(bias)):.1f} MeV, "
              f"mean fit error {results[column + '_err'].mean() * 1e3:.1f} MeV, mean chi2 {results['chi2_isr'].mean():.2f}")
    saved = DATA_PROCESSED / f"pull_study_summary{suffix}.csv"
    summary.to_csv(saved, index=False)
    print(f"\nSaved {saved}" + ("" if args.replot else f" and {toys_csv}"))

    plot_pulls(results, summary, DATA_PROCESSED / f"pull_study{suffix}.png")


if __name__ == "__main__":
    main()
