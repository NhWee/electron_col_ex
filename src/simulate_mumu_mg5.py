"""
Generator-level closure test for e+e- -> mu+mu-, using MadGraph5_aMC@NLO
tree-level matrix elements as the ground truth instead of the hand-typed
"improved Born approximation" formula `simulate_mumu.py` uses.

`simulate_mumu.py` already showed that `fit_breit_wigner_isr`/`fit_afb`
recover a known truth -- but that truth was our own formula, so the fit
model and the "data" it's tested against share a common author. Here the
events come from MadGraph5's independently implemented Standard Model
matrix elements (2 diagrams: s-channel photon + Z) instead, so recovering
the right mZ, GammaZ and sin2(theta_W) is a genuinely stronger check of the
extraction pipeline. Everything downstream of "here are some simulated
events" -- the acceptance-correction pattern, the Poisson luminosity
statistics, the lineshape and A_FB fits, the plotting -- is reused unchanged
from `compute_cross_section.py` / `simulate_mumu.py`; only the event source
changes.

This mirrors the generate -> detector-level reconstruction -> analysis
structure of the user's own collider_analysis workflow
(MadGraph5 -> Pythia8 -> Delphes -> FastJet -> ROOT analysis), without
adopting the heavy shower/detector/jet-clustering machinery that workflow
needs for hadronic LHC final states -- not relevant here, since mu+mu- is a
clean two-lepton final state read directly off the parton-level event
record. No PDF, no ISR/beamstrahlung, no showering: a deliberate scope
choice (see README), not an oversight.

Requires a working MadGraph5_aMC@NLO installation (gfortran + gcc), external
to this repo. Point `MG5_PATH` at your `bin/mg5_aMC` if it isn't at the
default path below.
"""

import gzip
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from uncertainties import ufloat

from compute_cross_section import DATA_PROCESSED, breit_wigner, fit_breit_wigner, fit_breit_wigner_isr, print_fit_result
from simulate_mumu import SCAN_POINTS_GEV, fit_afb, plot_afb, plot_closure

MG5_EXECUTABLE = Path(
    os.environ.get(
        "MG5_PATH",
        str(Path.home() / "Documents/collider_analysis/MG5_aMC_v3_5_0/bin/mg5_aMC"),
    )
)
PROCESS_DIR = DATA_PROCESSED / "mg5_eemumu"

LUMI_PBINV_MG5 = 0.3  # smaller than simulate_mumu.py's 2.0 pb^-1, to keep each MG5 launch fast
EVENTS_PER_LAUNCH = 1200  # headroom above the on-peak Poisson mean (~600 at L=0.3 pb^-1)
EFFICIENCY_EVENTS = 8000
ACCEPTANCE_COSTHETA_MAX = 0.95  # same cut as simulate_mumu.py, for direct comparability

RNG = np.random.default_rng(20260803)

CROSS_SECTION_RE = re.compile(r"Cross-section\s*:\s*([\d.eE+-]+)\s*\+-\s*[\d.eE+-]+\s*pb")
RUN_NAME_RE = re.compile(r"Results Summary for run:\s*(\S+)\s+tag:")
MASS_LINE_RE = re.compile(r"^\s*(\d+)\s+([\d.eE+-]+)\s*#\s*(\S+)")
DECAY_LINE_RE = re.compile(r"^\s*DECAY\s+(\d+)\s+([\d.eE+-]+)")


def _run_mg5(commands: list[str]) -> str:
    if not MG5_EXECUTABLE.exists():
        raise FileNotFoundError(
            f"MadGraph5 executable not found at {MG5_EXECUTABLE}. "
            "Set the MG5_PATH environment variable to your bin/mg5_aMC."
        )
    with tempfile.NamedTemporaryFile("w", suffix=".mg5", delete=False) as f:
        f.write("\n".join(commands) + "\n")
        script_path = Path(f.name)
    try:
        result = subprocess.run(
            [str(MG5_EXECUTABLE), str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        script_path.unlink(missing_ok=True)
    return result.stdout


def ensure_process_generated() -> None:
    """Generate the e+e- -> mu+mu- process directory once (idempotent);
    reused by every launch below. gitignored under data/processed/."""
    if PROCESS_DIR.exists():
        return
    PROCESS_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating MadGraph5 process directory at {PROCESS_DIR} ...")
    _run_mg5(["generate e+ e- > mu+ mu-", f"output {PROCESS_DIR}"])


def read_param_card_truth() -> dict:
    """Ground truth read straight from MadGraph5's own param_card.dat, not
    hand-typed: MZ and its decay width, plus the *dependent* on-shell W mass
    (MG5 computes MW from MZ, Gf and alpha_EW under the on-shell scheme --
    the exact formula is printed as a comment in the file itself). From
    this, sin2(theta_W)_on-shell = 1 - (MW/MZ)^2 -- the tree-level on-shell
    value MG5's matrix elements actually use, which is *not* the same number
    as PDG's effective sin2(theta_W_eff) = 0.23155 used in
    simulate_mumu.py's IBA fit (a real, well-known ~0.009 scheme difference
    between the on-shell and effective definitions, not a bug)."""
    param_card = PROCESS_DIR / "Cards" / "param_card.dat"
    mass_values: dict[str, float] = {}
    decay_values: dict[str, float] = {}
    for line in param_card.read_text().splitlines():
        m = MASS_LINE_RE.match(line)
        if m:
            _, value, name = m.groups()
            mass_values[name] = float(value)
            continue
        m = DECAY_LINE_RE.match(line)
        if m:
            pdg, value = m.groups()
            decay_values[pdg] = float(value)

    mZ = mass_values["MZ"]
    mW = mass_values["w+"]
    gammaZ = decay_values["23"]
    sin2w_onshell = 1 - (mW / mZ) ** 2
    return {"mZ": mZ, "gammaZ": gammaZ, "mW": mW, "sin2w_onshell": sin2w_onshell}


def run_mg5_launch(sqrt_s: float, nevents: int, seed: int) -> tuple[Path, float]:
    """One MG5 launch at fixed sqrt(s): point-particle e+e- beams (lpp=0,
    no PDF -- correct for a lepton collider), showering/detector off (we
    apply our own toy acceptance below on the parton-level muons). Returns
    the generated LHE file path and MG5's own VEGAS-integrated cross
    section in pb, parsed from its stdout summary."""
    ebeam = sqrt_s / 2
    stdout = _run_mg5(
        [
            f"launch {PROCESS_DIR}",
            "shower=OFF",
            "detector=OFF",
            "done",
            f"set ebeam1 {ebeam}",
            f"set ebeam2 {ebeam}",
            "set lpp1 0",
            "set lpp2 0",
            f"set nevents {nevents}",
            f"set iseed {seed}",
            "done",
        ]
    )
    xsec_match = CROSS_SECTION_RE.search(stdout)
    run_match = RUN_NAME_RE.search(stdout)
    if not xsec_match or not run_match:
        raise RuntimeError(f"Could not parse MG5 output for sqrt_s={sqrt_s}:\n{stdout[-3000:]}")
    sigma_pb = float(xsec_match.group(1))
    lhe_path = PROCESS_DIR / "Events" / run_match.group(1) / "unweighted_events.lhe.gz"
    if not lhe_path.exists():
        raise RuntimeError(f"Expected LHE file not found: {lhe_path}")
    return lhe_path, sigma_pb


def parse_lhe_costheta(lhe_path: Path) -> np.ndarray:
    """cos(theta) of the outgoing mu- relative to the e+ beam direction (pdg
    -11, confirmed by hand to consistently carry pz=+E_beam in this
    process's LHE output). The lab frame is the CM frame here since
    lpp1=lpp2=0 -- no PDF/parton boost to undo."""
    with gzip.open(lhe_path, "rt") as f:
        text = f.read()
    costhetas = []
    for block in text.split("<event>")[1:]:
        body = block.split("</event>")[0]
        for line in body.splitlines():
            fields = line.split()
            if len(fields) < 10:
                continue
            try:
                pdg, status = int(fields[0]), int(fields[1])
            except ValueError:
                continue
            if pdg == 13 and status == 1:
                px, py, pz = float(fields[6]), float(fields[7]), float(fields[8])
                costhetas.append(pz / np.sqrt(px**2 + py**2 + pz**2))
                break
    return np.array(costhetas)


def measure_acceptance_mg5(truth: dict) -> tuple[float, float]:
    """A_hat and its binomial error, from a dedicated on-peak MG5 sample --
    plays the same role as measure_acceptance() in simulate_mumu.py, except
    measured from real generated events instead of rejection-sampled ones."""
    print(f"Measuring detector acceptance from a dedicated {EFFICIENCY_EVENTS}-event MG5 sample at sqrt_s = mZ ...")
    lhe_path, _ = run_mg5_launch(truth["mZ"], EFFICIENCY_EVENTS, seed=1)
    costheta = parse_lhe_costheta(lhe_path)
    passed = np.abs(costheta) < ACCEPTANCE_COSTHETA_MAX
    a_hat = passed.mean()
    a_hat_err = np.sqrt(a_hat * (1 - a_hat) / len(costheta))
    print(f"Acceptance A_hat = {a_hat:.4f} +/- {a_hat_err:.4f} (from {len(costheta)} real MG5 events)")
    return a_hat, a_hat_err


def generate_scan_point_mg5(sqrt_s: float, seed: int, a_hat: float, a_hat_err: float, rng: np.random.Generator) -> dict:
    """One MG5 launch at this energy; MG5's own reported sigma_pb (VEGAS-
    integrated, not a rough survey number) sets the Poisson mean for how
    many of the generated events we 'keep' at a toy luminosity of
    LUMI_PBINV_MG5 -- this is where Poisson statistics enter, applied by us
    on top of MG5's deterministic event count, mirroring the role
    n_gen = rng.poisson(...) plays in simulate_mumu.py's toy."""
    lhe_path, sigma_pb = run_mg5_launch(sqrt_s, EVENTS_PER_LAUNCH, seed=seed)
    costheta_pool = parse_lhe_costheta(lhe_path)
    n_pool = len(costheta_pool)

    n_keep = int(rng.poisson(LUMI_PBINV_MG5 * sigma_pb))
    if n_keep > n_pool:
        print(f"  WARNING: sqrt_s={sqrt_s:.2f}: wanted {n_keep} events, only {n_pool} generated -- capping.")
        n_keep = n_pool

    selected_idx = rng.choice(n_pool, size=n_keep, replace=False)
    costheta = costheta_pool[selected_idx]

    accepted = costheta[np.abs(costheta) < ACCEPTANCE_COSTHETA_MAX]
    n_selected = accepted.size
    n_forward = int(np.sum(accepted > 0))
    n_backward = n_selected - n_forward

    f_corr = ufloat(1 / a_hat, a_hat_err / a_hat**2)
    sigma = f_corr * ufloat(n_selected, np.sqrt(n_selected)) / LUMI_PBINV_MG5 / 1000  # pb -> nb

    afb_measured = (n_forward - n_backward) / n_selected
    afb_err = np.sqrt(max(1 - afb_measured**2, 0.0) / n_selected)

    return {
        "sqrt_s_GeV": sqrt_s,
        "sigma_mg5_true_pb": sigma_pb,
        "n_generated_pool": n_pool,
        "n_kept": n_keep,
        "n_selected": n_selected,
        "sigma_computed_nb": sigma.nominal_value,
        "sigma_computed_err_nb": sigma.std_dev,
        "n_forward": n_forward,
        "n_backward": n_backward,
        "afb_measured": afb_measured,
        "afb_err": afb_err,
    }


def run_scan_mg5(rng: np.random.Generator, truth: dict) -> pd.DataFrame:
    a_hat, a_hat_err = measure_acceptance_mg5(truth)
    n_points = len(SCAN_POINTS_GEV)
    rows = []
    print(f"Running {n_points}-point MadGraph5 scan (each point is a real MC generation call, this takes a while) ...")
    for i, sqrt_s in enumerate(SCAN_POINTS_GEV, start=1):
        row = generate_scan_point_mg5(sqrt_s, seed=100 + i, a_hat=a_hat, a_hat_err=a_hat_err, rng=rng)
        print(
            f"  [{i}/{n_points}] sqrt_s={sqrt_s:.3f} GeV  sigma_MG5={row['sigma_mg5_true_pb']:.2f} pb  "
            f"n_kept={row['n_kept']}  n_selected={row['n_selected']}  afb={row['afb_measured']:+.3f}"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ensure_process_generated()
    truth = read_param_card_truth()
    print(
        f"MG5 ground truth (from param_card.dat): mZ={truth['mZ']:.4f} GeV, "
        f"gammaZ={truth['gammaZ']:.4f} GeV, mW={truth['mW']:.4f} GeV\n"
        f"  -> sin2(theta_W)_on-shell = 1-(mW/mZ)^2 = {truth['sin2w_onshell']:.4f} "
        f"(NOT PDG's effective sin2(theta_W_eff)=0.23155 used in simulate_mumu.py -- "
        f"on-shell vs. effective is a real ~0.009 scheme difference, not a bug)"
    )

    scan = run_scan_mg5(RNG, truth)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n=== MadGraph5-generated mu+mu- cross section scan ===")
    print(scan.to_string(index=False))

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    scan.to_csv(DATA_PROCESSED / "simulated_mumu_mg5_scan.csv", index=False)
    print(f"\nSaved scan table to {DATA_PROCESSED / 'simulated_mumu_mg5_scan.csv'}")

    print("\n=== Naive Breit-Wigner fit to MG5-generated points ===")
    fit_plain = fit_breit_wigner(scan)
    print_fit_result(fit_plain)
    pull_mZ_plain = (fit_plain.values["mZ"] - truth["mZ"]) / fit_plain.errors["mZ"]
    pull_gammaZ_plain = (fit_plain.values["gammaZ"] - truth["gammaZ"]) / fit_plain.errors["gammaZ"]
    print(f"  pull vs. MG5 param_card truth: mZ {pull_mZ_plain:+.2f} sigma, gammaZ {pull_gammaZ_plain:+.2f} sigma")

    print("\n=== ISR-convolved Breit-Wigner fit to MG5-generated points ===")
    fit_isr = fit_breit_wigner_isr(scan)
    print_fit_result(fit_isr)
    pull_mZ = (fit_isr.values["mZ"] - truth["mZ"]) / fit_isr.errors["mZ"]
    pull_gammaZ = (fit_isr.values["gammaZ"] - truth["gammaZ"]) / fit_isr.errors["gammaZ"]
    print(f"  pull vs. MG5 param_card truth: mZ {pull_mZ:+.2f} sigma, gammaZ {pull_gammaZ:+.2f} sigma")
    print(
        "  NOTE: this MG5 process has ISR/beamstrahlung switched off (see README), so the\n"
        "  *naive* fit above is the physically correct lineshape model here -- convolving in\n"
        "  an ISR radiator that isn't actually present over-corrects and biases mZ/gammaZ\n"
        "  low, the mirror image of the OPAL case where the real data DOES have ISR and the\n"
        "  naive fit is the biased one. The AFB fit below deliberately uses the naive fit's\n"
        "  mZ/gammaZ, not this ISR-convolved one."
    )

    peak_row = scan.loc[(scan["sqrt_s_GeV"] - truth["mZ"]).abs().idxmin()]
    sigma_peak_true_nb = peak_row["sigma_mg5_true_pb"] / 1000

    plot_closure(
        scan,
        fit_plain,
        fit_isr,
        DATA_PROCESSED / "simulated_mumu_mg5_closure.png",
        sigma_peak_true=sigma_peak_true_nb,
        mZ_true=truth["mZ"],
        gammaZ_true=truth["gammaZ"],
        true_model=breit_wigner,
        true_label="True generating curve (MG5, no ISR)",
    )

    print("\n=== A_FB(s) fit (mZ, gammaZ fixed from the NAIVE lineshape fit -- see note above) ===")
    fit_afb_result = fit_afb(scan, fit_plain.values["mZ"], fit_plain.values["gammaZ"])
    print(f"chi2/ndof = {fit_afb_result.fval:.2f}/{fit_afb_result.ndof:.0f}")
    fitted_sin2 = fit_afb_result.values["sin2thetaw"]
    fitted_sin2_err = fit_afb_result.errors["sin2thetaw"]
    pull_sin2 = (fitted_sin2 - truth["sin2w_onshell"]) / fitted_sin2_err
    print(
        f"  sin2(theta_W)_on-shell = {fitted_sin2:.5f} +/- {fitted_sin2_err:.5f}   "
        f"(MG5 truth: {truth['sin2w_onshell']:.5f})"
    )
    print(f"  pull vs. MG5 param_card truth: {pull_sin2:+.2f} sigma")

    plot_afb(
        scan,
        fit_afb_result,
        fit_plain.values["mZ"],
        fit_plain.values["gammaZ"],
        DATA_PROCESSED / "simulated_mumu_mg5_afb.png",
        sin2w_true=truth["sin2w_onshell"],
        mZ_true=truth["mZ"],
        gammaZ_true=truth["gammaZ"],
    )


if __name__ == "__main__":
    main()
