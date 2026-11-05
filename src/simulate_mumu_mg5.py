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

    python src/simulate_mumu_mg5.py          # ISR off (the original test)
    python src/simulate_mumu_mg5.py --isr    # ISR on via MG5's e+e- structure functions

With `--isr` (MadGraph5 >= 3.2.0) the beams are given the leading-log
electron/positron PDFs (`lpp = -/+3`, `pdlabel = isronlyll`), so the
generated lineshape is radiatively distorted the way OPAL's data is, and
the ISR-convolved fit should now be the one that recovers the truth. The
photon-exchange term makes the radiative-return tail diverge without a cut,
so a cut sqrt(s')/sqrt(s) > SPRIME_MIN_FRACTION is applied as a minimum
mu+mu- invariant mass (`mmll`) and the fit's ISR integral is truncated at
the matching x_max = 1 - SPRIME_MIN_FRACTION^2. cos(theta) is evaluated in
the post-radiation e+e- rest frame, read from each event's incoming legs.

The --isr path was written without a MadGraph5 installation to test it
against (2026-11-05); the run_card parameter names follow the 3.5.x
template and MG5 will refuse unknown names, so a mismatch fails loudly.
"""

import argparse
import gzip
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from uncertainties import ufloat

import compute_cross_section as ccs
from compute_cross_section import DATA_PROCESSED, breit_wigner, breit_wigner_isr, fit_breit_wigner, fit_breit_wigner_isr, print_fit_result
from simulate_mumu import SCAN_POINTS_GEV, fit_afb, plot_afb, plot_closure

MG5_EXECUTABLE = Path(
    os.environ.get(
        "MG5_PATH",
        str(Path.home() / "Documents/collider_analysis/MG5_aMC_v3_5_0/bin/mg5_aMC"),
    )
)
PROCESS_DIR = DATA_PROCESSED / "mg5_eemumu"

ISR_ON = False  # set by --isr
SPRIME_MIN_FRACTION = 0.9  # sqrt(s')/sqrt(s) cut used only with ISR on

LUMI_PBINV_MG5 = 0.3  # smaller than simulate_mumu.py's 2.0 pb^-1, to keep each MG5 launch fast
EVENTS_PER_LAUNCH = 1200  # headroom above the on-peak Poisson mean (~600 at L=0.3 pb^-1)
EFFICIENCY_EVENTS = 8000
ACCEPTANCE_COSTHETA_MAX = 0.95  # same cut as simulate_mumu.py, for direct comparability

RNG = np.random.default_rng(20260803)

CROSS_SECTION_RE = re.compile(r"Cross-section\s*:\s*([\d.eE+-]+)\s*\+-\s*[\d.eE+-]+\s*pb")
RUN_NAME_RE = re.compile(r"Results Summary for run:\s*(\S+)\s+tag:")
MASS_LINE_RE = re.compile(r"^\s*(\d+)\s+([\d.eE+-]+)\s*#\s*(\S+)")
DECAY_LINE_RE = re.compile(r"^\s*DECAY\s+(\d+)\s+([\d.eE+-]+)")


MG5_LOG_DIR = DATA_PROCESSED / "mg5_logs"
MG5_TIMEOUT_S = 30 * 60  # per launch; a normal launch here takes ~30 s
_launch_counter = 0


def _run_mg5(commands: list[str]) -> str:
    """Run one MG5 command script. MG5's stdout+stderr are streamed to a
    log file under data/processed/mg5_logs/ (so progress can be followed
    with `tail -f` and a stuck run can be diagnosed), stdin is closed so
    that any question MG5 asks beyond the scripted answers falls through
    to its default instead of waiting for a keyboard that nobody is
    watching, and a timeout turns a hang into an error with the log path.
    Returns the full output text, which the caller parses."""
    global _launch_counter
    if not MG5_EXECUTABLE.exists():
        raise FileNotFoundError(
            f"MadGraph5 executable not found at {MG5_EXECUTABLE}. "
            "Set the MG5_PATH environment variable to your bin/mg5_aMC."
        )
    MG5_LOG_DIR.mkdir(parents=True, exist_ok=True)
    _launch_counter += 1
    log_path = MG5_LOG_DIR / f"mg5_launch_{_launch_counter:02d}.log"
    with tempfile.NamedTemporaryFile("w", suffix=".mg5", delete=False) as f:
        f.write("\n".join(commands) + "\n")
        script_path = Path(f.name)
    print(f"    MG5 output -> {log_path}")
    try:
        with open(log_path, "w") as log:
            log.write("# MG5 command script:\n# " + "\n# ".join(commands) + "\n\n")
            log.flush()
            try:
                subprocess.run(
                    [str(MG5_EXECUTABLE), str(script_path)],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=MG5_TIMEOUT_S,
                    check=True,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"MG5 did not finish within {MG5_TIMEOUT_S / 60:.0f} min; see the end of {log_path}"
                ) from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"MG5 exited with status {exc.returncode}; see {log_path}") from exc
    finally:
        script_path.unlink(missing_ok=True)
    return log_path.read_text()


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
    if ISR_ON:
        # e+ is beam 1 in "generate e+ e- > mu+ mu-"; the sign of lpp follows
        # the beam particle's charge, as for proton (+1) / antiproton (-1).
        beam_settings = [
            "set lpp1 -3",
            "set lpp2 3",
            "set pdlabel isronlyll",
            f"set mmll {SPRIME_MIN_FRACTION * sqrt_s:.4f}",
        ]
    else:
        beam_settings = ["set lpp1 0", "set lpp2 0"]
    stdout = _run_mg5(
        [
            f"launch {PROCESS_DIR}",
            "shower=OFF",
            "detector=OFF",
            "done",
            f"set ebeam1 {ebeam}",
            f"set ebeam2 {ebeam}",
            *beam_settings,
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


def boost_to_rest_frame(p: np.ndarray, p_total: np.ndarray) -> np.ndarray:
    """Lorentz-boost four-vector p (E, px, py, pz) into the rest frame of
    p_total. Identity when p_total has no spatial momentum (lpp=0 case)."""
    m = np.sqrt(max(p_total[0] ** 2 - p_total[1:] @ p_total[1:], 1e-300))
    beta = p_total[1:] / p_total[0]
    b2 = beta @ beta
    if b2 < 1e-20:
        return p.copy()
    gamma = p_total[0] / m
    bp = beta @ p[1:]
    e_new = gamma * (p[0] - bp)
    p_new = p[1:] + ((gamma - 1) * bp / b2 - gamma * p[0]) * beta
    return np.array([e_new, *p_new])


def parse_lhe_costheta(lhe_path: Path) -> np.ndarray:
    """cos(theta) of the outgoing mu- relative to the incoming e- direction,
    the standard A_FB convention, evaluated in the rest frame of the two
    incoming (status -1) legs. The e- direction is read from each event
    rather than assumed, so the result does not depend on which beam MG5 puts
    along +z. With lpp=0 the incoming legs are the beams and the boost is
    the identity; with ISR on they are the post-radiation e+e-, so the boost
    undoes the longitudinal recoil against the radiated photons.

    Until 2026-09-27 this measured theta from the e+ beam, which flips A_FB.
    Combined with a sign error in simulate_mumu.iba_f1_f2 the two flips
    cancelled in F2, and the leftover mismatch in F1 pulled the A_FB fit to
    sin2(theta_W) = 0.25 exactly."""
    with gzip.open(lhe_path, "rt") as f:
        text = f.read()
    costhetas = []
    for block in text.split("<event>")[1:]:
        body = block.split("</event>")[0]
        p_mu = p_em = None
        p_in = np.zeros(4)
        for line in body.splitlines():
            fields = line.split()
            if len(fields) < 10:
                continue
            try:
                pdg, status = int(fields[0]), int(fields[1])
            except ValueError:
                continue
            p = np.array([float(fields[9]), float(fields[6]), float(fields[7]), float(fields[8])])
            if status == -1:
                p_in += p
            if pdg == 11 and status == -1:
                p_em = p
            elif pdg == 13 and status == 1:
                p_mu = p
        if p_mu is not None and p_em is not None:
            mu = boost_to_rest_frame(p_mu, p_in)[1:]
            em = boost_to_rest_frame(p_em, p_in)[1:]
            costhetas.append(float(mu @ em / (np.linalg.norm(mu) * np.linalg.norm(em))))
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
    global ISR_ON
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isr", action="store_true", help="switch on MG5's e+e- ISR structure functions (MG5 >= 3.2.0)")
    args = parser.parse_args()
    ISR_ON = args.isr
    tag = "_isr" if ISR_ON else ""
    if ISR_ON:
        # match the fit's radiator integral to the generator-level sqrt(s') cut
        ccs.ISR_X_MAX = 1 - SPRIME_MIN_FRACTION**2
        print(
            f"ISR ON: lpp=-/+3, pdlabel=isronlyll, sqrt(s')/sqrt(s) > {SPRIME_MIN_FRACTION} "
            f"(fit radiator truncated at x_max = {ccs.ISR_X_MAX:.3f})"
        )

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
    scan.to_csv(DATA_PROCESSED / f"simulated_mumu_mg5{tag}_scan.csv", index=False)
    print(f"\nSaved scan table to {DATA_PROCESSED / f'simulated_mumu_mg5{tag}_scan.csv'}")

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
    if ISR_ON:
        print(
            "  NOTE: ISR is ON in this MG5 run, so the ISR-convolved fit is the physically\n"
            "  correct lineshape model and the naive fit should be biased high, as on the\n"
            "  real OPAL data. The AFB fit below uses this ISR-convolved fit's mZ/gammaZ."
        )
        lineshape_for_afb = fit_isr
        true_model, true_label = breit_wigner_isr, "True generating curve (MG5 with ISR, x_max cut)"
    else:
        print(
            "  NOTE: this MG5 process has ISR/beamstrahlung switched off (see README), so the\n"
            "  *naive* fit above is the physically correct lineshape model here -- convolving in\n"
            "  an ISR radiator that isn't actually present over-corrects and biases mZ/gammaZ\n"
            "  low, the mirror image of the OPAL case where the real data DOES have ISR and the\n"
            "  naive fit is the biased one. The AFB fit below deliberately uses the naive fit's\n"
            "  mZ/gammaZ, not this ISR-convolved one. Re-run with --isr for the other case."
        )
        lineshape_for_afb = fit_plain
        true_model, true_label = breit_wigner, "True generating curve (MG5, no ISR)"

    peak_row = scan.loc[(scan["sqrt_s_GeV"] - truth["mZ"]).abs().idxmin()]
    sigma_peak_true_nb = peak_row["sigma_mg5_true_pb"] / 1000
    if ISR_ON:
        # MG5's on-peak number is the radiated cross section; undo the
        # radiator's on-peak reduction so the true curve is drawn with the
        # born peak height the convolution expects.
        born_over_obs = breit_wigner(truth["mZ"], 1.0, truth["mZ"], truth["gammaZ"]) / breit_wigner_isr(
            truth["mZ"], 1.0, truth["mZ"], truth["gammaZ"]
        )[0]
        sigma_peak_true_nb *= born_over_obs

    plot_closure(
        scan,
        fit_plain,
        fit_isr,
        DATA_PROCESSED / f"simulated_mumu_mg5{tag}_closure.png",
        sigma_peak_true=sigma_peak_true_nb,
        mZ_true=truth["mZ"],
        gammaZ_true=truth["gammaZ"],
        true_model=true_model,
        true_label=true_label,
    )

    which = "ISR-convolved" if ISR_ON else "NAIVE"
    print(f"\n=== A_FB(s) fit (mZ, gammaZ fixed from the {which} lineshape fit -- see note above) ===")
    fit_afb_result = fit_afb(scan, lineshape_for_afb.values["mZ"], lineshape_for_afb.values["gammaZ"])
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
        lineshape_for_afb.values["mZ"],
        lineshape_for_afb.values["gammaZ"],
        DATA_PROCESSED / f"simulated_mumu_mg5{tag}_afb.png",
        sin2w_true=truth["sin2w_onshell"],
        mZ_true=truth["mZ"],
        gammaZ_true=truth["gammaZ"],
    )


if __name__ == "__main__":
    main()
