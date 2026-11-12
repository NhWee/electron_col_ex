#!/usr/bin/env bash
# Run the whole analysis in the order the README presents it, with a banner
# before each step saying what it does. Safe to re-run; everything it writes
# goes to data/processed/ (gitignored).
#
#   ./run_all.sh              # tests + all MG5-free steps (~10 min: pull study 5, extensions 4)
#   N_TOYS=50 ./run_all.sh    # quicker pull study
#   MG5_PATH=/path/to/bin/mg5_aMC ./run_all.sh   # also runs the MadGraph5 closure tests
#
# See docs/WALKTHROUGH.md for what each step does and what to look for.

set -euo pipefail
cd "$(dirname "$0")"

N_TOYS="${N_TOYS:-500}"

banner() {
    printf '\n\n============================================================\n'
    printf ' STEP %s: %s\n' "$1" "$2"
    printf '============================================================\n'
    printf ' %s\n\n' "$3"
}

banner 0 "Regression tests" \
"Checks the formulas (radiator normalisation, propagator identities, A_FB
 sign), that the fits recover a known truth on noiseless data, and that the
 hand-typed OPAL tables still reproduce the published cross sections."
python -m pytest

banner 1 "Real OPAL data  (src/compute_cross_section.py)" \
"Computes sigma = f N / L for 7 energy points, checks them against OPAL's
 published values, fits M_Z and Gamma_Z with and without ISR, and prints the
 error budget with the complete-model result (photon exchange included, LEP
 running-width convention) at its end."
python src/compute_cross_section.py

banner 2 "Toy Monte Carlo closure test  (src/simulate_mumu.py)" \
"Generates a fake experiment from a truth model where M_Z, Gamma_Z and
 sin2(theta_W) are known exactly, then runs the SAME fit code on it. If the
 fits return the inputs, the pipeline is not just tuned to land on PDG."
python src/simulate_mumu.py

banner 3 "Pull study: the toy repeated ${N_TOYS} times  (src/pull_study.py)" \
"One toy can only say 'close this time'. Many toys test whether the fit is
 unbiased (mean pull 0) and whether its error bars are honest (pull width 1)."
python src/pull_study.py --n-toys "$N_TOYS"

banner 4 "Extensions  (src/extended_fits.py)" \
"Adds six near-peak points with borrowed correction factors, compares three
 lineshape models (propagator conventions, photon exchange), and fits qq,
 mumu and tautau jointly with a common M_Z and Gamma_Z, with a Breit-Wigner
 and with the full photon + Z + interference cross section."
python src/extended_fits.py

if [ -n "${MG5_PATH:-}" ]; then
    banner 5 "MadGraph5 closure test, ISR off  (src/simulate_mumu_mg5.py)" \
"Same closure test, but the events come from an independent matrix-element
 generator. With ISR off the NAIVE fit should win and the ISR fit be biased
 low -- the mirror image of the real data."
    python src/simulate_mumu_mg5.py

    banner 6 "MadGraph5 closure test, ISR on  (src/simulate_mumu_mg5.py --isr)" \
"MG5's own electron structure functions radiate before the collision. Now the
 ISR-convolved fit should recover the truth and the naive fit be biased high,
 as on OPAL data. This is the test the toy MC cannot provide."
    python src/simulate_mumu_mg5.py --isr
else
    printf '\n(Steps 5-6, the MadGraph5 closure tests, skipped: set MG5_PATH to your bin/mg5_aMC to run them.)\n'
fi

printf '\nDone. Tables and plots are in data/processed/. Compare the numbers with README.md.\n'
printf 'make figures copies the plots README.md embeds into figures/.\n'
