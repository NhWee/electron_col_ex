# e+e- -> f fbar Cross-Section Extraction (OPAL Zedometry)

Computes e+e- -> qq / e+e- / mu+mu- / tau+tau- cross sections directly from
raw event counts and integrated luminosity, then validates the result
against the published measurement. The point of the project is the *first*
part: cross sections are derived from the same raw ingredients the original
experiment used, not read off a table that already contains the final
answer.

Two scripts:
- `src/compute_cross_section.py` -- real OPAL data, described below.
- `src/simulate_mumu.py` -- a simulation-based closure test: generates a toy
  Monte Carlo sample from a known truth model and checks whether the same
  fit pipeline can recover it, addressing a fair critique that even the
  real-data script above still consumes numbers OPAL had already computed.
  See [Simulation-based closure test](#simulation-based-closure-test-srcsimulate_mumupy)
  below.

## Why raw counts, not a published table

The obvious shortcut is to grab a cross-section table from HEPData and call
it done -- but HEPData tables are almost always the *final* published
result. There is nothing left to compute. This project instead uses OPAL's
"Zedometry" paper (Abbiendi et al., Eur.Phys.J.C19:587-651, 2001,
[hep-ex/0012018](https://arxiv.org/abs/hep-ex/0012018)), which tabulates the
actual inputs to the measurement: selected event counts (N), integrated
luminosity (L), and efficiency/background correction factors (f) per energy
point and final state. From those,

```
sigma = f * N / L
```

is computed here from scratch and compared against OPAL's own published
cross sections as a check.

## Data (`data/raw/`)

All transcribed by hand from the paper's tables (page-checked against the
PDF, also kept in this folder for provenance).

| File | Paper table | Contents |
|---|---|---|
| `opal_zedometry_table1_event_counts.csv` | Table 1 | N (qq/ee/mumu/tautau) and L per energy point, 1990-1995 (29 rows) -- the actual raw input |
| `opal_zedometry_table3/5/6/7_*_corrections.csv` | Tables 3, 5, 6, 7 | Signal/background correction factor `f` and its systematic error, one file per final state. Only tabulated for 7 points: 1993/1995 peak-2/peak/peak+2 and 1994 peak |
| `opal_zedometry_table8/9/10/11_*_xsec_published.csv` | Tables 8-11 | OPAL's own published cross sections, used only to validate the computed values -- never as calculation input |
| `HEPData-ins1808875-v1-Table_1.csv` | -- | BESIII e+e- -> mu+mu- cross section table (arXiv:2007.12872), an earlier dead end -- this is a *final* published result, kept only for historical reference |
| `opal_zedometry_hep-ex-0012018.pdf` | -- | Source paper, for provenance (not tracked in git) |

Because correction factors only exist for 7 of the 29 tabulated points, that
is also all the script currently computes.

## What the script does (`src/compute_cross_section.py`)

1. Loads N + L (Table 1) and the correction factors for the 7 available
   points, computes `sigma = f * N / L` in nb with Poisson (`sqrt(N)`) and
   systematic (correction-factor) error propagated via the `uncertainties`
   package.
2. Validates the mu+mu- channel against OPAL's published values.
3. Runs the same computation for all four final states (qq, ee, mu+mu-,
   tau+tau-) and pivots the percent differences into one table, since all
   four channels share the same raw luminosity -- a discrepancy isolated to
   one or two channels points at a channel-specific systematic rather than
   a bad L or a transcription error.
4. Fits a Breit-Wigner resonance curve to the computed mu+mu- points via
   `iminuit` to extract M_Z, Gamma_Z, and the peak cross section -- both a
   plain fit and one with the Breit-Wigner convolved against a leading-log
   QED initial-state-radiation radiator (numerically integrated via
   `scipy.integrate.quad`).
5. Plots computed vs. published mu+mu- cross section with the ISR-convolved
   fit overlaid.

## What we found

- 6 of 7 mu+mu- points match OPAL's published values within ~1%. The 1993
  "peak" point is off by -7.5%.
- Extending the cross-channel check to all four final states at that same
  point: only mu+mu- and tau+tau- (track-based selections) show the
  anomaly; qq and ee (calorimeter-based) are fine. The paper itself notes
  that Table 1's luminosity is nominally the qq-selection value and can
  differ for other channels "by about 1%" -- for this run period it
  evidently differed by much more, most likely a tracking-subdetector
  livetime gap the calorimeter-based channels didn't share.
- A first Breit-Wigner fit to all 7 mu+mu- points is dominated by that
  outlier (chi2/ndof = 76/4). Excluding it gives a near-perfect fit
  (chi2/ndof = 0.52/3), but M_Z and Gamma_Z still land ~100-300 MeV off the
  PDG values (91.46 / 2.80 GeV vs. PDG's 91.19 / 2.50 GeV).
- Convolving that same Breit-Wigner against a leading-log QED
  initial-state-radiation radiator (see "ISR-convolved fit" below) and
  refitting the 6 clean points shifts M_Z and Gamma_Z down by ~280 MeV each,
  landing at 91.18 GeV and 2.52 GeV -- within 1 sigma of PDG's M_Z and
  ~1.2 sigma of Gamma_Z. Most of the naive fit's bias was the missing ISR
  treatment, not the 3-cluster sparsity.

### ISR-convolved fit

Radiation of a photon before annihilation always lowers the effective
collision energy, so the *observed* cross section at a given nominal
sqrt(s) is a mix of the true resonance shape at nearby lower energies:

```
sigma_obs(sqrt_s) = integral_0^1 H(x) * sigma_born(sqrt_s * sqrt(1-x)) dx
```

`H(x)` is the leading-log structure function `beta * x^(beta-1) * (1-x/2)`,
with `beta = (2*alpha/pi) * (ln(s/m_e^2) - 1)` -- the standard
Kuraev-Fadin-style exponentiated-soft-photon treatment, truncated at O(alpha)
hard-photon terms. It is not the full ZFITTER-level radiator LEP itself
used, but it captures the right direction and roughly the right size of the
effect from just 6 data points.

## Simulation-based closure test (`src/simulate_mumu.py`)

Everything above still ultimately consumes OPAL's own N, L and correction-
factor (`f`) tables -- real arithmetic and real error propagation, but every
input number was already OPAL's own analysis output. This script instead
generates a toy Monte Carlo sample from a *known* truth model, reconstructs
sigma(sqrt_s) with purely statistical error bars from the simulated event
counts, and reuses the exact same fit functions (`fit_breit_wigner`,
`fit_breit_wigner_isr`, imported unmodified from `compute_cross_section.py`)
to check whether they recover the exact input truth -- a real closure test
of the pipeline, not just "close to PDG."

- **Truth model**: mZ and gammaZ fixed to the PDG values; the peak cross
  section is *derived*, not guessed, from the standard resonance formula
  `sigma_peak = (12*pi/M_Z^2) * Gamma_ee*Gamma_mumu / Gamma_Z^2` using PDG's
  leptonic partial width -- comes out to 1.9997 nb, matching the ~2.0 nb LEP
  actually measured.
- **Angular distribution**: events are drawn from the standard spin-1-
  exchange shape `(1+cos^2 theta) + (8/3)*A_FB(s)*cos(theta)`. `A_FB(s)` is
  the real energy-dependent gamma/Z interference asymmetry (see "Forward-
  backward asymmetry" below), not a constant approximation -- the acceptance
  cut below is symmetric in cos(theta), so it has zero effect on the
  cross-section measurement, but the per-event angular info is real and is
  used for a genuine second observable extraction.
- **Toy detector acceptance**, mirroring OPAL's own `f` factor: a dedicated
  500,000-event simulation sample gives `A_hat = 0.9270 +/- 0.0004` for a
  `|cos(theta)| < 0.95` cut. Every scan point's cross section is corrected by
  this *same* A_hat, derived once from simulation and never from the point's
  own data -- structurally identical to how OPAL's own correction tables are
  used above, except computed here instead of read from a paper.
- **Scan**: 13 points from mZ-3 to mZ+3 GeV in 0.5 GeV steps (vs. only 3
  distinct clusters available in the real OPAL data), 2 pb^-1 toy luminosity
  per point.

### Result

- Naive (no-ISR) fit: chi2/ndof = 67.46/10, mZ = 91.4205 +/- 0.0145 GeV,
  gammaZ = 2.9188 +/- 0.0431 GeV -- **+16.1 sigma / +9.8 sigma** away from the
  literal true input. With 13 well-separated points instead of 3 clusters,
  the missing-ISR bias is no longer hideable inside a good chi2/ndof the way
  it nearly was in the real-data fit -- it shows up as a bad fit outright.
- ISR-convolved fit: chi2/ndof = 5.82/10, mZ = 91.1701 +/- 0.0144 GeV,
  gammaZ = 2.4808 +/- 0.0404 GeV -- **-1.21 sigma / -0.36 sigma** from the true
  input. The same ISR treatment that only got the real-data fit to within
  ~1 sigma of PDG here recovers the *exact* generating parameters to well
  within its own statistical uncertainty.

This is the actual point of the exercise: it demonstrates the fitting
pipeline is self-consistent and correctly unfolds the ISR effect it was
built to unfold, using data this script generated itself rather than
borrowing OPAL's.

### Known limitations

- The closure test validates *self-consistency* of the ISR treatment, not
  agreement with a fully independent higher-order QED calculation -- the
  same truncated leading-log radiator is used both to generate the toy
  sample and to fit it, so a shared blind spot in that radiator would not
  show up here.
- Constant toy luminosity and a flat |cos(theta)| acceptance cut, not real
  detector geometry or a realistic trigger/reconstruction efficiency curve.

### Forward-backward asymmetry (A_FB)

The mZ/gammaZ fit above says nothing about the weak mixing angle. A_FB does:
its energy dependence comes from gamma/Z interference, and its shape is
sensitive to sin2(theta_W_eff) -- the second classic LEP Z-lineshape
observable, extracted here as a genuinely new quantity (not fed in from
OPAL or PDG in any way).

Standard "improved Born approximation" formula (the same formalism behind
LEP's own AFB fits):

```
dsigma/dOmega ~ (1+cos^2 theta)*F1(s) + cos(theta)*F2(s)
A_FB(s) = (3/8) * F2(s) / F1(s)
```

`F1`, `F2` are built from the photon/Z propagator ratio
`chi(s) = kappa0 * s / (s - M_Z^2 + i*M_Z*Gamma_Z)` and standard lepton
vector/axial couplings (`v = -1/2 + 2*sin2(theta_W_eff)`, `a = -1/2`). This
is a *different* approximation than the non-relativistic fixed-width
`breit_wigner()` used for the rate/lineshape fit above -- two different
approximations for two different observables in the same script, not a bug.

Two independent sanity checks are printed every run, before trusting
anything downstream:
- `Gamma_ee` computed from the same couplings via
  `G_F*M_Z^3/(6*sqrt(2)*pi) * (v^2+a^2)` gives 83.39 MeV, matching PDG's
  ~84 MeV to <1%.
- Integrating `F1(s)` at `s=M_Z^2` over solid angle gives sigma_peak =
  1.9818 nb, matching the independently-derived `TRUE_SIGMA_PEAK_NB` =
  1.9997 nb to <1%. Two unrelated formulas agreeing is real evidence the
  interference formula's normalization is right, not just plausible-looking.

Each scan point's accepted events are split by cos(theta) sign into
forward/backward counts, giving `A_FB_measured(s) = (F-B)/(F+B)` with the
standard asymptotic error `sqrt((1-A_FB^2)/N)`. A second fit
(`fit_afb`) then holds mZ and gammaZ fixed at the values already recovered
from the ISR-convolved lineshape fit and fits `sin2(theta_W_eff)` as the one
free parameter -- mirroring the real two-step LEP procedure (lineshape
fit first, then AFB fit at fixed mZ/gammaZ).

**Result**: chi2/ndof = 7.50/12, sin2(theta_W_eff) = 0.22596 +/- 0.00343
vs. the true input 0.23155 -- **-1.63 sigma**. The measured A_FB(s) points
trace a clear sign flip across the Z peak (positive below, negative above),
and the fitted curve tracks the true generating curve closely -- see
`data/processed/simulated_mumu_afb.png`.

**Known limitation**: only the leptonic couplings/A_FB are modeled; a real
sin2(theta_W_eff) extraction at LEP combined multiple final states and used
much higher statistics per point. This closure test uses 13 points at
2 pb^-1 each, so -1.63 sigma is a perfectly normal statistical fluctuation,
not a claim of MeV-level precision.

## Setup

```
python3 -m venv .venv   # or: pip install --user virtualenv && python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```
python src/compute_cross_section.py   # real OPAL data: compute, validate, fit
python src/simulate_mumu.py           # simulation-based closure test
```

Both print their results to stdout and save tables/plots to
`data/processed/` (gitignored -- regenerate by running the scripts).

## Known limitations / next steps

- The ISR radiator used is a truncated leading-log approximation, not the
  full treatment (multi-photon exponentiation beyond leading log, exact
  O(alpha^2) terms) that real LEP electroweak fits use -- adequate to show
  the effect exists and roughly how big it is, not to claim LEP-level
  precision from 6 points.
- The fit is constrained by only 3 distinct energy points (peak-2/peak/
  peak+2); the other scanned points (peak-3/-1/+1/+3, prescan) exist in
  Table 1 but have no published correction factors to compute a trustworthy
  cross section from. More independent points would make the good chi2/ndof
  mean something stronger than "3 parameters matched 3 clusters."
- Only the mu+mu- channel has a full compute -> validate -> fit pipeline;
  qq/ee/tau+tau- are currently used only for the cross-channel consistency
  check.
- `simulate_mumu.py` now extracts both mZ/gammaZ (lineshape) and
  sin2(theta_W_eff) (forward-backward asymmetry) -- the two classic LEP
  Z-pole observables. A natural further step is combining multiple final
  states and/or a running-width Z propagator for a more realistic
  precision-electroweak-style joint fit, rather than the single leptonic
  channel and fixed-width propagator used here.
