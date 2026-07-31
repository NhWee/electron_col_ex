# e+e- -> f fbar Cross-Section Extraction (OPAL Zedometry)

Computes e+e- -> qq / e+e- / mu+mu- / tau+tau- cross sections directly from
raw event counts and integrated luminosity, then validates the result
against the published measurement. The point of the project is the *first*
part: cross sections are derived from the same raw ingredients the original
experiment used, not read off a table that already contains the final
answer.

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

## Setup

```
python3 -m venv .venv   # or: pip install --user virtualenv && python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```
python src/compute_cross_section.py
```

Prints the computed vs. published mu+mu- table, the Breit-Wigner fit
results, and the four-channel consistency check; saves outputs to
`data/processed/` (gitignored -- regenerate by running the script).

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
