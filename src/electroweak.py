"""
Improved-Born electroweak cross sections for e+e- -> f fbar near the Z
(README section 0.3), shared by the lineshape fits and the A_FB fits.

Conventions: natural units, s = E_cm^2 in GeV^2, theta between the outgoing
fermion and the incoming e-, and neutral-current couplings

    v_f = T3_f - 2 Q_f sin2(theta_W_eff),   a_f = T3_f.

With the photon/Z propagator ratio

    chi(s) = G_F M_Z^2 / (2 sqrt(2) pi alpha) * s / (s - M_Z^2 + i M_Z Gamma_Z)

(fixed width) the differential cross section is

    dsigma/dOmega = alpha^2/(4s) [(1 + cos^2 theta) F1 + cos theta F2],
    F1 = Q_f^2 - 2 Q_f v_e v_f Re chi + (v_e^2 + a_e^2)(v_f^2 + a_f^2) |chi|^2,
    F2 = -4 Q_f a_e a_f Re chi + 8 v_e a_e v_f a_f |chi|^2,

summed over the final-state fermions of a channel. The electron's charge,
Q_e = -1, is absorbed into the signs: Q_f enters the interference terms
linearly. For f = mu this is the form checked to machine precision against
an independent helicity-amplitude calculation (the 2026-09-27 A_FB fix).

alpha: the Z term is alpha-independent (G_F scheme); the interference term
scales as alpha and the photon term as alpha^2. The improved Born
approximation uses alpha(s) ~ alpha(M_Z^2) there, which is what the
lineshape models use. The A_FB fits keep alpha(0), as before, so that the
MadGraph5 numbers in README section 3 stay reproducible; switching would
rescale the interference term in A_FB by 6%.
"""

import numpy as np

ALPHA_0 = 1 / 137.035999
# alpha^-1(M_Z^2) = 128.947 +/- 0.013, Davier et al., arXiv:1908.00921, eq. (9)
ALPHA_MZ = 1 / 128.947
G_FERMI_GEV2 = 1.16637e-5
GEV2_TO_NB = 389379.0  # (hbar*c)^2, CODATA, converted from mb to nb
SIN2_THETAW_EFF = 0.23155

# (electric charge Q, weak isospin T3) of the final-state fermions of each
# channel. The top quark is above threshold. ee is the s-channel part only.
CHANNEL_FERMIONS = {
    "ee": [(-1.0, -0.5)],
    "mumu": [(-1.0, -0.5)],
    "tautau": [(-1.0, -0.5)],
    "qq": [(2 / 3, 0.5)] * 2 + [(-1 / 3, -0.5)] * 3,  # u, c and d, s, b
}
ELECTRON = (-1.0, -0.5)


def couplings(charge: float, t3: float, sin2thetaw: float = SIN2_THETAW_EFF) -> tuple[float, float]:
    """(v_f, a_f) for a fermion of charge Q and weak isospin T3."""
    return t3 - 2 * charge * sin2thetaw, t3


def chi(sqrt_s: np.ndarray, mZ: float, gammaZ: float, alpha: float = ALPHA_0) -> np.ndarray:
    s = np.asarray(sqrt_s, dtype=float) ** 2
    kappa0 = G_FERMI_GEV2 * mZ**2 / (2 * np.sqrt(2) * np.pi * alpha)
    return kappa0 * s / (s - mZ**2 + 1j * mZ * gammaZ)


def f1_f2(
    sqrt_s: np.ndarray,
    mZ: float,
    gammaZ: float,
    channel: str = "mumu",
    sin2thetaw: float = SIN2_THETAW_EFF,
    alpha: float = ALPHA_0,
) -> tuple[np.ndarray, np.ndarray]:
    """Angular coefficients F1, F2 of the module docstring, summed over the
    channel's final-state fermions. Colour and QCD factors are not included:
    they multiply all terms of a quark channel alike and are absorbed by a
    free normalisation."""
    c = chi(sqrt_s, mZ, gammaZ, alpha)
    v_e, a_e = couplings(*ELECTRON, sin2thetaw)
    f1 = 0.0
    f2 = 0.0
    for charge, t3 in CHANNEL_FERMIONS[channel]:
        v_f, a_f = couplings(charge, t3, sin2thetaw)
        f1 = f1 + charge**2 - 2 * charge * v_e * v_f * np.real(c) + (v_e**2 + a_e**2) * (v_f**2 + a_f**2) * np.abs(c) ** 2
        f2 = f2 - 4 * charge * a_e * a_f * np.real(c) + 8 * v_e * a_e * v_f * a_f * np.abs(c) ** 2
    return f1, f2


def born_cross_section_nb(
    sqrt_s: np.ndarray,
    mZ: float,
    gammaZ: float,
    channel: str = "mumu",
    sin2thetaw: float = SIN2_THETAW_EFF,
    alpha: float = ALPHA_MZ,
) -> np.ndarray:
    """Total s-channel cross section, integrating dsigma/dOmega over the
    full solid angle: alpha^2/(4s) * 16 pi/3 * F1, in nb."""
    sqrt_s = np.asarray(sqrt_s, dtype=float)
    f1, _ = f1_f2(sqrt_s, mZ, gammaZ, channel, sin2thetaw, alpha)
    return alpha**2 / (4 * sqrt_s**2) * (16 * np.pi / 3) * f1 * GEV2_TO_NB


def fixed_to_running_width(mZ_fixed: float, gammaZ_fixed: float) -> tuple[float, float]:
    """Convert fixed-width (M, Gamma) to the running-width convention LEP
    quotes. README 0.2: D_run(s; M, G) = k D_fix(s; M/sqrt(k), G/sqrt(k))
    with k = 1 + G^2/M^2, which the rescaling leaves unchanged, so
    M_run = M_fix sqrt(k) and G_run = G_fix sqrt(k): about +34 MeV and
    +1 MeV at the Z."""
    k = 1 + gammaZ_fixed**2 / mZ_fixed**2
    return mZ_fixed * np.sqrt(k), gammaZ_fixed * np.sqrt(k)
