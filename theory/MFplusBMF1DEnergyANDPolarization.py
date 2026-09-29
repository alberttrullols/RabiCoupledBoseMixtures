#!/usr/bin/env python3
"""
MF + BMF (LHY) energy and P^2 for a coherently coupled 1D two-component BEC
with finite detuning.
Python translation of MFplusBMF1DEnergyANDPolarization_finite_detuning.m

Physical model (m = 1):
  - Equal-mass, equal-intraspecies coupling  g11 = g22 = gp
  - Coherent coupling  Omega = 2*t
  - Detuning  delta  (single-atom spin Hamiltonian: {{-d/2,-O/2},{-O/2,d/2}})
    Positive detuning favours component 1 -> positive polarization.
  - Polarization  P = (n1 - n2) / (n1 + n2)
  - MF transition at  g12_MF = gp + Omega / ntot  (for delta = 0)

Outputs:
  P2vsg12MF.txt      : (g12/g12MF, P_MF^2)
  P2vsg12.txt        : (g12/g12MF, (P_MF + dP_BMF)^2)   thermodynamic limit
  P2vsg12FiniteL.txt : (g12/g12MF, (P_MF + dP_BMF)^2)   finite ring, L = Num/ntot
  Evsg12MF.txt       : (g12/g12MF, E_MF / L)
  Evsg12.txt         : (g12/g12MF, (E_MF + E_LHY) / L)  thermodynamic limit
  Evsg12L.txt        : (g12/g12MF, (E_MF + E_LHY) / L)  finite ring
  bmf_results.png    : plots
"""

import numpy as np
from scipy import integrate, optimize

# ============================================================
# Parameters
# ============================================================
gp     = 0.00005           # intraspecies coupling constant
ntotp  = 1.0            # total density
Omegap = 2.0 * 0.15 * gp  # Omega = 2*t
deltap = 0.0 * Omegap   # detuning (positive -> favours component 1)
g12MF  = gp + Omegap / ntotp  # MF transition point (exact only at delta=0)
g12min = 0.835*g12MF
g12max = 1.3*g12MF
GridSize = 4000

# Finite-size parameters
Num  = 90              # number of particles
L    = Num / ntotp      # ring length
NumK = 10000             # momentum cutoff index
K    = 2.0 * np.pi * np.arange(1, NumK + 1) / L   # discrete momenta k_m = 2pi*m/L

dn   = 0.001 * ntotp     # finite-difference step for numerical derivatives

print(f"Omegap  = {Omegap:.6g}")
print(f"deltap  = {deltap:.6g}")
print(f"g12MF   = {g12MF:.6g}")
print(f"g12 in  [{g12min:.6g}, {g12max:.6g}]")
print(f"L = {L:.2f},  K_max = {K[-1]:.4f}")


# ============================================================
# Core Bogoliubov dispersion function (vectorized in p)
# ============================================================

def lhy_integrand(p, n1, n2, g11, g22, g12, Omega):
    """
    Returns  omega_-(p) + omega_+(p) - p^2 - subtraction_terms

    This is the integrand for the LHY (zero-point) energy density.
    Works for both scalar p and numpy arrays.

    Parameters
    ----------
    p     : momentum (scalar or array)
    n1,n2 : partial densities
    g11,g22,g12 : coupling constants
    Omega : coherent coupling (= 2*tunneling)
    """
    p2h = 0.5 * p * p           # p^2 / 2

    sqrt_n1n2 = np.sqrt(n1 * n2)
    sqrt_r21  = np.sqrt(n2 / n1)  # sqrt(n2/n1)
    sqrt_r12  = np.sqrt(n1 / n2)  # sqrt(n1/n2)

    # Single-particle energies (before interactions, with coherent shift)
    eps1 = p2h + Omega / 2.0 * sqrt_r21
    eps2 = p2h + Omega / 2.0 * sqrt_r12

    # Mean value A  (= average of the two diagonal Bogoliubov blocks)
    A = (0.5 * (eps1 * (eps1 + 2.0*g11*n1) + eps2 * (eps2 + 2.0*g22*n2))
         + Omega/2.0 * (Omega/2.0 - 2.0*g12*sqrt_n1n2))

    # Discriminant D^2
    D_sq = (A*A
            - p2h * (p2h + Omega/2.0*(n1 + n2)/sqrt_n1n2)
            * ((eps1 + 2.0*g11*n1)*(eps2 + 2.0*g22*n2)
               - (Omega/2.0 - 2.0*g12*sqrt_n1n2)**2))

    D_sq   = np.maximum(D_sq, 0.0)   # guard against tiny negative values
    sqrt_D = np.sqrt(D_sq)

    om_m = np.sqrt(np.maximum(A - sqrt_D, 0.0))   # lower Bogoliubov branch
    om_p = np.sqrt(np.maximum(A + sqrt_D, 0.0))   # upper Bogoliubov branch

    # Subtraction terms (remove free-particle + MF contributions, improve UV)
    sub = p*p + (n1 + n2)/sqrt_n1n2 * Omega/2.0 + g11*n1 + g22*n2

    return om_m + om_p - sub


# ============================================================
# LHY energy in the thermodynamic limit
# ============================================================

def elhy_inf(n1, n2, g11, g22, g12, Omega):
    """
    E_LHY / L in the thermodynamic limit:
        integral_0^{K_max} dp / (2*pi) * lhy_integrand(p)
    (Factor 1/(2*pi) is the 1D density of states; the factor-of-2
    from +-p symmetry and the 1/2 zero-point prefactor cancel.)
    """
    val, _ = integrate.quad(
        lambda p: lhy_integrand(p, n1, n2, g11, g22, g12, Omega),
        0.0, K[-1],
        limit=200
    )
    return val / (2.0 * np.pi)


# ============================================================
# LHY energy for finite ring (periodic BC)
# ============================================================

def elhy_finite_sum(n1, n2, g11, g22, g12, Omega):
    """
    LHY sum for finite ring before dividing by L:
        f(k=0)/2  +  sum_{m=1}^{NumK} f(K_m)

    The k=0 mode is counted once (zero-point: 1/2 * omega), while
    each k_m = 2*pi*m/L with m >= 1 contributes once (representing
    the +-k_m pair with the 1/2 zero-point factor absorbed: 2 * 1/2 = 1).
    Divide by L afterwards to get the energy density.
    """
    f0 = lhy_integrand(0.0, n1, n2, g11, g22, g12, Omega)
    fK = lhy_integrand(K,   n1, n2, g11, g22, g12, Omega)
    return 0.5 * f0 + np.sum(fK)


# ============================================================
# MF compressibility matrix  d mu_i / d n_j   (analytic, g11=g22=g)
# ============================================================

def musurn_matrix(n1, n2, g, g12, Omega):
    """2x2 matrix of second derivatives of E_MF w.r.t. n1, n2."""
    sqrt_n1n2 = np.sqrt(n1 * n2)
    off = g12 - Omega / 4.0 / sqrt_n1n2
    return np.array([
        [g + Omega/4.0 * np.sqrt(n2) / n1**1.5,  off],
        [off,   g + Omega/4.0 * np.sqrt(n1) / n2**1.5]
    ])


# ============================================================
# Polarization correction formula
# ============================================================

def compute_dP(Esurn, nsurmu, ntot):
    """
    BMF correction to the polarization (matches Mathematica sign convention):
        dP = 2 * (sum_row2*(-E1*N00 - E2*N10) - sum_row1*(-E1*N01 - E2*N11))
             / sum_all / ntot
    where N = nsurmu = (d mu / d n)^{-1}, Esurn = d E_LHY / d n_i.
    The negative sign on Esurn reflects that if LHY energy cost increases with n_i,
    the system lowers n_i, reducing the polarization imbalance.
    """
    sr0 = nsurmu[0, 0] + nsurmu[0, 1]   # sum of row 0  (= Total[N[[1]]] in Mma)
    sr1 = nsurmu[1, 0] + nsurmu[1, 1]   # sum of row 1  (= Total[N[[2]]] in Mma)
    sa  = sr0 + sr1                       # sum of all elements
    return (
        2.0 * (
            sr1 * (-Esurn[0]*nsurmu[0, 0] - Esurn[1]*nsurmu[1, 0])
            - sr0 * (-Esurn[0]*nsurmu[0, 1] - Esurn[1]*nsurmu[1, 1])
        ) / sa / ntot
    )


# ============================================================
# Quantum polarization fluctuations from the k=0 upper BdG mode
# ============================================================

def compute_P2Q(n1, n2, g11, g22, g12, Omega, L, ntot):
    """
    Mean-square quantum fluctuation of the polarization from the p=0
    upper Bogoliubov-de Gennes mode in a finite ring of length L:

        <delta P^2> = 4 * N1 * (u1 + v1)^2
                      / ( (u1^2 + u2^2 - v1^2 - v2^2) * Ntot^2 )

    where (u1, u2, v1, v2) is the BdG eigenvector at k=0 for the upper
    branch (omega_+), obtained by diagonalising the 4x4 BdG matrix:

        M = [[ te1,  cpl,   g11*n1,  g12*c  ],
             [ cpl,  te2,   g12*c,   g22*n2 ],
             [-g11*n1, -g12*c, -te1, -cpl   ],
             [-g12*c, -g22*n2, -cpl, -te2   ]]

    where te_i = eps_i(k=0) + g_ii*n_i  and  cpl = g12*sqrt(n1*n2) - Omega/2.

    Also returns the spin-down check quantity P2Q_down.
    """
    c   = np.sqrt(n1 * n2)
    e1  = Omega / 2.0 * np.sqrt(n2 / n1)   # eps1 at k=0
    e2  = Omega / 2.0 * np.sqrt(n1 / n2)   # eps2 at k=0
    te1 = e1 + g11 * n1
    te2 = e2 + g22 * n2
    cpl = g12 * c - Omega / 2.0             # normal-sector off-diagonal

    M = np.array([
        [ te1,    cpl,     g11*n1,  g12*c  ],
        [ cpl,    te2,     g12*c,   g22*n2 ],
        [-g11*n1, -g12*c,  -te1,    -cpl   ],
        [-g12*c,  -g22*n2, -cpl,    -te2   ],
    ], dtype=float)

    evals, evecs = np.linalg.eig(M)
    evals = evals.real
    evecs = evecs.real

    # Upper branch: eigenvector for the largest positive eigenvalue (omega_+)
    idx = np.argmax(evals)
    ev  = evecs[:, idx]          # components (u1, u2, v1, v2)

    u1, u2, v1, v2 = ev
    bdg_norm = u1**2 + u2**2 - v1**2 - v2**2

    N1   = n1 * L
    N2   = n2 * L
    Ntot = ntot * L

    P2Q_up   = 4.0 * N1 * (u1 + v1)**2 / (bdg_norm * Ntot**2)
    P2Q_down = 4.0 * N2 * (u2 + v2)**2 / (bdg_norm * Ntot**2)  # cross-check
    return P2Q_up, P2Q_down


# ============================================================
# Hellmann-Feynman helper: energy density at arbitrary detuning
# ============================================================

def _energy_at_delta(g12p, delta, P1_guess):
    """
    Solve the MF polarization at the given delta, then return
    (E_MF, E_MF + ELHY_inf, E_MF + ELHY_L/L).
    Used exclusively for the Hellmann-Feynman numerical derivative.
    """
    if delta == 0.0:
        P1 = (0.0 if g12p < g12MF else
              2.0*np.sqrt(max(ntotp**2/4.0 - Omegap**2/4.0/(g12p-gp)**2, 0.0))/ntotp)
    else:
        def _eq(x):
            return (gp - g12p + Omegap/ntotp/np.sqrt(max(1.0-x**2, 1e-14)))*ntotp*x - delta
        P1 = float(np.clip(optimize.fsolve(_eq, np.clip(P1_guess, 0.0, 0.9999))[0], 0.0, 0.9999))

    alpha = np.sqrt((1.0+P1)/(1.0-P1)) if P1 < 1.0-1e-12 else 1e12
    E_MF_val = (
        (delta*(1.0-alpha**2) - 2.0*Omegap*alpha) / (2.0*(1.0+alpha**2)) * ntotp
        + (gp*alpha**4 + gp + 2.0*g12p*alpha**2) / (1.0+alpha**2)**2 * ntotp**2/2.0
    )
    n1_ = (1.0+P1)/2.0 * ntotp
    n2_ = (1.0-P1)/2.0 * ntotp
    ei  = elhy_inf(n1_, n2_, gp, gp, g12p, Omegap)
    eL  = elhy_finite_sum(n1_, n2_, gp, gp, g12p, Omegap) / L
    return E_MF_val, E_MF_val + ei, E_MF_val + eL


# ============================================================
# Main scan over g12
# ============================================================
g12_list = [g12min + i * (g12max - g12min) / GridSize for i in range(GridSize)]

P2vsg12MF_list      = []
P2vsg12_list        = []
P2vsg12FiniteL_list = []
P2Qvsg12_list       = []
P2Qvsg12check_list  = []
Evsg12MF_list       = []
Evsg12_list         = []
Evsg12L_list        = []
PHF_MF_list         = []
PHF_inf_list        = []
PHF_L_list          = []

dh_hf = 1e-5   # step size for the 5-point HF stencil

for g12p in g12_list:
    print(f"  g12p = {g12p:.6f}  (target {g12max:.6f})")

    # ----------------------------------------------------------
    # MF polarization
    # ----------------------------------------------------------
    if deltap == 0.0:
        if g12p < g12MF:
            P1 = 0.0
        else:
            arg = ntotp**2 / 4.0 - Omegap**2 / 4.0 / (g12p - gp)**2
            P1  = 2.0 * np.sqrt(max(arg, 0.0)) / ntotp
    else:
        # Solve: deltap = (gp - g12p + Omegap/ntotp/sqrt(1-x^2)) * ntotp * x
        # Initial guess: 0.5 for first point; (P^2_prev)^(1/4) thereafter
        # (power 1/4 keeps the guess slightly above the solution, aiding convergence)
        if len(P2vsg12MF_list) == 0:
            P1init = 0.5
        else:
            P1init = P2vsg12MF_list[-1][1] ** 0.25
        def _mf_eq(x):
            return (gp - g12p + Omegap / ntotp / np.sqrt(1.0 - x**2)) * ntotp * x - deltap
        P1 = float(optimize.fsolve(_mf_eq, P1init)[0])

    P2vsg12MF_list.append((g12p / g12MF, P1**2))

    # ----------------------------------------------------------
    # MF energy density  (g11 = g22 = gp, finite detuning deltap)
    # ----------------------------------------------------------
    alpha = np.sqrt((1.0 + P1) / (1.0 - P1)) if P1 < 1.0 - 1e-12 else 1e12
    E_MF = (
        (deltap*(1.0 - alpha**2) - 2.0*Omegap*alpha) / (2.0*(1.0 + alpha**2)) * ntotp
        + (gp*alpha**4 + gp + 2.0*g12p*alpha**2) / (1.0 + alpha**2)**2 * ntotp**2 / 2.0
    )
    Evsg12MF_list.append((g12p / g12MF, E_MF))

    n1 = (1.0 + P1) / 2.0 * ntotp
    n2 = (1.0 - P1) / 2.0 * ntotp

    # ----------------------------------------------------------
    # LHY energy correction – thermodynamic limit
    # ----------------------------------------------------------
    ELHY = elhy_inf(n1, n2, gp, gp, g12p, Omegap)
    Evsg12_list.append((g12p / g12MF, E_MF + ELHY))

    # ----------------------------------------------------------
    # LHY energy correction – finite ring
    # ----------------------------------------------------------
    ELHY_L = elhy_finite_sum(n1, n2, gp, gp, g12p, Omegap) / L
    Evsg12L_list.append((g12p / g12MF, E_MF + ELHY_L))

    # ----------------------------------------------------------
    # Compressibility matrices (same for both corrections)
    # ----------------------------------------------------------
    mu_mat = musurn_matrix(n1, n2, gp, g12p, Omegap)
    nsurmu = np.linalg.pinv(mu_mat)

    # ----------------------------------------------------------
    # Polarization BMF correction – thermodynamic limit
    # ----------------------------------------------------------
    e1p = elhy_inf(n1 + dn, n2,      gp, gp, g12p, Omegap)
    e1m = elhy_inf(n1 - dn, n2,      gp, gp, g12p, Omegap)
    e2p = elhy_inf(n1,      n2 + dn, gp, gp, g12p, Omegap)
    e2m = elhy_inf(n1,      n2 - dn, gp, gp, g12p, Omegap)

    Esurn_inf = np.array([(e1p - e1m) / (2.0*dn),
                          (e2p - e2m) / (2.0*dn)])
    dP_inf = compute_dP(Esurn_inf, nsurmu, ntotp)
    P2vsg12_list.append((g12p / g12MF, (P1 + dP_inf)**2))

    # ----------------------------------------------------------
    # Polarization BMF correction – finite ring
    # ----------------------------------------------------------
    s1p = elhy_finite_sum(n1 + dn, n2,      gp, gp, g12p, Omegap)
    s1m = elhy_finite_sum(n1 - dn, n2,      gp, gp, g12p, Omegap)
    s2p = elhy_finite_sum(n1,      n2 + dn, gp, gp, g12p, Omegap)
    s2m = elhy_finite_sum(n1,      n2 - dn, gp, gp, g12p, Omegap)

    Esurn_L = np.array([(s1p - s1m) / (2.0*dn) / L,
                        (s2p - s2m) / (2.0*dn) / L])
    dP_L = compute_dP(Esurn_L, nsurmu, ntotp)
    P2vsg12FiniteL_list.append((g12p / g12MF, (P1 + dP_L)**2))

    # ----------------------------------------------------------
    # Quantum polarization fluctuations (finite-size, k=0 upper BdG mode)
    # ----------------------------------------------------------
    P2Q, P2Qcheck = compute_P2Q(n1, n2, gp, gp, g12p, Omegap, L, ntotp)
    P2Qvsg12_list.append((g12p / g12MF, P2Q))
    P2Qvsg12check_list.append((g12p / g12MF, P2Qcheck))

    # ----------------------------------------------------------
    # Hellmann-Feynman polarization: P = -2/ntot * dE/d(deltap)
    # 5-point central stencil: f' ≈ (-f2 + 8f1 - 8fm1 + fm2) / 12h
    # ----------------------------------------------------------
    Em2_mf,  Em2_inf,  Em2_L  = _energy_at_delta(g12p, deltap - 2*dh_hf, P1)
    Em1_mf,  Em1_inf,  Em1_L  = _energy_at_delta(g12p, deltap -   dh_hf, P1)
    Ep1_mf,  Ep1_inf,  Ep1_L  = _energy_at_delta(g12p, deltap +   dh_hf, P1)
    Ep2_mf,  Ep2_inf,  Ep2_L  = _energy_at_delta(g12p, deltap + 2*dh_hf, P1)

    def _stencil5(Em2, Em1, Ep1, Ep2):
        return (-Ep2 + 8*Ep1 - 8*Em1 + Em2) / (12.0 * dh_hf)

    prefac = -2.0 / ntotp
    PHF_MF_list.append( (g12p/g12MF, prefac * _stencil5(Em2_mf,  Em1_mf,  Ep1_mf,  Ep2_mf )))
    PHF_inf_list.append((g12p/g12MF, prefac * _stencil5(Em2_inf, Em1_inf, Ep1_inf, Ep2_inf)))
    PHF_L_list.append(  (g12p/g12MF, prefac * _stencil5(Em2_L,   Em1_L,   Ep1_L,   Ep2_L  )))


# ============================================================
# Convert to arrays and save
# ============================================================
P2vsg12MF      = np.array(P2vsg12MF_list)
P2vsg12        = np.array(P2vsg12_list)
P2vsg12FiniteL = np.array(P2vsg12FiniteL_list)
P2Qvsg12       = np.array(P2Qvsg12_list)
P2Qvsg12check  = np.array(P2Qvsg12check_list)
Evsg12MF       = np.array(Evsg12MF_list)
Evsg12         = np.array(Evsg12_list)
Evsg12L        = np.array(Evsg12L_list)
PHF_MF         = np.array(PHF_MF_list)
PHF_inf        = np.array(PHF_inf_list)
PHF_L          = np.array(PHF_L_list)

import os
outdir = os.path.dirname(os.path.abspath(__file__))

np.savetxt(os.path.join(outdir, "P2vsg12MF.txt"),
           P2vsg12MF,      header="g12/g12MF  P^2_MF")
np.savetxt(os.path.join(outdir, "P2vsg12.txt"),
           P2vsg12,        header="g12/g12MF  P^2_MF+BMF_inf")
np.savetxt(os.path.join(outdir, "P2vsg12FiniteL.txt"),
           P2vsg12FiniteL, header="g12/g12MF  P^2_MF+BMF_finiteL")
np.savetxt(os.path.join(outdir, "P2Qvsg12.txt"),
           P2Qvsg12,       header="g12/g12MF  <dP^2>_quantum (spin-up)")
np.savetxt(os.path.join(outdir, "P2Qvsg12check.txt"),
           P2Qvsg12check,  header="g12/g12MF  <dP^2>_quantum (spin-down check)")
np.savetxt(os.path.join(outdir, "Evsg12MF.txt"),
           Evsg12MF,       header="g12/g12MF  E_MF/L")
np.savetxt(os.path.join(outdir, "Evsg12.txt"),
           Evsg12,         header="g12/g12MF  (E_MF+E_LHY)/L_inf")
np.savetxt(os.path.join(outdir, "Evsg12L.txt"),
           Evsg12L,        header="g12/g12MF  (E_MF+E_LHY)/L_finiteL")
np.savetxt(os.path.join(outdir, "PHF_L.txt"),
           PHF_L,          header="g12/g12MF  P_HF (MF+BMF finiteL, Hellmann-Feynman)")

print("\nAll output files saved to", outdir)


