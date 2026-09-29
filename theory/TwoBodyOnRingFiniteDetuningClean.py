from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Parameters (matching the Mathematica defaults) ─────────────────────────
g     = 10.
Omega = 2 * (0.023)    # Rabi frequency  Ω = 2t
Delta = 1.0 * (0.000)          # detuning: single-atom Hamiltonian {{-Δ/2,-Ω/2},{-Ω/2,Δ/2}}
                            # DMC uses h = Delta/2, so Delta = 2*h_DMC = 2*0.003 = 0.006
                            # positive Δ favours component 1
L     = 2.0                 # ring circumference
ntot  = 2.0 / L             # density for two atoms
g12MF = g + Omega    # mean-field crossover interaction

Emin = -2.0                 # starting energy (measured from the free 2-body threshold)
dE   = 0.00001                 # energy step

g12_min =  -9.5            # g12/g12MF window to collect  (-np.inf = no lower cut)
g12_max = 15.       # g12/g12MF window to collect  ( np.inf = no upper cut)


# ── Helper: analytically-continued  −√En · tan(√En · L/2) ─────────────────
#
#   En < 0  →   √(−En) · tanh(√(−En) · L/2)
#   En > 0  →  −√En    · tan (  √En  · L/2)
#   En = 0  →   2/L  (limit)

def ktan(En, L):
    if En < 0.0:
        k = np.sqrt(-En)
        return k * np.tanh(0.5 * L * k)
    elif En > 0.0:
        k = np.sqrt(En)
        return -k * np.tan(0.5 * L * k)
    else:
        return 2.0 / L


# ── g12(En): finite-detuning transcendental equation ──────────────────────
#
# Inversion of the exact two-body equation at energy En.
# En is measured from the non-interacting two-body threshold.
# The absolute energy stored is  En − √(Ω²+Δ²).
#
# Three channels contribute:
#   channel 0  (free):      threshold 0,     momentum K0 = ktan(En, L)
#   channel 1  (dressed−):  threshold  λ,    κ1 = √(−En + λ)
#   channel 2  (dressed+):  threshold 2λ,    κ2 = √(−En + 2λ)
# with  λ = √(Δ²+Ω²).

def get_g12(En, g, Omega, Delta, L):
    lam = np.sqrt(Delta**2 + Omega**2)

    # ktan handles the analytic continuation tanh→tan for each channel
    kt1 = ktan(En - lam,       L)   # κ1·tanh(κ1·L/2)  or  −√·tan  above threshold
    kt2 = ktan(En - 2.0 * lam, L)   # κ2·tanh(κ2·L/2)  or  −√·tan  above threshold
    D1  = g + 2.0 * kt1
    D2  = g + 2.0 * kt2

    K0 = ktan(En, L)
    A  = g + 2.0 * K0      # = g − 2√En·tan(√En·L/2)

    numer = (Omega**2 * K0
             + 2.0 * Delta**2 * kt1 * A / D1
             + Omega**2       * kt2 * A / D2)
    denom = Omega**2 + 2.0 * Delta**2 * A / D1 + Omega**2 * A / D2

    if abs(denom) < 1e-14:
        return np.nan
    return -2.0 * numer / denom


# ── Populations P11, P12, P22 ──────────────────────────────────────────────
#
# P11: both atoms spin-up    (↑↑)
# P12: one up one down       (↑↓ + ↓↑),  factor of 2 already included
# P22: both atoms spin-down  (↓↓)
# Normalised so P11 + P12 + P22 = 1.
#
# Building blocks — valid for any En via analytic continuation:
#   sq0, kappa1, kappa2 are taken as complex so that cosh→cos, sinh→i·sin
#   when a channel crosses its threshold.  All P_ij remain real.
#
#   C_a = g·cosh(κ_a·L/2) + 2κ_a·sinh(κ_a·L/2)
#   I_a = L + sinh(κ_a·L)/κ_a
#   X_ab = −κ_a·cosh(κ_b·L/2)·sinh(κ_a·L/2)
#          + κ_b·cosh(κ_a·L/2)·sinh(κ_b·L/2)

def get_populations(En, g, Omega, Delta, L):
    lam    = np.sqrt(Delta**2 + Omega**2)
    u      = Delta + lam               # Δ + √(Δ²+Ω²)  [appears as (Δ+lam)]

    # complex sqrt → analytic continuation above threshold
    sq0    = np.sqrt(complex(-En))
    kappa1 = np.sqrt(complex(-En + lam))
    kappa2 = np.sqrt(complex(-En + 2.0 * lam))

    C0 = g * np.cosh(sq0    * L / 2) + 2.0 * sq0    * np.sinh(sq0    * L / 2)
    C1 = g * np.cosh(kappa1 * L / 2) + 2.0 * kappa1 * np.sinh(kappa1 * L / 2)
    C2 = g * np.cosh(kappa2 * L / 2) + 2.0 * kappa2 * np.sinh(kappa2 * L / 2)

    I0 = L + np.sinh(sq0    * L) / sq0
    I1 = L + np.sinh(kappa1 * L) / kappa1
    I2 = L + np.sinh(kappa2 * L) / kappa2

    X01 = (-sq0    * np.cosh(kappa1 * L / 2) * np.sinh(sq0    * L / 2)
           + kappa1 * np.cosh(sq0    * L / 2) * np.sinh(kappa1 * L / 2))
    X02 = (-sq0    * np.cosh(kappa2 * L / 2) * np.sinh(sq0    * L / 2)
           + kappa2 * np.cosh(sq0    * L / 2) * np.sinh(kappa2 * L / 2))
    X12 = (-kappa1 * np.cosh(kappa2 * L / 2) * np.sinh(kappa1 * L / 2)
           + kappa2 * np.cosh(kappa1 * L / 2) * np.sinh(kappa2 * L / 2))

    P11 = (I0 / 4.0
           - 4.0 * Delta          * C0      * X01 / (lam * u    * C1)
           + Delta**2             * C0**2   * I1  / (u**2        * C1**2)
           - Omega**2             * C0      * X02 / (lam * u**2  * C2)
           + 4.0 * Delta * Omega**2 * C0**2 * X12 / (lam * u**3  * C1 * C2)
           + Omega**4             * C0**2   * I2  / (4.0 * u**4  * C2**2))

    P12 = 2.0 * (
            Omega**2 * I0                          / (4.0 * u**2)
          + 4.0 * Delta**2 * C0  * X01             / (lam * u**2  * C1)
          + Delta**4       * C0**2 * I1            / (Omega**2 * u**2 * C1**2)
          + Omega**2       * C0  * X02             / (lam * u**2  * C2)
          + 4.0 * Delta**2 * C0**2 * X12           / (lam * u**2  * C1 * C2)
          + Omega**2       * C0**2 * I2            / (4.0 * u**2  * C2**2))

    P22 = (Omega**4             * I0              / (4.0 * u**4)
           + 4.0 * Delta * Omega**2 * C0 * X01    / (lam * u**3  * C1)
           + Delta**2             * C0**2 * I1    / (u**2         * C1**2)
           - Omega**2             * C0 * X02      / (lam * u**2   * C2)
           - 4.0 * Delta          * C0**2 * X12   / (lam * u      * C1 * C2)
           + C0**2                * I2            / (4.0           * C2**2))

    TotP = P11 + P12 + P22
    if abs(TotP) < 1e-14:
        return np.nan, np.nan, np.nan
    return float(np.real(P11 / TotP)), float(np.real(P12 / TotP)), float(np.real(P22 / TotP))


# ── Hellmann-Feynman check: dE_total/dΔ  (numerical, implicit function thm)
#
# E_total = En(Δ) − √(Ω²+Δ²)
# By the implicit function theorem  g12(En, Δ) = const:
#   dEn/dΔ = −(∂g12/∂Δ) / (∂g12/∂En)
# so:
#   dE_total/dΔ = dEn/dΔ − Δ/√(Ω²+Δ²)
#
# Hellmann-Feynman predicts  dE_total/dΔ = −P11 + P22, i.e. P11−P22 = −dE/dΔ.

def dEtot_dDelta(En, g, Omega, Delta, L, h=1e-5):
    dg12_dD  = (get_g12(En,     g, Omega, Delta + h, L)
              - get_g12(En,     g, Omega, Delta - h, L)) / (2.0 * h)
    dg12_dEn = (get_g12(En + h, g, Omega, Delta,     L)
              - get_g12(En - h, g, Omega, Delta,     L)) / (2.0 * h)
    lam = np.sqrt(Omega**2 + Delta**2)
    return -dg12_dD / dg12_dEn - Delta / lam


# ── Scan (mirrors the Mathematica While loop) ──────────────────────────────
#
# Scan En from Emin upward; at each step invert g12(En) and record
# {g12/g12MF, En − λ}.  Stop when g12 stops growing.

lam = np.sqrt(Omega**2 + Delta**2)

Envsg12   = []
P11vsg12  = []
P12vsg12  = []
P22vsg12  = []
DEdDvsg12 = []   # Hellmann-Feynman check
VarPvsg12    = []   # polarisation fluctuations  Var(P)   = <P²> - <P>²
VarAbsPvsg12 = []   # polarisation fluctuations  Var|P|(P) = <P²> - <|P|>²

En   = Emin
flag = 1.0

while flag > 0:
    g12 = get_g12(En, g, Omega, Delta, L)

    if not np.isfinite(g12):
        En += dE
        continue

    g12_norm = g12 / 1.0
    En_abs   = En - lam

    if g12_norm < g12_min:
        En += dE
        continue
    if g12_norm > g12_max:
        break

    Envsg12.append([g12_norm, En_abs])

    if len(Envsg12) > 1:
        flag = Envsg12[-1][0] - Envsg12[-2][0]

    P11, P12, P22 = get_populations(En, g, Omega, Delta, L)
    P11vsg12.append([g12_norm, P11])
    P12vsg12.append([g12_norm, P12])
    P22vsg12.append([g12_norm, P22])

    var_P    = (P11 + P22) - (P11 - P22)**2
    var_absP = (P11 + P22) - (P11 + P22)**2
    VarPvsg12.append([g12_norm, var_P])
    VarAbsPvsg12.append([g12_norm, var_absP])

    DEdDvsg12.append([g12_norm, dEtot_dDelta(En, g, Omega, Delta, L)])

    En += dE

# Drop last point (mirrors Mathematica's  Re[Delete[..., -1]])
Envsg12   = np.real(np.array(Envsg12[:-1]))
P11vsg12  = np.real(np.array(P11vsg12[:-1]))
P12vsg12  = np.real(np.array(P12vsg12[:-1]))
P22vsg12  = np.real(np.array(P22vsg12[:-1]))
DEdDvsg12 = np.real(np.array(DEdDvsg12[:-1]))
VarPvsg12    = np.real(np.array(VarPvsg12[:-1]))
VarAbsPvsg12 = np.real(np.array(VarAbsPvsg12[:-1]))


# ── Save TXT ───────────────────────────────────────────────────────────────
#
# Columns: g12  E/N  P
# E/N  = two-body absolute energy / 2 (N=2 particles)
# P    = P11 - P22  (spin polarisation)

t_val = Omega / 2.0   # tunnelling amplitude
h_val = Delta / 2.0   # DMC convention: h = Delta/2

_g12_col     = Envsg12[:, 0]
_EN_col      = Envsg12[:, 1] / 2.0
_P_col       = P11vsg12[:, 1] - P22vsg12[:, 1]          # <P>   = P11 - P22
_P2_col      = P11vsg12[:, 1] + P22vsg12[:, 1]          # <P²>  = P11 + P22
_absP_col    = P11vsg12[:, 1] + P22vsg12[:, 1]          # <|P|> = P11 + P22
_varP_col    = VarPvsg12[:, 1]                           # <P²> - <P>²
_varAbsP_col = VarAbsPvsg12[:, 1]                        # <P²> - <|P|>²

_out = np.column_stack([_g12_col, _EN_col, _P_col, _P2_col, _absP_col,
                        _varP_col, _varAbsP_col])
E_at_11 = np.interp(11.0, _g12_col, _EN_col)
Pabs_at_11 = np.interp(11.0, _g12_col, _absP_col)
print(f"E/N at g12 = 11.0: {E_at_11:.10f}")
print(f"|P| at g12 = 11.0: {Pabs_at_11:.10f}")

_fname = (Path(__file__).resolve().parent
          / f"2b_exact_t{t_val}_g{g}_h{h_val}.txt")
_header = (f"g12 scan — exact two-body on ring\n"
           f"g (intra) = {g}, N1 = 1, N2 = 1, t = {t_val}, Delta = {h_val}, L = {L}\n"
           f"g12\tE/N\tP (=P11-P22)\tP2 (=P11+P22)\t|P| (=P11+P22)\tVar(P)\tVar|P|(P)")
np.savetxt(_fname, _out, header=_header, fmt="%.10e", delimiter="\t")
print(f"\nSaved: {_fname}")


# ── Plots ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 5, figsize=(25, 5))
g12c_MF = g + 2.0 * t_val / ntot   # MF crossover  g12c = g + 2t/n

fig.suptitle(
    rf"$g={g}$,  $\Omega={Omega:.4g}$,  $\Delta={Delta:.4g}$,  $L={L}$,"
    rf"  $g_{{12}}^{{\rm MF}}=g+2t/n={g12c_MF:.4g}$",
    fontsize=12)

# Plot 1 — energy
ax = axes[0]
ax.plot(Envsg12[:, 0], Envsg12[:, 1], lw=1.5, color='steelblue')
ax.axvline(g12c_MF, color='k', lw=1.0, linestyle='--')
ax.set_xlabel(r"$g_{12}$")
ax.set_ylabel(r"$E_2$  $(m=\hbar=1)$")
ax.grid(True, alpha=0.4)

# Plot 2 — populations
ax = axes[1]
ax.plot(P11vsg12[:, 0], P11vsg12[:, 1], lw=1.5, label=r"$P_{11}$")
ax.plot(P12vsg12[:, 0], P12vsg12[:, 1], lw=1.5, label=r"$P_{12}$")
ax.plot(P22vsg12[:, 0], P22vsg12[:, 1], lw=1.5, label=r"$P_{22}$")
ax.axvline(g12c_MF, color='k', lw=1.0, linestyle='--')
ax.set_xlabel(r"$g_{12}$")
ax.set_ylabel(r"$P_{\sigma\sigma'}$")
ax.legend()
ax.grid(True, alpha=0.4)

# Plot 3 — polarisation + Hellmann-Feynman check
ax = axes[2]
pol = P11vsg12[:, 1] - P22vsg12[:, 1]
ax.plot(P11vsg12[:, 0], pol,
        lw=1.5, label=r"$P_{11}-P_{22}$")
ax.plot(DEdDvsg12[:, 0], -DEdDvsg12[:, 1],
        lw=1.5, linestyle='--',
        label=r"$-dE/d\Delta$  (H-F check)")
ax.axvline(g12c_MF, color='k', lw=1.0, linestyle='--')
ax.set_xlabel(r"$g_{12}$")
ax.set_ylabel(r"Polarisation $= P_{11}-P_{22}$")
ax.legend()
ax.grid(True, alpha=0.4)

# Plot 4 — polarisation fluctuations
ax = axes[3]
ax.plot(VarPvsg12[:, 0],    VarPvsg12[:, 1],    lw=1.5, color='darkorange',
        label=r"$\langle P^2\rangle - \langle P\rangle^2$")
ax.plot(VarAbsPvsg12[:, 0], VarAbsPvsg12[:, 1], lw=1.5, color='purple', linestyle='--',
        label=r"$\langle P^2\rangle - \langle|P|\rangle^2$")
ax.axvline(g12c_MF, color='k', lw=1.0, linestyle='--')
ax.set_xlabel(r"$g_{12}$")
ax.set_ylabel(r"Polarisation fluctuations")
ax.legend()
ax.grid(True, alpha=0.4)

# Plot 5 — mean |P|
ax = axes[4]
abspol = P11vsg12[:, 1] + P22vsg12[:, 1]
ax.plot(P11vsg12[:, 0], abspol, lw=1.5, color='teal')
ax.axvline(g12c_MF, color='k', lw=1.0, linestyle='--')
ax.set_xlabel(r"$g_{12}$")
ax.set_ylabel(r"$\langle|P|\rangle = P_{11}+P_{22}$")
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("TwoBodyFiniteDetuning.png", dpi=150, bbox_inches='tight')
plt.show()


# ── Save individual plots ──────────────────────────────────────────────────

_single_dir = Path(__file__).resolve().parent / "single_plots_2b"
_single_dir.mkdir(exist_ok=True)

_single_plots = [
    # (filename_stem,  plotting_function)
    ("energy",         lambda ax: (
        ax.plot(Envsg12[:, 0], Envsg12[:, 1], lw=1.5, color='steelblue'),
        ax.axvline(g12c_MF, color='k', lw=1.0, linestyle='--'),
        ax.set_xlabel(r"$g_{12}$"),
        ax.set_ylabel(r"$E_2$  $(m=\hbar=1)$"),
        ax.grid(True, alpha=0.4),
    )),
    ("populations",    lambda ax: (
        ax.plot(P11vsg12[:, 0], P11vsg12[:, 1], lw=1.5, label=r"$P_{11}$"),
        ax.plot(P12vsg12[:, 0], P12vsg12[:, 1], lw=1.5, label=r"$P_{12}$"),
        ax.plot(P22vsg12[:, 0], P22vsg12[:, 1], lw=1.5, label=r"$P_{22}$"),
        ax.axvline(g12c_MF, color='k', lw=1.0, linestyle='--'),
        ax.set_xlabel(r"$g_{12}$"),
        ax.set_ylabel(r"$P_{\sigma\sigma'}$"),
        ax.legend(),
        ax.grid(True, alpha=0.4),
    )),
    ("polarisation",   lambda ax: (
        ax.plot(P11vsg12[:, 0], P11vsg12[:, 1] - P22vsg12[:, 1],
                lw=1.5, label=r"$P_{11}-P_{22}$"),
        ax.plot(DEdDvsg12[:, 0], -DEdDvsg12[:, 1], lw=1.5, linestyle='--',
                label=r"$-dE/d\Delta$  (H-F check)"),
        ax.axvline(g12c_MF, color='k', lw=1.0, linestyle='--'),
        ax.set_xlabel(r"$g_{12}$"),
        ax.set_ylabel(r"Polarisation $= P_{11}-P_{22}$"),
        ax.legend(),
        ax.grid(True, alpha=0.4),
    )),
    ("fluctuations",   lambda ax: (
        ax.plot(VarPvsg12[:, 0],    VarPvsg12[:, 1],    lw=1.5, color='darkorange',
                label=r"$\langle P^2\rangle - \langle P\rangle^2$"),
        ax.plot(VarAbsPvsg12[:, 0], VarAbsPvsg12[:, 1], lw=1.5, color='purple',
                linestyle='--', label=r"$\langle P^2\rangle - \langle|P|\rangle^2$"),
        ax.axvline(g12c_MF, color='k', lw=1.0, linestyle='--'),
        ax.set_xlabel(r"$g_{12}$"),
        ax.set_ylabel(r"Polarisation fluctuations"),
        ax.legend(),
        ax.grid(True, alpha=0.4),
    )),
    ("abs_polarisation", lambda ax: (
        ax.plot(P11vsg12[:, 0], P11vsg12[:, 1] + P22vsg12[:, 1], lw=1.5, color='teal'),
        ax.axvline(g12c_MF, color='k', lw=1.0, linestyle='--'),
        ax.set_xlabel(r"$g_{12}$"),
        ax.set_ylabel(r"$\langle|P|\rangle = P_{11}+P_{22}$"),
        ax.grid(True, alpha=0.4),
    )),
]

for _stem, _plot_fn in _single_plots:
    _fig, _ax = plt.subplots(figsize=(5, 4))
    _plot_fn(_ax)
    _fig.suptitle(
        rf"$g={g}$,  $t={t_val:.4g}$,  $h={h_val:.4g}$",
        fontsize=10)
    _fig.tight_layout()
    _fpath = _single_dir / f"{_stem}.png"
    _fig.savefig(_fpath, dpi=150, bbox_inches='tight')
    plt.close(_fig)
    print(f"Saved: {_fpath}")


# ── Summary ────────────────────────────────────────────────────────────────
print(f"g = {g},  Omega = {Omega:.4g},  Delta = {Delta:.4g},  L = {L}")
print(f"g12MF = {g12MF:.4g},   lambda = {lam:.4g}")
print(f"Points computed: {len(Envsg12)}")
print("\nFirst 5 rows of Envsg12  (g12/g12MF,  E2):")
print(Envsg12[:5])
print("\nFirst 5 rows of P11vsg12  (g12/g12MF,  P11):")
print(P11vsg12[:5])
