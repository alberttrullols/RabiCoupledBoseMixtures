from math import hypot, log
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize_scalar


# ============================================================
# USER SETTINGS
# ============================================================

EXCLUDE_SIZES = [10]

ADD_ERRORS = True
WEIGHT_ERRORS = False

# Critical coupling. This is FIXED in the present analysis.
G12C = 0.01478

G12_MIN = 0.013
G12_MAX = 0.016

# Range used for the nu search
NU_MIN = 0.5
NU_MAX = 2.5

# Number of points used to construct the P_b(nu) curve
NU_COUNT = 200

# Fractional increase in P_b used for the uncertainty estimate.
#
# This follows the spirit of Appendix A Eq. (8), where the
# uncertainty is obtained from the width of the minimum.
#
# For example:
#   UNCERTAINTY_LEVEL = 0.01
# corresponds to the 1% level used as an example in Appendix A.
UNCERTAINTY_LEVEL = 0.01

# Number of x points used in the interpolation comparison.
# Set to None to use all available points from both curves.
N_INTERP = 200

# Minimum number of overlapping points required for a pair
MIN_OVERLAP_POINTS = 3


# ============================================================
# INPUT
# ============================================================

def read_params_file(folder):
    for filename in ("params.txt", "params.dat"):
        path = folder / filename

        if not path.exists():
            continue

        data = {}

        for line in path.read_text().splitlines():
            parts = line.split()

            if len(parts) < 2:
                continue

            try:
                data[parts[0]] = float(parts[1])
            except ValueError:
                pass

        return data

    raise FileNotFoundError(
        f"No params file found in {folder}"
    )


def parse_params(folder):
    params = read_params_file(folder)

    if "g12" not in params:
        raise ValueError(
            f"Could not infer g12 from params file in {folder}"
        )

    g12 = float(params["g12"])

    if "N1" in params and "N2" in params:
        N = int(round(float(params["N1"]) + float(params["N2"])))

    elif "N" in params:
        N = int(round(float(params["N"])))

    else:
        raise ValueError(
            f"Could not infer N from params file in {folder}"
        )

    return g12, N


def read_results(folder):
    rfile = folder / "results.txt"

    if not rfile.exists():
        return None

    data = {}

    for line in rfile.read_text().splitlines():
        parts = line.split()

        if len(parts) < 2:
            continue

        try:
            data[parts[0]] = float(parts[1])
        except ValueError:
            pass

    return data


# ============================================================
# BINDER CUMULANT
# ============================================================

def binder_u4(data):
    p2 = data.get("P2_avg")
    p4 = data.get("P4_avg")

    if p2 is None or p4 is None:
        return None

    if abs(p2) < 1e-30:
        return None

    return 1.0 - p4 / (3.0 * p2**2)


def binder_u4_error(data):
    p2 = data.get("P2_avg")
    p4 = data.get("P4_avg")

    p2_err = data.get("P2_err")
    p4_err = data.get("P4_err")

    if p2 is None or p4 is None:
        return None

    if p2_err is None or p4_err is None:
        return None

    if abs(p2) < 1e-30:
        return None

    dU_dP2 = -2.0 * p4 / (3.0 * p2**3)
    dU_dP4 = -1.0 / (3.0 * p2**2)

    return hypot(
        dU_dP2 * p2_err,
        dU_dP4 * p4_err
    )


# ============================================================
# READ ALL DATA
# ============================================================

records = {}

for folder in sorted(Path(".").iterdir()):

    if not folder.is_dir():
        continue

    try:
        g12, N = parse_params(folder)
    except (FileNotFoundError, ValueError):
        continue

    if N in EXCLUDE_SIZES:
        continue

    data = read_results(folder)

    if data is None:
        continue

    if G12_MIN is not None and g12 < G12_MIN:
        continue

    if G12_MAX is not None and g12 > G12_MAX:
        continue

    u4 = binder_u4(data)

    if u4 is None:
        continue

    u4_err = binder_u4_error(data)

    records.setdefault(N, []).append(
        (float(g12), float(u4), u4_err)
    )


if not records:
    raise SystemExit(
        "No usable Binder data found in this folder."
    )


# Sort every curve by g12
for N in records:
    records[N].sort(key=lambda x: x[0])


print()
print("Loaded Binder data:")
for N in sorted(records):
    print(
        f"  N={N:>5}: "
        f"{len(records[N]):>4} points, "
        f"g12=[{records[N][0][0]:.6f}, "
        f"{records[N][-1][0]:.6f}]"
    )


# ============================================================
# RESCALE A CURVE
# ============================================================

def rescaled_curve(N, nu):
    """
    Return x, U4, sigma arrays for

        x = (g12 - g12c) N^(1/nu)

    """

    pts = records[N]

    g = np.array([p[0] for p in pts], dtype=float)
    u = np.array([p[1] for p in pts], dtype=float)

    x = (g - G12C) * N**(1.0 / nu)

    if all(p[2] is not None for p in pts):
        sigma = np.array(
            [p[2] for p in pts],
            dtype=float
        )
    else:
        sigma = np.zeros_like(u)

    # Ensure increasing x
    order = np.argsort(x)

    return x[order], u[order], sigma[order]


# ============================================================
# PAIRWISE COLLAPSE RESIDUAL
# ============================================================

def pair_residual(curve_a, curve_b):
    """
    Compare two rescaled Binder curves.

    Only the overlapping x-region is used.

    Linear interpolation is used to compare the curves at
    common x positions.

    This is the Binder-cumulant analogue of the interpolation
    procedure used in Appendix A.
    """

    xa, ya, sa = curve_a
    xb, yb, sb = curve_b

    if len(xa) < 2 or len(xb) < 2:
        return None

    xmin = max(xa.min(), xb.min())
    xmax = min(xa.max(), xb.max())

    if xmax <= xmin:
        return None

    # Candidate points from both curves inside overlap
    mask_a = (xa >= xmin) & (xa <= xmax)
    mask_b = (xb >= xmin) & (xb <= xmax)

    x_common = np.concatenate(
        [xa[mask_a], xb[mask_b]]
    )

    x_common = np.unique(x_common)

    if len(x_common) < MIN_OVERLAP_POINTS:
        return None

    # Optional interpolation grid
    if N_INTERP is not None and len(x_common) > N_INTERP:
        x_common = np.linspace(
            xmin,
            xmax,
            N_INTERP
        )

    ya_interp = np.interp(
        x_common,
        xa,
        ya
    )

    yb_interp = np.interp(
        x_common,
        xb,
        yb
    )

    residual = ya_interp - yb_interp

    if WEIGHT_ERRORS:

        sa_interp = np.interp(
            x_common,
            xa,
            sa
        )

        sb_interp = np.interp(
            x_common,
            xb,
            sb
        )

        variance = sa_interp**2 + sb_interp**2

        # Avoid division by zero
        valid = variance > 0

        if not np.any(valid):
            return None

        residual = residual[valid]
        variance = variance[valid]

        chi2 = np.mean(
            residual**2 / variance
        )

        return chi2

    else:

        return np.mean(
            residual**2
        )


# ============================================================
# TOTAL COLLAPSE MEASURE P_b
# ============================================================

def collapse_measure(nu, return_details=False):

    sizes = sorted(records)

    curves = {
        N: rescaled_curve(N, nu)
        for N in sizes
    }

    pair_values = []
    pair_names = []

    for i in range(len(sizes)):

        for j in range(i + 1, len(sizes)):

            Ni = sizes[i]
            Nj = sizes[j]

            value = pair_residual(
                curves[Ni],
                curves[Nj]
            )

            if value is None:
                continue

            pair_values.append(value)
            pair_names.append((Ni, Nj))

    if not pair_values:
        return np.inf

    # Average over all pairs.
    Pb = np.mean(pair_values)

    if return_details:
        return Pb, pair_names, pair_values

    return Pb


# ============================================================
# CALCULATE P_b(nu)
# ============================================================

nus = np.linspace(
    NU_MIN,
    NU_MAX,
    NU_COUNT
)

Pb_values = np.array([
    collapse_measure(nu)
    for nu in nus
])


# ============================================================
# FIND MINIMUM
# ============================================================

finite = np.isfinite(Pb_values)

if not np.any(finite):
    raise SystemExit(
        "Could not calculate a valid collapse measure."
    )

nu_grid_best = nus[np.nanargmin(Pb_values)]


# Refine the minimum using scipy
result = minimize_scalar(
    collapse_measure,
    bounds=(
        max(NU_MIN, nu_grid_best - 0.1),
        min(NU_MAX, nu_grid_best + 0.1)
    ),
    method="bounded",
    options={
        "xatol": 1e-8
    }
)

if result.success:
    nu_best = result.x
    Pb_min = result.fun
else:
    nu_best = nu_grid_best
    Pb_min = collapse_measure(nu_best)


# ============================================================
# UNCERTAINTY FROM WIDTH OF MINIMUM
# ============================================================

def find_crossing(side, target):

    if side == "left":

        mask = nus < nu_best

        x = nus[mask]
        y = Pb_values[mask]

        if len(x) < 2:
            return None

        # Work from minimum outward
        order = np.argsort(x)[::-1]
        x = x[order]
        y = y[order]

    else:

        mask = nus > nu_best

        x = nus[mask]
        y = Pb_values[mask]

        if len(x) < 2:
            return None

        order = np.argsort(x)
        x = x[order]
        y = y[order]

    target_mask = y >= target

    if not np.any(target_mask):
        return None

    idx = np.where(target_mask)[0][0]

    if idx == 0:
        return x[idx]

    x1 = x[idx - 1]
    x2 = x[idx]

    y1 = y[idx - 1]
    y2 = y[idx]

    if abs(y2 - y1) < 1e-30:
        return x2

    return x1 + (target - y1) * (
        x2 - x1
    ) / (y2 - y1)


# Following the logarithmic-width idea of Appendix A Eq. (8),
# use P/P_min = exp(UNCERTAINTY_LEVEL).
#
# For a nonzero minimum:
#
#     target = P_min * exp(eta)
#
# where eta=0.01 corresponds to a 1% logarithmic level.

if Pb_min > 0:

    target = Pb_min * np.exp(
        UNCERTAINTY_LEVEL
    )

    nu_left = find_crossing(
        "left",
        target
    )

    nu_right = find_crossing(
        "right",
        target
    )

    if nu_left is not None and nu_right is not None:

        nu_err_minus = nu_best - nu_left
        nu_err_plus = nu_right - nu_best

        nu_error = 0.5 * (
            nu_err_minus +
            nu_err_plus
        )

    else:

        nu_left = None
        nu_right = None
        nu_err_minus = None
        nu_err_plus = None
        nu_error = None

else:

    nu_left = None
    nu_right = None
    nu_err_minus = None
    nu_err_plus = None
    nu_error = None


# ============================================================
# PRINT RESULT
# ============================================================

print()
print("=" * 60)
print("FINITE-SIZE SCALING RESULT")
print("=" * 60)

print(
    f"Fixed g12,c = {G12C:.8f}"
)

print(
    f"Best nu     = {nu_best:.6f}"
)

print(
    f"Minimum P_b = {Pb_min:.8e}"
)

if nu_error is not None:

    print(
        f"nu error    = +/- {nu_error:.6f}"
    )

    print(
        f"nu          = "
        f"{nu_best:.6f} "
        f"+{nu_err_plus:.6f} "
        f"-{nu_err_minus:.6f}"
    )

else:

    print(
        "Could not determine a finite "
        "collapse-width uncertainty."
    )

print("=" * 60)
print()


# ============================================================
# OPTIMAL COLLAPSE PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

for N in sorted(records):

    x, y, yerr = rescaled_curve(
        N,
        nu_best
    )

    if ADD_ERRORS:

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            capsize=2,
            label=f"N={N}"
        )

    else:

        ax.plot(
            x,
            y,
            "o-",
            label=f"N={N}"
        )


ax.axvline(
    0.0,
    color="k",
    linestyle="--",
    linewidth=1.0,
    alpha=0.8
)

ax.set_xlabel(
    r"$(g_{12}-g_{12,c})N^{1/\nu}$"
)

ax.set_ylabel(
    r"$U_4$"
)

ax.set_title(
    rf"Optimal Binder collapse: "
    rf"$\nu={nu_best:.4f}$"
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.4
)

ax.legend(
    loc="best",
    fontsize="small"
)

plt.tight_layout()

out_png = Path(
    "binder_u4_optimal_collapse.png"
)

fig.savefig(
    out_png,
    dpi=200
)

plt.close(fig)

print(
    f"Saved {out_png}"
)


# ============================================================
# P_b(nu) PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    nus,
    Pb_values,
    "o-",
    markersize=3
)

ax.axvline(
    nu_best,
    color="k",
    linestyle="--",
    linewidth=1.0,
    label=rf"$\nu={nu_best:.4f}$"
)

if Pb_min > 0:

    target = Pb_min * np.exp(
        UNCERTAINTY_LEVEL
    )

    ax.axhline(
        target,
        color="k",
        linestyle=":",
        linewidth=1.0,
        label="uncertainty level"
    )

    if nu_left is not None:
        ax.axvline(
            nu_left,
            color="k",
            linestyle=":",
            linewidth=1.0
        )

    if nu_right is not None:
        ax.axvline(
            nu_right,
            color="k",
            linestyle=":",
            linewidth=1.0
        )


ax.set_xlabel(
    r"$\nu$"
)

ax.set_ylabel(
    r"$P_b(\nu)$"
)

ax.set_title(
    "Binder-collapse quality"
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.4
)

ax.legend()

plt.tight_layout()

out_png = Path(
    "binder_u4_Pb_vs_nu.png"
)

fig.savefig(
    out_png,
    dpi=200
)

plt.close(fig)

print(
    f"Saved {out_png}"
)