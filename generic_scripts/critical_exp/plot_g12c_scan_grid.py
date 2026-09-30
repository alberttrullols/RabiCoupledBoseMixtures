from math import hypot
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXCLUDE_SIZES = [20]
ADD_ERRORS = True

NU = 1.0
BETA = 0.5

G12C_MIN = 0.0122
G12C_MAX = 0.0132
G12C_COUNT = 15

G12_MIN = None
G12_MAX = None


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
    raise FileNotFoundError(f"No params file found in {folder}")


def parse_params(folder):
    params = read_params_file(folder)
    if "g12" not in params:
        raise ValueError(f"Could not infer g12 from params file in {folder}")
    g12 = float(params["g12"])

    if "N1" in params and "N2" in params:
        N = int(round(float(params["N1"]) + float(params["N2"])))
    elif "N" in params:
        N = int(round(float(params["N"])))
    else:
        raise ValueError(f"Could not infer N from params file in {folder}")
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
    return hypot(dU_dP2 * p2_err, dU_dP4 * p4_err)


def collect_records():
    records = {}
    for folder in sorted(Path(".").iterdir()):
        if not folder.is_dir():
            continue

        try:
            g12, N = parse_params(folder)
        except (FileNotFoundError, ValueError):
            continue

        N = int(N)
        if N in EXCLUDE_SIZES:
            continue

        data = read_results(folder)
        if data is None:
            continue

        if G12_MIN is not None and float(g12) < G12_MIN:
            continue
        if G12_MAX is not None and float(g12) > G12_MAX:
            continue

        u4 = binder_u4(data)
        p2 = data.get("P2_avg")
        p2_err = data.get("P2_err")

        if u4 is None and p2 is None:
            continue

        records.setdefault(N, []).append(
            {
                "g12": float(g12),
                "u4": u4,
                "u4_err": binder_u4_error(data),
                "p2": p2,
                "p2_err": p2_err,
            }
        )

    return records


records = collect_records()
if not records:
    raise SystemExit("No usable Binder or P2 data found in this folder.")

# Scan around a plausible critical coupling estimate and inspect data collapse.
g12c_values = [
    G12C_MIN + (G12C_MAX - G12C_MIN) * i / max(G12C_COUNT - 1, 1)
    for i in range(G12C_COUNT)
]

fig, axes = plt.subplots(2, len(g12c_values), figsize=(3.4 * len(g12c_values), 7.5), sharex=True)
if len(g12c_values) == 1:
    axes = axes.reshape(2, 1)

for j, g12c in enumerate(g12c_values):
    uax = axes[0, j]
    p2ax = axes[1, j]

    for N in sorted(records):
        pts = sorted(records[N], key=lambda d: d["g12"])

        u4_x = [(d["g12"] - g12c) * (N ** (1.0 / NU)) for d in pts if d["u4"] is not None]
        u4_y = [d["u4"] for d in pts if d["u4"] is not None]
        u4_yerr = [d["u4_err"] if d["u4_err"] is not None else 0.0 for d in pts if d["u4"] is not None]

        p2_x = [(d["g12"] - g12c) * (N ** (1.0 / NU)) for d in pts if d["p2"] is not None]
        p2_y = [d["p2"] * (N ** (2.0 * BETA / NU)) for d in pts if d["p2"] is not None]
        p2_yerr = [d["p2_err"] if d["p2_err"] is not None else 0.0 for d in pts if d["p2"] is not None]

        if u4_x:
            label = f"N={N}" if j == 0 else None
            if ADD_ERRORS:
                uax.errorbar(u4_x, u4_y, yerr=u4_yerr, fmt="o-", capsize=2, alpha=0.9, label=label)
            else:
                uax.plot(u4_x, u4_y, "o-", alpha=0.9, label=label)

        if p2_x:
            label = f"N={N}" if j == 0 else None
            if ADD_ERRORS:
                p2ax.errorbar(p2_x, p2_y, yerr=p2_yerr, fmt="o-", capsize=2, alpha=0.9, label=label)
            else:
                p2ax.plot(p2_x, p2_y, "o-", alpha=0.9, label=label)

    uax.axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.8)
    uax.grid(True, linestyle="--", alpha=0.4)
    uax.set_ylabel(r"$U_4$")
    uax.set_title(rf"$g_{{12,c}} = {g12c:.5f}$", fontsize=10)

    p2ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.8)
    p2ax.grid(True, linestyle="--", alpha=0.4)
    p2ax.set_ylabel(r"$P^2 N^{2\beta/\nu}$")
    p2ax.set_xlabel(r"$(g_{12}-g_{12,c})N^{1/\nu}$")

    if j == 0:
        uax.legend(loc="best", fontsize="small")
        p2ax.legend(loc="best", fontsize="small")

fig.suptitle(rf"Grid scan over $g_{{12,c}}$ for fixed $\nu={NU}$ and $\beta={BETA}$")
fig.tight_layout(rect=[0, 0, 1, 0.97])
out_png = Path("g12c_scan_u4_p2_grid.png")
fig.savefig(out_png, dpi=200)
print(f"Saved {out_png}")
