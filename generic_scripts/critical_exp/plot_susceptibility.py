from math import hypot
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXCLUDE_SIZES = [20]
ADD_ERRORS = True

RESCALE = False
G12C = None
NU = None
GAMMA = None

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


def susceptibility_value(data, N):
    p2 = data.get("P2_avg")
    pabs = data.get("Pabs_avg")
    if p2 is None or pabs is None:
        return None
    return N * (p2 - pabs**2)


def susceptibility_error(data, N):
    p2 = data.get("P2_avg")
    pabs = data.get("Pabs_avg")
    p2_err = data.get("P2_err")
    pabs_err = data.get("Pabs_err")

    if p2 is None or pabs is None:
        return None
    if p2_err is None or pabs_err is None:
        return None

    # Error propagation for chi = N * (P2 - Pabs^2)
    dchi_dP2 = N
    dchi_dPabs = -2.0 * N * pabs
    return N * hypot(p2_err, 2.0 * pabs * pabs_err)


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

    chi = susceptibility_value(data, N)
    if chi is None:
        continue

    records.setdefault(N, []).append((float(g12), chi, susceptibility_error(data, N)))

if not records:
    raise SystemExit("No usable susceptibility data found in this folder.")

if RESCALE:
    if G12C is None or NU is None or GAMMA is None:
        raise SystemExit("RESCALE=True requires G12C, NU, and GAMMA to be set by the user.")

fig, ax = plt.subplots(figsize=(8, 5))

for N in sorted(records):
    pts = sorted(records[N], key=lambda x: x[0])
    if RESCALE:
        x = [(g - G12C) * (N ** (1.0 / NU)) for g, _, _ in pts]
        y = [chi / (N ** (GAMMA / NU)) for _, chi, _ in pts]
    else:
        x = [g for g, _, _ in pts]
        y = [chi for _, chi, _ in pts]
    yerr = [err if err is not None else 0.0 for _, _, err in pts] if ADD_ERRORS else None

    if ADD_ERRORS:
        ax.errorbar(x, y, yerr=yerr, fmt="o-", label=f"N={N}", capsize=2)
    else:
        ax.plot(x, y, "o-", label=f"N={N}")

if RESCALE:
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel(r"$(g_{12}-g_{12,c})\,N^{1/\nu}$")
    ax.set_ylabel(r"$\chi\,N^{-\gamma/\nu}$")
else:
    ax.axvline(G12C if G12C is not None else 0.01282, color="k", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel(r"$g_{12}$")
    ax.set_ylabel(r"$\chi = N\left(P^2 - |P|^2\right)$")

ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()
plt.tight_layout()
out_png = Path("susceptibility_collapse.png" if RESCALE else "susceptibility_vs_g12.png")
fig.savefig(out_png, dpi=200)
print(f"Saved {out_png}")
