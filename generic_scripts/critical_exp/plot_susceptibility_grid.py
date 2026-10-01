from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXCLUDE_SIZES = [20]
ADD_ERRORS = True

G12C = 0.01282
NU = 1.0

G12_MIN = None
G12_MAX = None

GAMMA_MIN = 0.5
GAMMA_MAX = 2.5
GAMMA_COUNT = 20


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

    if G12_MIN is not None and float(g12) < G12_MIN:
        continue
    if G12_MAX is not None and float(g12) > G12_MAX:
        continue

    chi = susceptibility_value(data, N)
    if chi is None:
        continue

    records.setdefault(N, []).append((float(g12), chi))

if not records:
    raise SystemExit("No usable susceptibility data found in this folder.")

gammas = [GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * i / max(GAMMA_COUNT - 1, 1) for i in range(GAMMA_COUNT)]

fig, axes = plt.subplots(len(gammas), 1, figsize=(8, 3.2 * len(gammas)), sharex=False)
if len(gammas) == 1:
    axes = [axes]

for ax, gamma in zip(axes, gammas):
    for N in sorted(records):
        pts = sorted(records[N], key=lambda x: x[0])
        x = [(g - G12C) * (N ** (1.0 / NU)) for g, _ in pts]
        y = [chi / (N ** (gamma / NU)) for _, chi in pts]

        ax.plot(x, y, "o-", label=f"N={N}")

    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylabel(r"$\chi N^{-\gamma/\nu}$")
    ax.set_title(rf"$\gamma = {gamma:.3f}$")

axes[-1].set_xlabel(r"$(g_{12}-g_{12,c})N^{1/\nu}$")
for ax in axes:
    ax.legend(loc="best", fontsize="small")

plt.tight_layout()
out_png = Path("susceptibility_gamma_grid.png")
fig.savefig(out_png, dpi=200)
print(f"Saved {out_png}")
