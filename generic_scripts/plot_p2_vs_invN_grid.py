from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXCLUDE_SIZES = [20]
ADD_ERRORS = True

DISCARD_FRACTION = 0.15

GRID_COLS = 10


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


def read_stream_values(path):
    if not path.exists():
        return None
    values = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            values.append(float(parts[1]))
        except ValueError:
            pass
    return values


def mean_and_error(values, discard_fraction=DISCARD_FRACTION):
    if not values:
        return None, None
    n_discard = int(len(values) * discard_fraction)
    kept = values[n_discard:]
    if not kept:
        return None, None
    n = len(kept)
    mean = sum(kept) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in kept) / (n - 1)
        err = (var / n) ** 0.5
    else:
        err = 0.0
    return mean, err


def read_results_file(folder):
    path = folder / "results.txt"
    if not path.exists():
        return {}
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


def read_p2(folder):
    stream_path = folder / "P2_stream.txt"
    if stream_path.exists():
        values = read_stream_values(stream_path)
        mean, err = mean_and_error(values)
        if mean is not None:
            return mean, err

    results = read_results_file(folder)
    return results.get("P2_avg"), results.get("P2_err")


# records[g12] = list of (N, p2, p2_err)
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

    p2, p2_err = read_p2(folder)
    if p2 is None:
        continue

    g12_key = round(float(g12), 6)
    records.setdefault(g12_key, []).append((N, p2, p2_err))

if not records:
    raise SystemExit("No usable P2 data found in this folder.")

g12_values = sorted(records)
n_panels = len(g12_values)
n_cols = min(GRID_COLS, n_panels)
n_rows = -(-n_panels // n_cols)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.2 * n_rows), sharex=True, sharey=False)
axs = axs.reshape(n_rows, n_cols) if n_panels > 1 else [[axs]]

for idx, g12 in enumerate(g12_values):
    row, col = divmod(idx, n_cols)
    ax = axs[row][col]

    pts = sorted(records[g12], key=lambda x: x[0])
    x = [1.0 / N for N, _, _ in pts]
    y = [p2 for _, p2, _ in pts]
    yerr = [err if err is not None else 0.0 for _, _, err in pts] if ADD_ERRORS else None

    if ADD_ERRORS:
        ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=2)
    else:
        ax.plot(x, y, "o-")

    ax.set_title(rf"$g_{{12}} = {g12:.5f}$", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

# Hide unused panels.
for idx in range(n_panels, n_rows * n_cols):
    row, col = divmod(idx, n_cols)
    axs[row][col].axis("off")

for col in range(n_cols):
    axs[n_rows - 1][col].set_xlabel(r"$1/N$")
for row in range(n_rows):
    axs[row][0].set_ylabel(r"$P^2$")

fig.suptitle(r"$P^2$ vs $1/N$ for each $g_{12}$")
plt.tight_layout(rect=(0, 0, 1, 0.97))
out_png = Path("p2_vs_invN_grid.png")
fig.savefig(out_png, dpi=200)
print(f"Saved {out_png}")
