import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


QUANTITIES = {
    "en": "energy per particle",
    "p2": "mean squared polarization",
    "u4": "Binder cumulant",
}


def read_key_value_file(path: Path) -> dict[str, float]:
    values = {}
    for line in path.read_text().splitlines():
        columns = line.split()
        if len(columns) < 2:
            continue
        try:
            values[columns[0]] = float(columns[1])
        except ValueError:
            continue
    return values


def normalize_quantity(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def get_point_value(
    quantity: str, params: dict[str, float], results: dict[str, float]
) -> tuple[int, float, float, float | None]:
    particle_count = int(params["N1"] + params["N2"])
    g12 = params["g12"]

    if quantity == "en":
        value = results["E_DMC"] / particle_count
        error = results.get("E_DMC_err", 0.0) / particle_count
    elif quantity == "p2":
        value = results["P2_avg"]
        error = results.get("P2_err", 0.0)
    else:
        p2 = results["P2_avg"]
        p4 = results["P4_avg"]
        if p2 <= 0:
            raise ValueError("P2_avg must be positive to calculate U4")
        value = 1.0 - p4 / (3.0 * p2**2)
        error = None

    if not math.isfinite(g12) or not math.isfinite(value):
        raise ValueError("g12 and the requested quantity must be finite")
    return particle_count, g12, value, error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot a simulation quantity against g12 for a folder of datapoints."
    )
    parser.add_argument("folder", type=Path, help="folder containing datapoint subfolders")
    parser.add_argument("quantity", help="quantity to plot: E/N, P^2, or U4")
    parser.add_argument(
        "size",
        nargs="?",
        type=int,
        help="optional total particle count; omit to plot all sizes",
    )
    args = parser.parse_args()

    quantity_key = normalize_quantity(args.quantity)
    aliases = {
        "en": "en",
        "energyperparticle": "en",
        "p2": "p2",
        "u4": "u4",
        "bindercumulant": "u4",
    }
    quantity = aliases.get(quantity_key)
    if quantity is None:
        parser.error("quantity must be E/N, P^2, or U4")
    if args.size is not None and args.size <= 0:
        parser.error("size must be a positive total particle count")

    root = args.folder.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"not a folder: {root}")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        if error.name == "matplotlib":
            print(
                "Matplotlib is required. Install it with: python -m pip install -r requirements.txt",
                file=sys.stderr,
            )
            return 1
        raise

    datasets = defaultdict(list)
    for point_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        params_path = point_dir / "params.dat"
        results_path = point_dir / "results.txt"
        if not params_path.is_file() or not results_path.is_file():
            continue

        try:
            params = read_key_value_file(params_path)
            results = read_key_value_file(results_path)
            particle_count, g12, value, error = get_point_value(
                quantity, params, results
            )
            if args.size is not None and particle_count != args.size:
                continue
            datasets[particle_count].append((g12, value, error))
        except (KeyError, ValueError, OSError) as error:
            print(f"Skipping {point_dir.name}: {error}", file=sys.stderr)

    if not datasets:
        size_message = f" for N={args.size}" if args.size is not None else ""
        print(f"No usable datapoints found{size_message} in {root}", file=sys.stderr)
        return 1

    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    for particle_count, points in sorted(datasets.items()):
        points.sort(key=lambda point: point[0])
        g12_values = [point[0] for point in points]
        values = [point[1] for point in points]
        errors = [point[2] for point in points]
        if any(error is not None and error > 0 for error in errors):
            axis.errorbar(
                g12_values,
                values,
                yerr=[error or 0.0 for error in errors],
                marker="o",
                markersize=4,
                linewidth=1.2,
                capsize=2,
                label=f"N={particle_count}",
            )
        else:
            axis.plot(
                g12_values,
                values,
                marker="o",
                markersize=4,
                linewidth=1.2,
                label=f"N={particle_count}",
            )

    ylabel = {
        "en": "E/N",
        "p2": "mean squared polarization, P^2",
        "u4": "U4 = 1 - <P^4> / (3 <P^2>^2)",
    }[quantity]
    axis.set_xlabel("g12")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{QUANTITIES[quantity]} vs g12")
    axis.grid(True, alpha=0.25)
    axis.legend(title="Particle count")

    size_suffix = f"_N{args.size}" if args.size is not None else ""
    output_path = root / f"sweep_{quantity}{size_suffix}.png"
    figure.savefig(output_path, dpi=160)
    print(f"Saved {output_path}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())