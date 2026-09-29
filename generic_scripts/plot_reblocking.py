from pathlib import Path
import re
import sys


QUANTITIES = {
    "energy": "energy",
    "p": "P",
    "p2": "P^2",
    "p4": "P^4",
    "pabs": "|P|",
    "pmixed": "P_mixed",
    "p2mixed": "P2_mixed",
    "p4mixed": "P4_mixed",
    "pabsmixed": "Pabs_mixed",
}


def normalize_quantity(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def read_reblocking_file(path: Path) -> tuple[list[int], list[float]]:
    block_sizes = []
    errors = []
    for line in path.read_text().splitlines():
        columns = line.split()
        if len(columns) < 2:
            continue
        try:
            block_size = int(columns[0])
            error = float(columns[1])
        except ValueError:
            continue
        block_sizes.append(block_size)
        errors.append(error)
    return block_sizes, errors


def main() -> int:
    if sys.argv[1:] in ([], ["-h"], ["--help"]):
        print("Usage: python plot_reblocking.py -<quantity>")
        print("Quantities: " + ", ".join(QUANTITIES))
        return 0 if sys.argv[1:] else 2

    if len(sys.argv) != 2:
        print("Usage: python plot_reblocking.py -<quantity>", file=sys.stderr)
        return 2

    requested = sys.argv[1].removeprefix("-")
    key = normalize_quantity(requested)
    if key not in QUANTITIES:
        print(
            f"Unknown quantity {requested!r}. Choose: {', '.join(QUANTITIES)}",
            file=sys.stderr,
        )
        return 2

    root = Path(__file__).resolve().parent
    target = normalize_quantity(QUANTITIES[key])
    files = sorted(
        path
        for path in root.rglob("reblocking_loop_analysis_*.txt")
        if normalize_quantity(path.stem.removeprefix("reblocking_loop_analysis_")) == target
    )
    if not files:
        print(f"No reblocking files found for {requested!r} under {root}", file=sys.stderr)
        return 1

    datasets = []
    for path in files:
        block_sizes, errors = read_reblocking_file(path)
        if block_sizes:
            label = path.parent.relative_to(root).as_posix()
            datasets.append((label if label != "." else path.parent.name, block_sizes, errors))
        else:
            print(f"Skipping empty reblocking file: {path}", file=sys.stderr)
    if not datasets:
        print(f"No numeric reblocking rows found for {requested!r}", file=sys.stderr)
        return 1

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        if error.name == "matplotlib":
            print("Matplotlib is required. Install it with: python -m pip install -r requirements.txt", file=sys.stderr)
            return 1
        raise

    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    for label, block_sizes, errors in datasets:
        axis.plot(block_sizes, errors, marker="o", linewidth=1.5, markersize=4, label=label)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Block size (samples)")
    axis.set_ylabel("Estimated standard error")
    axis.set_title(f"Reblocking error: {requested}")
    axis.grid(True, which="both", alpha=0.25)
    if len(datasets) > 1:
        axis.legend(title="Run folder", fontsize="small")

    output_name = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    output_path = root / f"reblocking_error_{output_name}.png"
    figure.savefig(output_path, dpi=160)
    print(f"Saved {output_path}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())