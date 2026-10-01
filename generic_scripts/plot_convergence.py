from pathlib import Path

import numpy as np


def read_convergence_file(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def main() -> int:
    root = Path(__file__).resolve().parent
    files = sorted(root.rglob("convergence_*.dat"))
    if not files:
        print(f"No convergence_*.dat files found under {root}")
        return 1

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        if error.name == "matplotlib":
            print("Matplotlib is required. Install it with: python -m pip install -r requirements.txt")
            return 1
        raise

    figure, (axis_e, axis_p2) = plt.subplots(2, 1, sharex=True, figsize=(8, 7), layout="constrained")
    for path in files:
        data = read_convergence_file(path)
        if data.size == 0:
            print(f"Skipping empty convergence file: {path}")
            continue
        label = path.parent.relative_to(root).as_posix()
        label = label if label != "." else path.stem
        step, e_per_n, p2 = data[:, 0], data[:, 1], data[:, 2]
        axis_e.plot(step, e_per_n, linewidth=1, label=label)
        axis_p2.plot(step, p2, linewidth=1, label=label)

    axis_e.set_ylabel("E/N")
    axis_e.grid(True, alpha=0.25)
    axis_p2.set_xlabel("MC step")
    axis_p2.set_ylabel(r"$P^2$")
    axis_p2.grid(True, alpha=0.25)
    if len(files) > 1:
        axis_e.legend(fontsize="small")

    output_path = root / "convergence.png"
    figure.savefig(output_path, dpi=160)
    print(f"Saved {output_path}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
