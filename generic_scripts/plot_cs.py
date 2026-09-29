from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


data_path = Path(__file__).with_name("Cs.txt")
data = np.loadtxt(data_path, comments="#")

figure, axis = plt.subplots()
axis.plot(data[:, 0], data[:, 1], marker="o", markersize=3, linewidth=1)
axis.set_xlabel("r")
axis.set_ylabel("Cs(r)")
axis.grid(True, alpha=0.25)
figure.tight_layout()

output_path = data_path.with_name("Cs.png")
figure.savefig(output_path, dpi=160)
print(f"Saved {output_path}")
plt.show()