import glob
import re
import os
import matplotlib.pyplot as plt

K12_list = []
E_list = []
Eerr_list = []
P2_list = []
P2err_list = []

for folder in sorted(glob.glob("piecewiseVMC_*_K12*_N*")):
    match = re.search(r"K12([\d.]+)_h", folder)
    results_file = os.path.join(folder, "VMC_flips", "results.txt")
    if match is None or not os.path.isfile(results_file):
        continue

    K12 = float(match.group(1))
    data = {}
    with open(results_file) as f:
        for line in f:
            parts = line.split()
            data[parts[0]] = parts[1]

    N1 = float(data["N1"])
    E = float(data["E_VMC_per_particle"])
    Eerr = float(data["E_VMC_err"]) / N1

    K12_list.append(K12)
    E_list.append(E)
    Eerr_list.append(Eerr)
    P2_list.append(float(data["P2"]))
    P2err_list.append(float(data["P2_err"]))

plt.figure()
plt.errorbar(K12_list, E_list, yerr=Eerr_list, fmt="o-")
plt.xlabel("K12")
plt.ylabel("E/N")
plt.tight_layout()
plt.savefig("energy_vs_K12.png")

plt.figure()
plt.errorbar(K12_list, P2_list, yerr=P2err_list, fmt="o-")
plt.xlabel("K12")
plt.ylabel("P^2")
plt.tight_layout()
plt.savefig("P2_vs_K12.png")

plt.show()
