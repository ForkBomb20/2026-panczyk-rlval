"""Plot desired vs observed vs actual power for the most-likely BFGS failure."""

import sys
import pickle
import numpy as np
import numpy.core
import matplotlib.pyplot as plt
from fuzzing import DisturbanceDistribution, Do, Da, Ds
from bfgs_falsification import extract

# Shim for pickles saved with numpy 2.x on the cluster
sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray

with open("results/bfgs_results.pkl", "rb") as f:
    data = pickle.load(f)

cfg = data["config"]
depth = data["depth"]
dd = DisturbanceDistribution(
    Do=Do(cfg["sigma_p"], cfg["sigma_drum"]),
    Da=Da(0), Ds=Ds())

failures = [r for r in data["results"] if r["failure"]]
best = min(failures, key=lambda r: r["obj"])

actual = best["actual"]
desired = best["desired"]
reset_xo, disturbances = extract(best["x_vec"], dd, depth)

# Reconstruct observed power: actual + sensor noise
observed = np.empty_like(actual)
observed[0] = actual[0] + reset_xo["p"]
for i in range(1, len(actual)):
    observed[i] = actual[i] + disturbances[i - 1].xo["p"]

timesteps = np.arange(len(actual))

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(timesteps, desired, color="tab:blue", linewidth=2, label="Desired Power")
ax.plot(timesteps, observed, color="tab:orange", linewidth=1.5, label="Observed Power")
ax.plot(timesteps, actual, color="tab:green", linewidth=2, label="Actual Power")
ax.set_xlabel("Time, [s]", fontsize=12)
ax.set_ylabel("Fraction of Total Power", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/mlf_desired_observed_actual.png", dpi=150)
plt.savefig("results/mlf_desired_observed_actual.svg")
print(f"Saved to results/mlf_desired_observed_actual.png")
print(f"obj={best['obj']:.4f}, max_dev={best['max_dev']:.4f}")
