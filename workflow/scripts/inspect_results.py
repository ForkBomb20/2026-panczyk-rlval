# import pickle
# with open("results/bfgs_results.pkl", "rb") as f:
#     data = pickle.load(f)

# for i, r in enumerate(data["results"]):
#     print(f"[{i+1}] failure={r['failure']}, obj={r['obj']:.4f}, "
#         f"max_dev={r['max_dev']:.4f}, iters={r['n_iterations']}")

# # Get the most-likely failure (highest magnitude obj)
# failures = [r for r in data["results"] if r["failure"]]
# best = min(failures, key=lambda r: r["obj"])  # most negative = most likely
# print(f"\nMost likely failure: obj={best['obj']:.4f}, max_dev={best['max_dev']:.4f}")

# # The optimized disturbance vector that produced it:
# x_vec = best["x_vec"]  # can feed back into extract() to get the Disturbance sequence


# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from fuzzing import DisturbanceDistribution, Do, Da, Ds
# from bfgs_falsification import extract

# with open("results/bfgs_results.pkl", "rb") as f:
#     data = pickle.load(f)

# cfg = data["config"]
# depth = data["depth"]
# dd = DisturbanceDistribution(
#     Do=Do(cfg["sigma_p"], cfg["sigma_drum"]),
#     Da=Da(0), Ds=Ds())

# # Get the most-likely failure
# failures = [r for r in data["results"] if r["failure"]]
# best = min(failures, key=lambda r: r["obj"])
# reset_xo, disturbances = extract(best["x_vec"], dd, depth)

# # Extract per-step disturbance values
# timesteps = np.arange(depth)
# power_noise = np.array([x.xo["p"] for x in disturbances])
# drum_noise = np.array([x.xo["drum_angles"] for x in disturbances])  # shape (depth, 8)

# fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# # Power sensor noise
# axes[0].plot(timesteps, power_noise, color="red", linewidth=1)
# axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.5)
# axes[0].set_ylabel("Power sensor noise")
# axes[0].set_title(f"Most-Likely Failure Disturbance (obj={best['obj']:.4f})")

# # Drum angle sensor noise (all 8 drums)
# for d in range(8):
#     axes[1].plot(timesteps, drum_noise[:, d], alpha=0.7, linewidth=0.8, label=f"Drum {d+1}")
# axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.5)
# axes[1].set_ylabel("Drum angle sensor noise")
# axes[1].set_xlabel("Time [s]")
# axes[1].legend(ncol=4, fontsize=8)

# plt.tight_layout()
# plt.savefig("results/most_likely_disturbance.svg", dpi=300)
# plt.show()

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("results/bfgs_results.pkl", "rb") as f:
    data = pickle.load(f)

cfg = data["config"]
failures = [r for r in data["results"] if r["failure"]]
best = min(failures, key=lambda r: r["obj"])

actual = best["actual"]
desired = best["desired"]
timesteps = np.arange(len(actual))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(timesteps, desired, color="black", linewidth=1.5, label="Desired power")
ax.plot(timesteps, actual, color="red", linewidth=1.5, label="Actual power (MLF)")
ax.plot(timesteps, desired * 1.03, color="black", linestyle="--", linewidth=1, label="±3% spec")
ax.plot(timesteps, desired * 0.97, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Power fraction")
ax.set_title(f"Most-Likely Failure (obj={best['obj']:.4f}, max_dev={best['max_dev']:.4f})")
ax.legend()
plt.tight_layout()
plt.savefig("results/most_likely_failure_power.svg", dpi=300)
plt.show()