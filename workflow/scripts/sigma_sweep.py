"""Sweep observation noise sigma to find the MLF falsification threshold.

For each sigma value, runs parallel BFGS MLF restarts and records
whether any failures were found. The goal is to find the maximum sigma
at which the controller can no longer be falsified within the given
computational budget.

Assumes sigma_p = sigma_drum = sigma (symmetric sweep).
"""

import numpy as np
from pathlib import Path
from multiprocessing import Pool
import pickle
from bfgs_falsification import _run_one_restart

# --- Configuration ---
sigmas = [0.005, 0.004, 0.003, 0.002, 0.001, 0.0005]
n_opt = 10          # restarts per sigma level
max_iter = 50       # BFGS iterations per restart
smoothness = 0.1
n_workers = 10      # parallel processes (set <= available cores)
model_path = "runs/train_fivemillion/models/best_model.zip"

results_dir = Path("results/sigma_sweep")
results_dir.mkdir(parents=True, exist_ok=True)

all_results = {}

for sigma in sigmas:
    print(f"\n{'='*60}")
    print(f"  sigma = {sigma}")
    print(f"{'='*60}")

    cfg = {
        "sigma_p": sigma,
        "sigma_drum": sigma,
        "smoothness": smoothness,
        "max_iter": max_iter,
        "n_opt": n_opt,
        "model_path": model_path,
    }

    seeds = np.random.SeedSequence(42).spawn(n_opt)
    worker_args = [(i, int(s.generate_state(1)[0]), cfg) for i, s in enumerate(seeds)]

    with Pool(n_workers) as pool:
        results = pool.map(_run_one_restart, worker_args)

    n_failures = sum(1 for r in results if r["failure"])
    best_obj = min(r["obj"] for r in results)
    best_max_dev = max(r["max_dev"] for r in results)
    failure_objs = [r["obj"] for r in results if r["failure"]]

    summary = {
        "sigma": sigma,
        "n_failures": n_failures,
        "n_opt": n_opt,
        "best_obj": best_obj,
        "best_max_dev": best_max_dev,
        "failure_objs": failure_objs,
    }
    all_results[sigma] = {"summary": summary, "results": results}

    print(f"\n  sigma={sigma}: {n_failures}/{n_opt} failures, "
          f"best_obj={best_obj:.4f}, best_max_dev={best_max_dev:.4f}")

    # Stop early if no failures found
    if n_failures == 0:
        print(f"\n  No failures at sigma={sigma}, stopping sweep.")
        break

# --- Save results ---
save_path = results_dir / "sigma_sweep_results.pkl"
with open(save_path, "wb") as f:
    pickle.dump(all_results, f)

# --- Print summary table ---
print(f"\n{'='*60}")
print(f"  Sigma Sweep Summary")
print(f"{'='*60}")
print(f"  {'sigma':>8}  {'failures':>10}  {'best_obj':>10}  {'best_max_dev':>12}")
print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}")
for sigma in sigmas:
    if sigma not in all_results:
        break
    s = all_results[sigma]["summary"]
    print(f"  {s['sigma']:>8.4f}  {s['n_failures']:>5}/{s['n_opt']:<4}  "
          f"{s['best_obj']:>10.4f}  {s['best_max_dev']:>12.4f}")

print(f"\nResults saved to {save_path}")
