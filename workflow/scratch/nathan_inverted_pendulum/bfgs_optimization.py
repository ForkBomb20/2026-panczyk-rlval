"""Use BFGS Optimization to Find Most Likely Failure modes for inverted pendulum."""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from inverted_pendulum import (
    Disturbance,
    DisturbanceDistribution,
    InvertedPendulum,
    plot_observation_disturbances,
    plot_trajectories,
)


class NominalTrajectoryDistribution:
    """NominalTrajectoryDistribution, bundles Ps (initial state dist), D (disturbance dist), d (depth)."""

    def __init__(self, Ps, D, d):
        self.Ps = Ps
        self.D = D
        self.d = d

    def logpdf(self, s0, disturbances):
        """Log probability of a trajectory under the nominal distribution."""
        lp = 0.0
        lp += self.Ps[0].logpdf(s0[0]) # Prob of Theta
        lp += self.Ps[1].logpdf(s0[1]) # Prob of Omega
        for x in disturbances:
            if self.D.Do_dim > 0:
                lp += np.sum(self.D.Do.logpdf(x.xo))
            if self.D.Da_dim > 0:
                lp += np.sum(self.D.Da.logpdf(x.xa))
            if self.D.Ds_dim > 0:
                lp += np.sum(self.D.Ds.logpdf(x.xs))
        return lp


def extract(x_vec, disturbance_dist):
    """extract(sys.env, x), unpack optimization vector into initial state and disturbances."""
    s0 = (x_vec[0], x_vec[1])
    depth = (len(x_vec) - 2) // disturbance_dist.dim
    disturbances = []
    idx = 2
    for t in range(depth):
        if disturbance_dist.Do_dim > 0:
            xo = x_vec[idx : idx + disturbance_dist.Do_dim].reshape(2, 1)
            idx += disturbance_dist.Do_dim
        else:
            xo = np.zeros((2, 1))

        if disturbance_dist.Da_dim > 0:
            xa = x_vec[idx : idx + disturbance_dist.Da_dim]
            idx += disturbance_dist.Da_dim
        else:
            xa = np.array([0.0])

        if disturbance_dist.Ds_dim > 0:
            xs = x_vec[idx : idx + disturbance_dist.Ds_dim].reshape(2, 1)
            idx += disturbance_dist.Ds_dim
        else:
            xs = np.zeros((2, 1))

        disturbances.append(Disturbance(xo, xa, xs))
    return s0, disturbances


def robustness(thetas, smoothness=0.0):
    """STL robustness for G(|theta| <= pi/4). Positive = satisfying, negative = violating."""
    margins = np.pi / 4 - np.abs(thetas)
    if smoothness > 0:
        return -smoothness * np.log(np.sum(np.exp(-margins / smoothness)))
    return np.min(margins)


def likelihood_objective(x_vec, sys, nominal_dist, smoothness=0.0):
    """likelihood_objective, drives BFGS toward the most likely failure.

    Failure case: return -pdf(p, tau) so the optimizer maximizes failure likelihood.
    Non-failure case: return robustness, guiding the optimizer toward the failure boundary.
    """
    s0, disturbances = extract(x_vec, nominal_dist.D)
    thetas, omegas = sys.rollout_with_disturbances(s0, disturbances)

    if not sys.check_spec(thetas):  # isfailure
        logp = nominal_dist.logpdf(s0, disturbances)
        return -logp / nominal_dist.d # Note this is the negative log likelihood so that minimizing this correspond to maximizing actual log likelihood
    else:
        return robustness(thetas, smoothness) # Minimizing will push toward ANY failure
    
def make_objective(sys, nominal_dist, smoothness=0.0):
    """Returns f(x_vec) -> scalar, suitable for scipy.optimize.minimize."""
    def objective(x_vec):
        return likelihood_objective(x_vec, sys, nominal_dist, smoothness)
    return objective


if __name__ == "__main__":
    sig = 0.15
    disturbance_dist = DisturbanceDistribution(
        Do=norm(0, sig), Da=norm(0, 0), Ds=norm(0, 0), Do_dim=2
    )
    sys = InvertedPendulum(m=1.0, l=1.0, dt=0.05)
    runtime = 10
    depth = int(np.ceil(runtime / sys.dt))
    Ps = (sys.theta_dist, sys.omega_dist)
    nominal_dist = NominalTrajectoryDistribution(Ps, disturbance_dist, depth)
    obj = make_objective(sys, nominal_dist, smoothness=0.1)

    n_opt = 20
    n_rollouts = 300
    vec_len = 2 + disturbance_dist.dim * depth
    bounds = [(None, None)] * vec_len
    bounds[0] = (-np.pi / 16, np.pi / 16)
    bounds[1] = (-1.0, 1.0)

    failure_thetas = []
    for i in range(n_opt):
        x0 = np.zeros(vec_len)
        x0[0], x0[1] = sys.get_initial_state()
        x0[2:] = disturbance_dist.Do.rvs(size=disturbance_dist.Do_dim * depth)

        result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds)
        s0_opt, dist_opt = extract(result.x, disturbance_dist)
        thetas_opt, _ = sys.rollout_with_disturbances(s0_opt, dist_opt)

        is_failure = not sys.check_spec(thetas_opt)
        print(f"[{i+1}/{n_opt}] converged={result.success}, failure={is_failure}, "
              f"obj={result.fun:.4f}, max|θ|={np.max(np.abs(thetas_opt)):.4f}")
        if is_failure:
            failure_thetas.append(thetas_opt)

    print(f"\nFound {len(failure_thetas)}/{n_opt} failures")

    # Run nominal rollouts for comparison
    import matplotlib.pyplot as plt
    theta_matrix = np.zeros((depth, n_rollouts))
    for r in range(n_rollouts):
        thetas, omegas, xs = sys.rollout(runtime, disturbance_dist)
        theta_matrix[:, r] = thetas

    fig, ax = plt.subplots()
    ax.plot(theta_matrix, color="steelblue", alpha=0.3, linewidth=0.8)
    for thetas_f in failure_thetas:
        ax.plot(thetas_f, color="red", alpha=0.7, linewidth=1.5)
    ax.axhline(np.pi / 4, color="black", linestyle="--", linewidth=1, label="Spec limit")
    ax.axhline(-np.pi / 4, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("t, [steps]")
    ax.set_ylabel(r"$\theta$, [rad]")
    ax.set_title(rf"BFGS Failures (n={len(failure_thetas)}) vs Nominal Rollouts ($\sigma = {sig}$)")
    ax.legend()
    plt.savefig("./results/bfgs_failures.svg", dpi=300)
    plt.show()