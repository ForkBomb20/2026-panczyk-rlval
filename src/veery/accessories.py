import numpy as np


def check_spec(history):
    """Checks if a rollout's power trajectory stays within 3% (relative) of desired power.

    Args:
        history (pd.DataFrame): rollout history with 'actual_power' and 'desired_power' columns

    Returns:
        bool: True if actual power never deviates >=3% (relative) from desired power
    """
    pct_dev = np.abs(
        (history["actual_power"] - history["desired_power"]) / history["desired_power"]
    )
    return (pct_dev < 0.03).all()


def percent_diff(ptrue, pdesired):
    return abs(ptrue - pdesired) / pdesired


def robustness(p_actual, p_desired, p_threshold):
    """Calculates the robustness of a trajectory defined by the actual power and the desired power for all timesteps. This is based on signal temporal logic.

    Args:
        p_actual (array): Actual achieved power
        p_desired (array): Desired power at corresponding times to p_actual
        p_threshold (float): power specification as a percent deviation from profile
    """
    # if we're at a terminal node, we won't get power arrays, so return infinity (we're going to take a min of this rho and the path rho, so path rho will always get picked if we're at a terminal node)
    if type(p_actual) != list:
        p_actual = np.array([p_actual])
    if type(p_desired) != list:
        p_desired = np.array([p_desired])
    if len(p_actual) == 0:
        return np.inf
    percent_diffs = abs(np.array(p_actual) - np.array(p_desired)) / np.array(p_desired)
    rho = np.min(p_threshold - percent_diffs)
    return rho


# check an operational spec on the drums!!
