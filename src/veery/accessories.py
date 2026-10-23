import numpy as np


def check_spec(history):
    """Checks if a rollout's power trajectory stays within 3% (relative) of desired power.

    Args:
        history (pd.DataFrame): rollout history with 'actual_power' and 'desired_power' columns

    Returns:
        bool: True if actual power never deviates >=3% (relative) from desired power
    """
    pct_dev = np.abs((history["Actual"] - history["Desired"]) / history["Desired"])
    return (pct_dev < 0.03).all()


def percent_diff(ptrue, pdesired):
    return abs(ptrue - pdesired) / pdesired


def robustness(actual, desired, threshold):
    """Calculates the robustness of a trajectory defined by the actual power and the desired power for all timesteps. This is based on signal temporal logic.

    Args:
        actual (array): Actual achieved output
        desired (array): Desired output at corresponding times to actual
        threshold (float): power specification as a percent deviation from profile
    """
    # if we're at a terminal node, we won't get power arrays, so return infinity (we're going to take a min of this rho and the path rho, so path rho will always get picked if we're at a terminal node)
    if type(actual) != list:
        actual = np.array([actual])
    if type(desired) != list:
        desired = np.array([desired])
    if len(actual) == 0:
        return np.inf
    percent_diffs = abs(np.array(actual) - np.array(desired)) / np.array(desired)
    rho = np.min(threshold - percent_diffs)
    return rho
