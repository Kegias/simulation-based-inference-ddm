import numpy as np


def summarize_behavior(reaction_times, choices):
    """
    Convert raw reaction-time and choice data into fixed-size summary statistics.

    Summary features:
    1. Mean reaction time
    2. Standard deviation of reaction time
    3. Choice-1 proportion
    4. 10th percentile reaction time
    5. Median reaction time
    6. 90th percentile reaction time
    """
    return np.array([
        np.mean(reaction_times),
        np.std(reaction_times),
        np.mean(choices),
        np.percentile(reaction_times, 10),
        np.percentile(reaction_times, 50),
        np.percentile(reaction_times, 90)
    ])