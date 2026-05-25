import numpy as np


def simulate_ddm_single_trial(
    drift_rate,
    boundary,
    non_decision_time,
    noise=1.0,
    dt=0.005,
    max_time=5.0
):
    """
    Simulate a single drift-diffusion model trial.

    Parameters
    ----------
    drift_rate : float
        Rate of evidence accumulation.
    boundary : float
        Decision boundary height.
    non_decision_time : float
        Time added for perceptual/motor processes.
    noise : float
        Diffusion noise scale.
    dt : float
        Simulation time step.
    max_time : float
        Maximum decision time before forced stopping.

    Returns
    -------
    reaction_time : float
        Simulated reaction time.
    choice : int
        1 if upper boundary is reached, 0 if lower boundary is reached.
    """
    evidence = 0.0
    t = 0.0

    while t < max_time:
        evidence += drift_rate * dt + noise * np.sqrt(dt) * np.random.randn()
        t += dt

        if evidence >= boundary:
            return t + non_decision_time, 1

        if evidence <= -boundary:
            return t + non_decision_time, 0

    return max_time + non_decision_time, int(evidence > 0)


def simulate_ddm_dataset(
    drift_rate,
    boundary,
    non_decision_time,
    n_trials=200,
    noise=1.0,
    dt=0.005,
    max_time=5.0
):
    """
    Simulate multiple drift-diffusion model trials.

    Returns
    -------
    reaction_times : np.ndarray
        Array of reaction times.
    choices : np.ndarray
        Array of binary choices.
    """
    reaction_times = []
    choices = []

    for _ in range(n_trials):
        rt, choice = simulate_ddm_single_trial(
            drift_rate=drift_rate,
            boundary=boundary,
            non_decision_time=non_decision_time,
            noise=noise,
            dt=dt,
            max_time=max_time
        )
        reaction_times.append(rt)
        choices.append(choice)

    return np.array(reaction_times), np.array(choices)