import torch

from src.ddm_simulator import simulate_ddm_dataset
from src.summary_statistics import summarize_behavior


def ddm_sbi_simulator(theta, n_trials=200):
    """
    SBI-compatible simulator.

    Parameters
    ----------
    theta : torch.Tensor
        Tensor containing [drift_rate, boundary, non_decision_time].
    n_trials : int
        Number of simulated trials per parameter setting.

    Returns
    -------
    torch.Tensor
        Summary-statistic vector.
    """
    theta_np = theta.detach().cpu().numpy()

    drift_rate = theta_np[0]
    boundary = theta_np[1]
    non_decision_time = theta_np[2]

    reaction_times, choices = simulate_ddm_dataset(
        drift_rate=drift_rate,
        boundary=boundary,
        non_decision_time=non_decision_time,
        n_trials=n_trials
    )

    summary = summarize_behavior(reaction_times, choices)

    return torch.tensor(summary, dtype=torch.float32)


def run_trial_count_experiment(
    posterior,
    true_params,
    n_observed_trials,
    n_posterior_samples=1000
):
    """
    Infer DDM parameters using different numbers of observed trials.

    This is used to study how posterior uncertainty changes as the amount
    of behavioural data increases.
    """
    reaction_times, choices = simulate_ddm_dataset(
        drift_rate=true_params["drift_rate"],
        boundary=true_params["boundary"],
        non_decision_time=true_params["non_decision_time"],
        n_trials=n_observed_trials
    )

    x_observed = summarize_behavior(reaction_times, choices)
    x_observed_tensor = torch.tensor(x_observed, dtype=torch.float32)

    posterior_samples = posterior.sample(
        (n_posterior_samples,),
        x=x_observed_tensor,
        show_progress_bars=False
    )

    posterior_mean = posterior_samples.mean(dim=0)
    posterior_std = posterior_samples.std(dim=0)

    return posterior_mean, posterior_std