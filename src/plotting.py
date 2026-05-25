import matplotlib.pyplot as plt
import numpy as np


def plot_reaction_time_distribution(reaction_times, save_path=None):
    """
    Plot simulated reaction-time distribution.
    """
    plt.figure(figsize=(7, 4))
    plt.hist(reaction_times, bins=30, alpha=0.8)
    plt.xlabel("Reaction Time")
    plt.ylabel("Frequency")
    plt.title("Simulated Reaction-Time Distribution")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_posterior_parameters(
    posterior_samples,
    true_values,
    posterior_mean,
    param_names,
    save_path=None
):
    """
    Plot posterior distributions for DDM parameters.
    """
    posterior_np = posterior_samples.detach().cpu().numpy()

    fig, axes = plt.subplots(1, len(param_names), figsize=(15, 4))

    for i, name in enumerate(param_names):
        axes[i].hist(posterior_np[:, i], bins=40, alpha=0.8)
        axes[i].axvline(true_values[i].item(), linestyle="--", label="True value")
        axes[i].axvline(posterior_mean[i].item(), linestyle="-", label="Posterior mean")
        axes[i].set_title(name)
        axes[i].set_xlabel("Parameter value")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_true_vs_inferred(results_df, param_names, save_path=None):
    """
    Plot true parameter values against posterior mean estimates.
    """
    plt.figure(figsize=(7, 5))

    x = np.arange(len(param_names))
    width = 0.35

    plt.bar(x - width / 2, results_df["true_value"], width, label="True value")
    plt.bar(x + width / 2, results_df["posterior_mean"], width, label="Posterior mean")

    plt.xticks(x, param_names, rotation=20)
    plt.ylabel("Parameter value")
    plt.title("True vs Inferred DDM Parameters")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_uncertainty_vs_trials(uncertainty_df, param_names, save_path=None):
    """
    Plot posterior uncertainty as a function of number of observed trials.
    """
    plt.figure(figsize=(8, 5))

    for name in param_names:
        subset = uncertainty_df[uncertainty_df["parameter"] == name]
        plt.plot(
            subset["n_trials"],
            subset["posterior_std"],
            marker="o",
            label=name
        )

    plt.xlabel("Number of observed trials")
    plt.ylabel("Posterior standard deviation")
    plt.title("Posterior Uncertainty vs Number of Observed Trials")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()