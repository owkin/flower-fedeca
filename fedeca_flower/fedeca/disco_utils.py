from copy import deepcopy
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import binom


def compute_uncentered_moment(data, order, weights=None):
    """Compute the uncentered moment."""
    if weights is None:
        weights = np.ones(len(data.index))

    # TODO categorical variables need a special treatment

    if isinstance(data, (pd.DataFrame, pd.Series)):
        moment = data.select_dtypes(include=np.number).pow(order)
        moment = moment.mul(weights, axis=0)
        moment = moment.sum(skipna=True)
        moment /= weights.sum()

    elif isinstance(data, np.ndarray):
        moment = np.power(data, order)
        moment = np.multiply(moment, weights)
        moment = np.sum(moment, axis=0)
        moment /= weights.sum()

    else:
        raise NotImplementedError(
            "Only DataFrame or numpy array are currently handled."
        )
    return moment


# pylint: disable=deprecated-typing-alias
def compute_centered_moment(uncentered_moments: list[Any]):
    """Compute the centered moment of order k."""
    mean = np.copy(uncentered_moments[0])
    order = len(uncentered_moments)
    result = (-mean) ** order  # i+1 = 0
    # We will go over the list of moments to add Newton's binomial
    # expansion formula terms one by one, where the current
    # moment is ahead of i by 1 hence we call it moment_i_plus_1
    for i, moment_i_plus_1 in enumerate(uncentered_moments):
        temp = (-mean) ** (order - i - 1)
        temp *= moment_i_plus_1  # the power is already computed
        temp *= binom(order, i + 1)
        result += temp
    return result


# pylint: disable=deprecated-typing-alias
def aggregation_mean(local_means: list[Any], n_local_samples: list[int]):
    """Aggregate local means."""
    tot_samples = np.nan_to_num(np.copy(n_local_samples[0]), nan=0, copy=False)
    tot_mean = np.nan_to_num(np.copy(local_means[0]), nan=0, copy=False)
    for mean, n_sample in zip(local_means[1:], n_local_samples[1:]):
        mean = np.nan_to_num(mean, nan=0, copy=False)
        tot_mean *= tot_samples / (tot_samples + n_sample)
        tot_mean += mean * (n_sample / (tot_samples + n_sample))
        tot_samples += n_sample

    return tot_mean


def compute_global_moments(shared_states):
    """Aggregate local moments to compute global moments."""
    tot_uncentered_moments = [
        aggregation_mean(
            [s[f"moment{k}"] for s in shared_states],
            [s["n_samples"] for s in shared_states],
        )
        for k in range(1, 2 + 1)
    ]
    n_samples = sum([s["n_samples"].iloc[0] for s in shared_states])

    results = {
        f"global_centered_moment_{k}": compute_centered_moment(
            tot_uncentered_moments[:k]
        )
        for k in range(1, 2 + 1)
    }

    results.update(
        {
            f"global_uncentered_moment_{k+1}": moment
            for k, moment in enumerate(tot_uncentered_moments)
        }
    )

    results.update({"total_n_samples": n_samples})

    return results


def compute_global_survival_statistics(shared_states):
    """Aggregate different needed statistics."""
    global_sum_features_on_events = np.zeros_like(
        shared_states[0]["sum_features_on_events"]
    )

    # find out all distinct values while avoiding duplicates
    distinct_event_times = []
    for ls_and_dv in shared_states:
        distinct_event_times += ls_and_dv["distinct_event_times"]
    distinct_event_times = list(set(distinct_event_times))
    distinct_event_times.sort()
    # count them
    num_global_event_times = len(distinct_event_times)
    # aggregate statistics by suming
    for ls_and_dv in shared_states:
        global_sum_features_on_events += ls_and_dv["sum_features_on_events"]
    # Count the number of tied event times for each client
    list_number_events_by_time = []
    total_number_samples = sum(
        [ls_and_dv["total_number_samples"] for ls_and_dv in shared_states]
    )

    # Very weird to double check that it cannot be written in a more readable way
    for ls_and_dv in shared_states:
        global_ndt = []
        for i, e in enumerate(distinct_event_times):
            if e in ls_and_dv["distinct_event_times"]:
                idx = ls_and_dv["distinct_event_times"].index(e)
                global_ndt.append(ls_and_dv["number_events_by_time"][idx])
            else:
                global_ndt.append(0)
        list_number_events_by_time.append(global_ndt)

    # We add what should amount at number events by time if weights=1
    weights_counts_on_events = []
    for d in distinct_event_times:
        weights_counts_on_event = 0.0
        for ls_and_dv in shared_states:
            if d in ls_and_dv["distinct_event_times"]:
                idx = ls_and_dv["distinct_event_times"].index(d)
                weights_counts_on_event += ls_and_dv["weights_counts_on_events"][idx]
        weights_counts_on_events.append(weights_counts_on_event)

    results = {}
    results["global_survival_statistics"] = {}
    results["global_survival_statistics"]["distinct_event_times"] = distinct_event_times
    results["global_survival_statistics"][
        "global_sum_features_on_events"
    ] = global_sum_features_on_events
    results["global_survival_statistics"][
        "list_number_events_by_time"
    ] = list_number_events_by_time
    results["global_survival_statistics"][
        "num_global_events_time"
    ] = num_global_event_times
    results["global_survival_statistics"]["total_number_samples"] = total_number_samples
    results["global_survival_statistics"][
        "weights_counts_on_events"
    ] = weights_counts_on_events
    results["moments"] = shared_states[0]["moments"]

    return results


def build_global_gradient_and_hessian(
    shared_states,
    tol: float = 1e-16,
):
    """Compute global gradient and Hessian."""
    # Server is stateless need to continuously feed it with
    # global_survival_statistics

    global_survival_statistics = shared_states[0]["global_survival_statistics"]
    # It is important to use deepcopy to avoid side effect
    # Otherwise, the value of self.global_sum_features_on_events will change
    # This is already weighted
    gradient = deepcopy(global_survival_statistics["global_sum_features_on_events"])
    ll = 0.0
    try:
        gradient_shape = [e for e in gradient.shape if e > 1][0]
    except IndexError:
        gradient_shape = 1

    risk_phi_stats_list = [e["local_phi_stats"] for e in shared_states]
    risk_phi_list = [e["risk_phi"] for e in risk_phi_stats_list]
    risk_phi_x_list = [e["risk_phi_x"] for e in risk_phi_stats_list]
    risk_phi_x_x_list = [e["risk_phi_x_x"] for e in risk_phi_stats_list]

    distinct_event_times = global_survival_statistics["distinct_event_times"]

    hessian = np.zeros((gradient_shape, gradient_shape))

    # Needed for robust estimation of SE
    global_risk_phi_list = []
    global_risk_phi_x_list = []
    # We first sum over each event
    for idxd, _ in enumerate(distinct_event_times):
        # This factor amounts to d_i the number of events per time i if no weights
        # otherwise it's the sum of the score of all d_i events
        weighted_average = global_survival_statistics["weights_counts_on_events"][idxd]

        # We initialize both tensors at zeros for numerators (all denominators are
        # scalar)
        numerator = np.zeros(risk_phi_x_list[0][0].shape)
        # The hessian has several terms due to deriving quotient of functions u/v
        first_numerator_hessian = np.zeros((gradient_shape, gradient_shape))
        denominator = 0.0
        if np.allclose(weighted_average, 0.0):
            continue
        for i in range(len(risk_phi_stats_list)):
            numerator += risk_phi_x_list[i][idxd]
            denominator += risk_phi_list[i][idxd]
            first_numerator_hessian += risk_phi_x_x_list[i][idxd]

        global_risk_phi_list.append(denominator)
        global_risk_phi_x_list.append(numerator)
        # denominator being a sum of exponential it's always positive

        assert denominator >= 0.0, "the sum of exponentials is negative..."
        denominator = max(denominator, tol)
        denominator_squared = max(denominator**2, tol)
        c = numerator / denominator
        ll -= weighted_average * np.log(denominator)
        gradient -= weighted_average * np.squeeze(c)
        hessian -= weighted_average * (
            (first_numerator_hessian / denominator)
            - (np.multiply.outer(numerator, numerator) / denominator_squared)
        )

    return {
        "hessian": hessian,
        "gradient": gradient,
        "second_part_ll": ll,
        "gradient_shape": gradient_shape,
        "global_risk_phi_list": global_risk_phi_list,
        "global_risk_phi_x_list": global_risk_phi_x_list,
    }


def get_final_cox_model_function(
    models,
    hessians,
    lls,
    num_seeds: int,
    global_moments_list,
    standardize_data: bool,
    global_robust_statistics: dict[str, Any],
):
    """Retrieve first converged Cox model and corresponding hessian."""

    if standardize_data:

        computed_vars_list = [g["vars"] for g in global_moments_list]
        # We need to match pandas standardization across seeds
        bias_correction_list = [g["bias_correction"] for g in global_moments_list]
        computed_stds_list = [
            computed_v.transform(lambda x: sqrt(x * bias_c + 1e-16))
            for computed_v, bias_c in zip(computed_vars_list, bias_correction_list)
        ]

    else:
        computed_stds_list = [
            pd.Series(np.ones((models[0].fc1.weight.shape)).squeeze())
            for _ in range(num_seeds)
        ]
        global_moments_list = [{} for _ in range(num_seeds)]

    # We unstandardize the weights across seeds
    final_params_list = [
        model.fc1.weight.data.numpy().squeeze() / computed_stds.to_numpy()
        for model, computed_stds in zip(models, computed_stds_list)
    ]

    # Robust estimation
    global_robust_statistics["global_moments"] = global_moments_list[0]

    return (
        hessians,
        lls,
        final_params_list,
        computed_stds_list,
        global_robust_statistics,
    )


def compute_summary_function(final_params, variance_matrix, alpha=0.05):
    """Compute summary function."""
    se = np.sqrt(variance_matrix.diagonal())
    ci = 100 * (1 - alpha)
    z = stats.norm.ppf(1 - alpha / 2)
    Z = final_params / se
    U = Z**2
    pvalues = stats.chi2.sf(U, 1)
    summary = pd.DataFrame()
    summary["coef"] = final_params
    summary["se(coef)"] = se
    summary[f"coef lower {round(ci)}%"] = final_params - z * se
    summary[f"coef upper {round(ci)}%"] = final_params + z * se
    summary["z"] = Z
    summary["p"] = pvalues

    return summary
