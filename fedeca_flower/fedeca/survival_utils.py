from math import sqrt

import numpy as np
import torch
from pandas.api.types import is_numeric_dtype


def build_X_y_function(
    data_from_opener,
    event_col,
    duration_col,
    treated_col,
    target_cols=None,
    standardize_data=True,
    propensity_model=None,
    cox_fit_cols=None,
    propensity_fit_cols=None,
    tol=1e-16,
    training_strategy="webdisco",
    shared_state={},
    global_moments={},
):
    """Build the inputs for a propensity model and for a Cox model and y."""
    # We need y to be in the format (2*event-1)*duration
    data_from_opener["time_multiplier"] = [
        2.0 * e - 1.0 for e in data_from_opener[event_col].tolist()
    ]

    # No funny business irrespective of the convention used
    y = np.abs(data_from_opener[duration_col]) * data_from_opener["time_multiplier"]
    y = y.to_numpy().astype("float64")
    data_from_opener.drop(columns=["time_multiplier"], inplace=True)

    # TODO very dangerous, to replace by removing client_identifier
    # in both cases this SHOULD NOT BE inplace
    string_columns = [
        col
        for col in data_from_opener.columns
        if not (is_numeric_dtype(data_from_opener[col]))
    ]
    data_from_opener = data_from_opener.drop(columns=string_columns)

    # We drop the targets from X
    if target_cols is None:
        target_cols = [event_col, duration_col]
    columns_to_drop = target_cols
    X = data_from_opener.drop(columns=columns_to_drop)

    if propensity_model is not None:
        assert treated_col is not None
        if training_strategy == "iptw":
            X = X.loc[:, [treated_col]]
        elif training_strategy == "aiptw":
            if len(cox_fit_cols) > 0:
                X = X.loc[:, [treated_col] + cox_fit_cols]
            else:
                pass
    else:
        assert training_strategy == "webdisco"
        if len(cox_fit_cols) > 0:
            X = X.loc[:, cox_fit_cols]
        else:
            pass

    # If X is to be standardized we do it
    if standardize_data:
        if shared_state:
            # Careful this shouldn't happen apart from the predict
            means = shared_state["global_uncentered_moment_1"]
            vars = shared_state["global_centered_moment_2"]
            # Careful we need to match pandas and use unbiased estimator
            bias_correction = (shared_state["total_n_samples"]) / float(
                shared_state["total_n_samples"] - 1
            )
            global_moments = {
                "means": means,
                "vars": vars,
                "bias_correction": bias_correction,
            }
            stds = vars.transform(lambda x: sqrt(x * bias_correction + tol))
            X = X.sub(means)
            X = X.div(stds)
        else:
            X = X.sub(global_moments["means"])
            stds = global_moments["vars"].transform(
                lambda x: sqrt(x * global_moments["bias_correction"] + tol)
            )
            X = X.div(stds)

    X = X.to_numpy().astype("float64")

    # If we have a propensity model we need to build X without the targets AND the
    # treated column
    if propensity_model is not None:
        # We do not normalize the data for the propensity model !!!
        Xprop = data_from_opener.drop(columns=columns_to_drop + [treated_col])
        if propensity_fit_cols is not None:
            Xprop = Xprop[propensity_fit_cols]
        Xprop = Xprop.to_numpy().astype("float64")
    else:
        Xprop = None

    # If WebDisco is used without propensity treated column does not exist
    if treated_col is not None:
        treated = (
            data_from_opener[treated_col].to_numpy().astype("float64").reshape((-1, 1))
        )
    else:
        treated = None

    return (X, y, treated, Xprop, global_moments)


def compute_X_y_and_propensity_weights_function(
    X, y, treated, Xprop, propensity_model, tol=1e-16
):
    """Build appropriate X, y and weights from raw output of opener."""
    if propensity_model is not None:
        assert (
            treated is not None
        ), """If you are using a propensity model the Treated
        column should be available"""
        assert np.all(
            np.isin(np.unique(treated.astype("uint8"))[0], [0, 1])
        ), "The treated column should have all its values in set([0, 1])"
        Xprop = torch.from_numpy(Xprop)

        with torch.no_grad():
            propensity_scores = propensity_model(Xprop)

        # print(f"{treated.shape = }")
        propensity_scores = propensity_scores.detach().numpy()
        # print(f"{propensity_scores.shape = }")

        # We robustify the division
        weights = treated * 1.0 / np.maximum(propensity_scores, tol) + (
            1 - treated
        ) * 1.0 / (np.maximum(1.0 - propensity_scores, tol))
    else:
        weights = np.ones((X.shape[0], 1))
    # print(f"{weights.shape = }")
    return X, y, weights


def compute_q_k(
    X_norm,
    y,
    scaled_variance_matrix,
    distinct_event_times,
    weights_counts_on_events,
    risk_phi,
    risk_phi_x,
    score,
    weights,
):
    """Compute local bricks for Q."""
    n, n_features = X_norm.shape
    phi_k = np.zeros((n, n_features))
    current_client_indices = np.arange(n).tolist()
    weights_counts_on_events_cumsum = np.concatenate(
        [wc.reshape((1, 1)) for wc in weights_counts_on_events],
        axis=0,
    )
    s0s_cumsum = np.concatenate(
        [risk_phi_s.reshape((1, 1)) for risk_phi_s in risk_phi],
        axis=0,
    )
    s1s_cumsum = np.concatenate(
        [risk_phi_x_s.reshape((1, n_features)) for risk_phi_x_s in risk_phi_x],
        axis=0,
    )
    # # of size (i + 1, n_features) this should be term by term
    s1_over_s0_cumsum = s1s_cumsum / (s0s_cumsum)

    # The division should happen term by term
    weights_over_s0_cumsum = weights_counts_on_events_cumsum / s0s_cumsum

    for i in current_client_indices:
        # This is the crux of the implementation, we only have to sum on times
        # with events <= ti
        # as otherwise delta_j = 0 and therefore the term don't contribute
        ti = np.abs(y[i])

        compatible_event_times = [
            idx for idx, td in enumerate(distinct_event_times) if td <= ti
        ]
        # It can happen that we have a censorship that happens before any
        # event times
        if len(compatible_event_times) > 0:
            # distinct_event_times is sorted so we can do that
            max_distinct_event_times = max(compatible_event_times)
            # These are the only indices of the sum, which will be active
            not_Rs_i = np.arange(max_distinct_event_times + 1)

            # Quantities below are global and used onl alread shared quantities
            s1_over_s0_in_sum = s1_over_s0_cumsum[not_Rs_i]
            weights_over_s0_in_sum = weights_over_s0_cumsum[not_Rs_i]

        else:
            # There is nothing in the sum we'll add nothing
            s1_over_s0_in_sum = np.zeros((n_features))
            weights_over_s0_in_sum = 0.0

        # Second term and 3rd term
        phi_i = -score[i] * (
            weights_over_s0_in_sum * (X_norm[i, :] - s1_over_s0_in_sum)
        ).sum(axis=0).reshape((n_features))

        # First term
        if y[i] > 0:
            phi_i += (
                X_norm[i, :]
                - risk_phi_x[max_distinct_event_times][None, :]
                / risk_phi[max_distinct_event_times][None, None]
            ).reshape((n_features))

        # We recallibrate by w_i only at the very end in here we deviate a
        # bit from Binder ?

        phi_k[i] = phi_i * weights[i]

    # We have computed scaled_variance_matrix globally this delta_beta is
    # (n_k, n_features)
    delta_betas_k = phi_k.dot(scaled_variance_matrix)
    # This Qk is n_features * n_features we will compute Q by block
    Qk = delta_betas_k.T.dot(delta_betas_k)

    return phi_k, delta_betas_k, Qk
