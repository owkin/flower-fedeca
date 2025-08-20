"""Estimate the variance for mispecified Cox models."""

import copy
from math import sqrt

import numpy as np
import pandas as pd
import torch

from .disco import CoxPHModelTorch, WebDiscoTrainer
from .survival_utils import compute_q_k


class RobustCoxVarianceAlgo(WebDiscoTrainer):
    """Implement local client method for robust cox variance estimation."""

    def __init__(
        self,
        beta: np.ndarray,
        variance_matrix: np.ndarray,
        global_robust_statistics: dict[list[np.ndarray]],
        propensity_model: torch.nn.Module,
        duration_col: str,
        event_col: str,
        treated_col: str,
        standardize_data: bool = True,
        training_strategy: str = "iptw",
        dtype: float = "float64",
        tol: float = 1e-16,
    ):
        """Initialize Robust Cox Variance Algo."""
        self.beta = beta
        self.duration_col = duration_col
        self.treated_col = treated_col
        self.event_col = event_col
        self.standardize_data = standardize_data
        self.variance_matrix = variance_matrix
        self._tol = tol

        assert isinstance(global_robust_statistics, dict)
        global_robust_statistics_arg = copy.deepcopy(global_robust_statistics)

        assert all(
            [
                attr in global_robust_statistics_arg
                for attr in [
                    "global_weights_counts_on_events",
                    "global_risk_phi_list",
                    "global_risk_phi_x_list",
                    "distinct_event_times",
                    "global_moments",
                ]
            ]
        )

        global_moments = global_robust_statistics_arg.pop("global_moments")

        assert all(
            [
                len(global_robust_statistics_arg["global_weights_counts_on_events"])
                == len(v)
                for k, v in global_robust_statistics_arg.items()
                if k != "global_weights_counts_on_events"
            ]
        )

        self.global_robust_statistics = global_robust_statistics_arg
        if self.standardize_data:
            computed_stds = (
                global_moments["vars"]
                .transform(
                    lambda x: sqrt(x * global_moments["bias_correction"] + self._tol)
                )
                .to_numpy()
            )
        else:
            computed_stds = np.ones((self.variance_matrix.shape[0])).squeeze()

        # We initialize the Cox model to the final parameters from WebDisco
        # that we need to unnormalize
        fc1_weight = torch.from_numpy(beta * computed_stds)
        # We need to scale the variance matrix
        self.scaled_variance_matrix = (
            self.variance_matrix
            * np.tile(computed_stds, (self.variance_matrix.shape[0], 1)).T
        )

        class InitializedCoxPHModelTorch(CoxPHModelTorch):
            def __init__(self):
                super().__init__(ndim=1)
                self.fc1.weight.data = fc1_weight

        init_cox = InitializedCoxPHModelTorch()

        super().__init__(
            model=init_cox,
            # batch_size=sys.maxsize,
            # dataset=survival_dataset_class,
            propensity_model=propensity_model,
            duration_col=duration_col,
            event_col=event_col,
            treated_col=treated_col,
            standardize_data=standardize_data,
            training_strategy=training_strategy,
            tol=tol,
        )
        # Now AND ONLY NOW we give it the global mean and weights computed by WebDisco
        # otherwise self.global_moments is set to None by
        # WebDisco init
        # TODO WebDisco init accept global_moments
        self.global_moments = global_moments

    def local_q_computation(self, data_from_opener: pd.DataFrame, shared_state=None):
        """Local Q computation."""
        df = data_from_opener

        distinct_event_times = self.global_robust_statistics["distinct_event_times"]
        weights_counts_on_events = self.global_robust_statistics[
            "global_weights_counts_on_events"
        ]
        risk_phi = self.global_robust_statistics["global_risk_phi_list"]
        risk_phi_x = self.global_robust_statistics["global_risk_phi_x_list"]

        (
            X_norm,
            y,
            weights,
        ) = self.compute_X_y_and_propensity_weights(df, shared_state=shared_state)

        self.model.eval()
        # The shape of expbetaTx is (N, 1)
        X_norm = torch.from_numpy(X_norm)
        score = self.model(X_norm).detach().numpy()
        X_norm = X_norm.numpy()

        phi_k, delta_betas_k, Qk = compute_q_k(
            X_norm,
            y,
            self.scaled_variance_matrix,
            distinct_event_times,
            weights_counts_on_events,
            risk_phi,
            risk_phi_x,
            score,
            weights,
        )

        # The attributes below are private to the client
        self._client_statistics = {}
        self._client_statistics["phi_k"] = phi_k
        self._client_statistics["delta_betas_k"] = delta_betas_k
        self._client_statistics["Qk"] = Qk

        return Qk
