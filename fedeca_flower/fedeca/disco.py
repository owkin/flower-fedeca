from copy import deepcopy
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from autograd import elementwise_grad
from autograd import numpy as anp
from lifelines.utils import StepSizer
from scipy.linalg import norm
from scipy.linalg import solve as spsolve

from .disco_utils import compute_uncentered_moment
from .survival_utils import (
    build_X_y_function,
    compute_X_y_and_propensity_weights_function,
)
from .utils import increment_parameters


class CoxPHModelTorch(torch.nn.Module):
    """Cox Proportional Hazards Model implemented using PyTorch."""

    def __init__(self, ndim=10, torch_dtype=torch.float64):
        """Initialize the CoxPHModelTorch.

        Parameters
        ----------
        ndim : int, optional
            Number of input dimensions or features, by default 10.
        torch_dtype : torch.dtype, optional
            Data type for PyTorch tensors, by default torch.float64.
        """
        super().__init__()
        self.ndim = ndim
        self.torch_dtype = torch_dtype
        self.fc1 = torch.nn.Linear(self.ndim, 1, bias=False).to(self.torch_dtype)
        self.fc1.weight.data.fill_(0.0)

    def forward(self, x):
        """Perform a forward pass through the CoxPH model."""
        return torch.exp(self.fc1(x))  # pylint: disable=not-callable


class WebDiscoTrainer:

    def __init__(
        self,
        model: torch.nn.Module,
        propensity_model: torch.nn.Module,
        duration_col: str,
        event_col: str,
        treated_col: str,
        initial_step_size: float = 0.95,
        penalizer: float = 0.0,
        l1_ratio: float = 0.0,
        learning_rate_strategy: str = "lifelines",
        training_strategy: str = "iptw",
        standardize_data: bool = True,
        tol: float = 1e-16,
        store_hessian: bool = False,
        cox_fit_cols: Union[None, list] = None,
        propensity_fit_cols: Union[None, list] = None,
        with_batch_norm_parameters: bool = False,
        robust: bool = False,
    ) -> None:
        self.model = model
        self.propensity_model = propensity_model
        self.propensity_model.eval()
        self._duration_col = duration_col
        self._event_col = event_col
        self._target_cols = [self._duration_col, self._event_col]
        self._treated_col = treated_col if treated_col is not None else []
        self._standardize_data = standardize_data
        self._tol = tol
        self.step_sizer = StepSizer(initial_step_size)
        self._penalizer = penalizer
        if penalizer:
            raise NotImplementedError(f"penalizer > 0 not supported yet")
        self._l1_ratio = l1_ratio
        self._cox_fit_cols = cox_fit_cols if cox_fit_cols is not None else []
        self._training_strategy = training_strategy
        self._propensity_fit_cols = propensity_fit_cols

        self.global_moments = None
        self.count_iter = None
        self.current_weights = None
        self.learning_rate_strategy = learning_rate_strategy
        self._with_batch_norm_parameters = with_batch_norm_parameters
        self._store_hessian = store_hessian
        self._robust = robust

        assert learning_rate_strategy in [
            "lifelines",
            # "constant",
        ], "Learning rate strategy not supported"

    def local_uncentered_moments(self, data_from_opener: pd.DataFrame):
        """Compute the local uncentered moments."""
        # We do not have to do the mean on the target columns
        data_from_opener = data_from_opener.drop(columns=self._target_cols)
        if self.propensity_model is not None:
            assert self._treated_col is not None
            if self._training_strategy == "iptw":
                data_from_opener = data_from_opener.loc[:, [self._treated_col]]
            elif self._training_strategy == "aiptw":
                if len(self._cox_fit_cols) > 0:
                    data_from_opener = data_from_opener.loc[
                        :, [self._treated_col] + self._cox_fit_cols
                    ]
                else:
                    pass
        else:
            assert self._training_strategy == "webdisco"
            if len(self._cox_fit_cols) > 0:
                data_from_opener = data_from_opener.loc[:, self._cox_fit_cols]
            else:
                pass

        results = {
            f"moment{k}": compute_uncentered_moment(data_from_opener, k)
            for k in range(1, 3)
        }
        results["n_samples"] = data_from_opener.select_dtypes(include=np.number).count()
        return results

    def compute_local_phi_stats(
        self,
        data_from_opener: pd.DataFrame,
        # Set shared_state to None per default for clarity reason as
        # the decorator will do it if the arg shared_state is not passed.
        shared_state: Optional[dict[str, Any]] = None,
        ignore_moments: bool = False,  #! New (since when called from train() we don't seem to want momemnt)
    ):
        """Compute local updates."""
        global_survival_statistics = shared_state["global_survival_statistics"]
        moments = None if ignore_moments else shared_state["moments"]

        X, y, weights = self.compute_X_y_and_propensity_weights(
            data_from_opener, moments
        )

        distinct_event_times = global_survival_statistics["distinct_event_times"]

        self.model.eval()
        # The shape of expbetaTx is (N, 1)
        X = torch.from_numpy(X)
        expbetaTx = self.model(X).detach().numpy()
        X = X.numpy()
        betaTx = np.log(expbetaTx)  # very inefficient, but whatever
        offset = betaTx.max(axis=0)
        factor = np.exp(offset)
        expbetaTx_stable = np.exp(betaTx - offset)
        # for risk_phi each element is a scalar
        risk_phi = []
        # for risk_phi_x each element is of the dimension of a feature N,
        risk_phi_x = []
        # for risk_phi_x_x each element is of the dimension of a feature squared N, N
        risk_phi_x_x = []
        for _, t in enumerate(distinct_event_times):
            Rt = np.where(np.abs(y) >= t)[0]
            weights_for_rt = weights[Rt]
            risk_phi.append(
                factor
                * (np.multiply(expbetaTx_stable[Rt], weights_for_rt).sum(axis=(0, 1)))
            )
            common_block = np.multiply(expbetaTx_stable[Rt] * weights_for_rt, X[Rt])
            risk_phi_x.append(factor * common_block.sum(axis=0))
            risk_phi_x_x.append(factor * np.einsum("ij,ik->jk", common_block, X[Rt]))
        local_phi_stats = {}
        local_phi_stats["risk_phi"] = risk_phi
        local_phi_stats["risk_phi_x"] = risk_phi_x
        local_phi_stats["risk_phi_x_x"] = risk_phi_x_x

        return {
            "local_phi_stats": local_phi_stats,
            # The server being stateless we need to feed it perpetually
            "global_survival_statistics": global_survival_statistics,
        }

    def init_train_step(self, current_weights: np.ndarray, count_inter: int):

        self.count_iter = count_inter
        self.current_weights = current_weights

    def train(
        self,
        data_from_opener: pd.DataFrame,
        # Set shared_state to None per default for clarity reason as
        # the decorator will do it if the arg shared_state is not passed.
        shared_state: Optional[dict[str, Any]] = None,
    ) -> Tuple[dict[str, Any], bool, bool, float, Any, dict[str, Any]]:
        """Local train function."""

        # We either simply update the model with NR update or we compute risk_phi_stats
        gradient = shared_state["gradient"]
        hessian = shared_state["hessian"]
        second_part_ll = shared_state["second_part_ll"]
        global_survival_statistics = shared_state["global_survival_statistics"]
        first_part_ll = deepcopy(
            global_survival_statistics["global_sum_features_on_events"]
        )

        n = global_survival_statistics["total_number_samples"]

        if self._penalizer > 0.0:
            # TODO
            if self.learning_rate_strategy == "lifelines":
                # This is used to multiply the penalty
                # We use a smooth approximation for the L1 norm (for more details
                # see docstring of function)
                # we use numpy autograd to be able to compute the first and second
                # order derivatives of this expression

                def soft_abs(x, a):
                    return 1 / a * (anp.logaddexp(0, -a * x) + anp.logaddexp(0, a * x))

                def elastic_net_penalty(beta, a):
                    l1 = self._l1_ratio * soft_abs(beta, a)
                    l2 = 0.5 * (1 - self._l1_ratio) * (beta**2)
                    reg = n * (self._penalizer * (l1 + l2)).sum()
                    return reg

                # Regularization affects both the gradient and the hessian
                # producing a better conditioned hessian.
                d_elastic_net_penalty = elementwise_grad(elastic_net_penalty)
                dd_elastic_net_penalty = elementwise_grad(d_elastic_net_penalty)
                # lifelines trick to progressively sharpen the approximation of
                # the l1 regularization.
                alpha = 1.3**self.count_iter
                # We are trying to **maximize** the log-likelihood that is why
                # we put a negative sign and not a plus sign on the regularization.
                # The fact that we are actually moving towards the maximum and
                # not towards the minimum is because -H is psd.
                gradient -= d_elastic_net_penalty(self.current_weights, alpha)
                hessian[
                    np.diag_indices(shared_state["gradient_shape"])
                ] -= dd_elastic_net_penalty(self.current_weights, alpha)
            else:
                raise NotImplementedError

        # second half line 6 algo 4
        inv_h_dot_g_T = spsolve(-hessian, gradient, assume_a="pos", check_finite=False)

        norm_delta = norm(inv_h_dot_g_T)

        step_size = self.step_sizer.update(norm_delta).next()
        self.count_iter += 1
        updates = step_size * inv_h_dot_g_T

        # We keep the current version of the weights, because of ll computations
        past_ll = (self.current_weights * first_part_ll).sum(axis=0) + second_part_ll
        self.current_weights += updates

        increment_parameters(
            model=self.model,
            updates=[torch.from_numpy(updates[None, :])],
            with_batch_norm_parameters=self._with_batch_norm_parameters,
        )

        # convergence criteria
        if norm_delta < 1e-07:
            converging, success = False, True
        elif step_size <= 0.00001:
            converging, success = False, False
        else:
            converging, success = True, False

        robust_stats = {}
        if self._robust:
            robust_stats["global_risk_phi_list"] = shared_state["global_risk_phi_list"]
            robust_stats["global_risk_phi_x_list"] = shared_state[
                "global_risk_phi_x_list"
            ]
            # TODO this renaming and moving around is useless and inefficient
            robust_stats["global_weights_counts_on_events"] = shared_state[
                "global_survival_statistics"
            ]["weights_counts_on_events"]
            robust_stats["distinct_event_times"] = shared_state[
                "global_survival_statistics"
            ]["distinct_event_times"]

        # !: I've removed the _skip argument, i pass the shared state, and i've introduced a flag to ingnore using moments from the shared_state
        return (
            self.compute_local_phi_stats(
                data_from_opener=data_from_opener,
                shared_state=shared_state,
                ignore_moments=True,
            ),
            converging,
            success,
            float(norm_delta),
            past_ll,
            robust_stats,
        )

    def _compute_local_constant_survival_statistics(
        self, data_from_opener, shared_state
    ):
        """Computes local statistics and Dt for all ts in the distinct event times."""
        X, y, weights = self.compute_X_y_and_propensity_weights(
            data_from_opener, shared_state
        )
        distinct_event_times = np.unique(y[y > 0]).tolist()

        sum_features_on_events = np.zeros(X.shape[1:])
        number_events_by_time = []
        weights_counts_on_events = []
        for t in distinct_event_times:
            Dt = np.where(y == t)[0]
            num_events = len(Dt)
            sum_features_on_events += (weights[Dt] * X[Dt, :]).sum(axis=0)
            number_events_by_time.append(num_events)
            weights_counts_on_events.append(weights[Dt].sum())

        return {
            "sum_features_on_events": sum_features_on_events,
            "distinct_event_times": distinct_event_times,
            "number_events_by_time": number_events_by_time,
            "total_number_samples": X.shape[0],
            "moments": shared_state,
            "weights_counts_on_events": weights_counts_on_events,
        }

    def compute_X_y_and_propensity_weights(self, data_from_opener, shared_state):
        """Build appropriate X, y and weights from raw output of opener."""
        X, y, treated, Xprop, self.global_moments = build_X_y_function(
            data_from_opener,
            self._event_col,
            self._duration_col,
            self._treated_col,
            self._target_cols,
            self._standardize_data,
            self.propensity_model,
            self._cox_fit_cols,
            self._propensity_fit_cols,
            self._tol,
            self._training_strategy,
            shared_state=shared_state,
            global_moments=(
                {} if not hasattr(self, "global_moments") else self.global_moments
            ),
        )
        X, y, weights = compute_X_y_and_propensity_weights_function(
            X, y, treated, Xprop, self.propensity_model, self._tol
        )
        return X, y, weights
