import pickle
from copy import deepcopy
from logging import INFO, WARN
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
import torch
from flwr.common import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.common.logger import log
from flwr.server import Grid, ServerApp
from scipy.linalg import inv

from fedeca_flower.fedeca.disco import CoxPHModelTorch
from fedeca_flower.fedeca.disco_utils import (
    build_global_gradient_and_hessian,
    compute_global_moments,
    compute_global_survival_statistics,
    compute_summary_function,
    get_final_cox_model_function,
)
from fedeca_flower.fedeca.newton_raphson import LogisticRegressionTorch
from fedeca_flower.fedeca.utils import increment_parameters, set_seeds

# Flower ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main entry point for the Flower ServerApp.

    Args:
        grid (Grid): The federated learning grid.
        context (Context): The server context containing configuration and state.
    Returns:
        None
    """

    seed = 42
    set_seeds(seed)
    is_bootstrap = context.run_config["bootstrap"]
    num_seeds = context.run_config["bootstrap-iterations"] if is_bootstrap else 1
    # Compute seeds for ClientApps to use in each bootstrap iteration
    seeds = np.random.randint(
        0, np.iinfo(np.int32).max, size=num_seeds, dtype=np.int32
    ).tolist()
    if is_bootstrap:
        # We need to run N+1 iterations (this is done so to align with the original FedECA implementation)
        # We simply insert a new seed at the 0-th position. #! at iteration 0 there is no data resampling
        # ! therefore it doesn't matter which seed we set.
        seeds.insert(0, seed)

    models = []
    hessians = []
    lls = []
    global_moments_list = []
    for it, seed in enumerate(seeds):

        # 0. Reset local states and set seeds
        reset_local_states(grid, context, seed=seed, iteration=it)

        # 1. FedNewtonRaphson
        propensity_model = fed_newton_raphson(grid, context)

        # 2. FedECA
        for i in range(context.run_config["cox-comp-iterations"]):
            (
                cox_model_state_dict,
                past_ll,
                robust_stats,
                norm_delta,
                success,
                hessian,
            ) = fed_cox_comp_round(grid, context, propensity_model, round_idx=i)
            # log(INFO, f"{norm_delta = } ({success = })")
            if success:
                log(
                    INFO,
                    f"(iter: {it}) Success after {i} FedCox steps ({norm_delta = })",
                )
                break

        if not success:
            # If iteration didn't converge, do not consider it for final COX computation
            log(
                WARN,
                f"Bootstrap with seed ({seed}) did not converge. Discarding model and "
                "statistics from variance estimation.",
            )
            continue

        cox_model = CoxPHModelTorch(ndim=1)
        cox_model.load_state_dict(cox_model_state_dict)
        global_moments = pickle.loads(
            context.state.config_records["stats"]["global-moments-bytes"]
        )

        # Iteration finished, append
        models.append(cox_model)
        hessians.append(hessian)
        lls.append(past_ll)
        global_moments_list.append(global_moments)

    if is_bootstrap:
        log(
            INFO,
            f"Completed {len(seeds)} bootstrapping iterations. Converged {len(models)}/{len(seeds)}",
        )

    # If no single model converged, exit
    if len(models) == 0:
        log(INFO, "Run didn't converge. Exiting.")
        return

    hessians, lls, final_params_list, computed_std_list, global_robust_statistics = (
        get_final_cox_model_function(
            models=models,
            hessians=hessians,
            lls=lls,
            num_seeds=num_seeds,
            global_moments_list=global_moments_list,
            standardize_data=True,
            global_robust_statistics=robust_stats,
        )
    )

    # 3. Computing Summary
    log_likelihood_ = lls[0]

    final_params_array = np.array(final_params_list)

    if not (is_bootstrap):
        variance_matrix = -inv(hessians[0]) / np.outer(
            computed_std_list[0], computed_std_list[0]
        )
        # Run RobustCoxVarianceAlgo-like stage
        Qk_list = run_robust_cox_variance_stage(
            grid,
            propensity_model,
            context.run_config["path-to-data"],
            final_params_list[0],  # beta
            variance_matrix,
            context.state.config_records["stats"]["global-surv-stats-bytes"],
            global_robust_statistics,
        )
        variance_matrix_robust = sum(Qk_list)

    else:
        variance_matrix = np.cov(
            final_params_array[1:], rowvar=False, bias=True
        ).reshape((final_params_array.shape[1], final_params_array.shape[1]))

    # The final params vector is the params wo bootstrap
    if final_params_array.ndim == 2:
        final_params = final_params_array[0]
    else:
        final_params = final_params_array

    # Print summary
    def print_final_summary(var_matrix, title: str):
        summary = compute_summary_function(final_params, var_matrix)

        summary["exp(coef)"] = np.exp(summary["coef"])
        summary["exp(coef) lower 95%"] = np.exp(summary["coef lower 95%"])
        summary["exp(coef) upper 95%"] = np.exp(summary["coef upper 95%"])

        print(f"\n-------------- {title} ---------------")
        print(summary)

    if is_bootstrap:
        print_final_summary(
            variance_matrix, f"Bootstrap FedECA (n_iter = {num_seeds} + 1)"
        )
    else:
        print_final_summary(variance_matrix, f"FedECA (naïve)")
        print_final_summary(variance_matrix_robust, f"FedECA (robust)")


def send_and_receive_all_nodes(
    grid: Grid, record: Union[RecordDict, dict[int, RecordDict]], stage: str
) -> list[Message]:
    """
    Send a message to all available nodes, either the same or a different one per node.

    Args:
        grid (Grid): The federated learning grid.
        record (Union[RecordDict, dict[int, RecordDict]]): Message content for each node.
        stage (str): Stage identifier for the message.
    Returns:
        list[Message]: Replies from all nodes.
    """

    # Sample all nodes
    all_node_ids = grid.get_node_ids()
    # log(INFO, "Sampled %s nodes", len(all_node_ids))
    messages = []
    for node_id in all_node_ids:
        message = Message(
            content=record if isinstance(record, RecordDict) else record[node_id],
            message_type=stage,
            dst_node_id=node_id,
        )
        messages.append(message)

    # Send messages and wait for all results
    replies = list(grid.send_and_receive(messages))
    # log(INFO, "Received %s/%s results", len(replies), len(messages))
    return replies


def generate_global_seeds(grid: Grid, cox_data: str, seed: int):
    """
    Query all ClientApps for their dataset sizes and perform global resampling for bootstrapping.
    Returns a DataFrame mapping which data indices each node should use in the next iteration.

    Args:
        grid (Grid): Federated learning grid.
        cox_data (str): A template-like path to the data that the `ClientApp` would load. For example `"data/center{}/data.csv"`
        seed (int): Random seed for reproducibility.
    Returns:
        pd.DataFrame: Columns ["idx", "node-id"], for bootstrapping assignment.
    """

    # Fetch size of local datasets
    replies = send_and_receive_all_nodes(
        grid,
        RecordDict({"config": ConfigRecord({"cox-data-path": cox_data})}),
        stage="query.get_data_size",
    )

    num_rows_dict = {}  # {node-id: num-rows}
    for msg in replies:
        if msg.has_content():
            count = msg.content["local-data-count"]["count"]
            center = msg.content["local-data-count"]["center"]
            num_rows_dict[msg.metadata.src_node_id] = (count, center)
        else:
            log(WARN, msg.error)

    # Resample with replacement
    # construct dict of 2-colum arrays: [sample-idx, node-id]
    centers_idices = {
        cen: np.column_stack((list(range(num)), [node_id] * num))
        for node_id, (num, cen) in num_rows_dict.items()
    }
    # stack respecting center order
    indices_mapping = np.vstack(
        [centers_idices[cen_id] for cen_id in range(len(centers_idices))]
    )
    im_df = pd.DataFrame(indices_mapping, columns=["idx", "node-id"])
    rng = np.random.default_rng(seed)
    im_df_resampled = im_df.sample(n=im_df.shape[0], replace=True, random_state=rng)

    return im_df_resampled


def reset_local_states(grid: Grid, context: Context, seed: int, iteration: int):
    """
    Reset local states of ClientApps and optionally set up global bootstrapping indices.

    Args:
        grid (Grid): The federated learning grid.
        context (Context): The server context.
        seed (int): Random seed for reproducibility.
        iteration (int): Current bootstrap iteration.
    Returns:
        None
    """

    # Rest local states of ClientApps
    # We don't want to resample the data on the first iteration of bootstrapping.
    is_bootstrap = context.run_config["bootstrap"] if iteration > 0 else False
    record = RecordDict(
        {
            "config": ConfigRecord(
                {"clientapp-seed": int(seed), "is-bootstrap": is_bootstrap}
            )
        }
    )

    if is_bootstrap:
        if context.run_config["bootstrap-fn"] == "global":
            # Get dataframe indicating which local data indices each ClientApp should use
            cox_data = context.run_config["path-to-data"]
            im_df_resampled = generate_global_seeds(grid, cox_data, seed=seed)

            all_node_ids = grid.get_node_ids()
            records = {}
            for node_id in all_node_ids:
                # Indices for this center
                indices_to_sample = im_df_resampled[
                    im_df_resampled["node-id"] == node_id
                ]["idx"].tolist()
                records[node_id] = deepcopy(record)
                records[node_id]["config"]["global-indices-array"] = indices_to_sample

            record = records

    _ = send_and_receive_all_nodes(grid, record, stage="query.reset_state_and_set_seed")

    # Reset local state in ServerApp
    context.state = RecordDict()


def fed_cox_comp_round(
    grid: Grid, context: Context, propensity_model: torch.nn.Module, round_idx: int
):
    """
    Perform a single round of FedECA's Cox model training.

    Args:
        grid (Grid): The federated learning grid.
        context (Context): The server context.
        propensity_model (torch.nn.Module): The propensity model.
        round_idx (int): Current round index.
    Returns:
        Tuple: (cox_model_state_dict, past_ll, robust_stats, norm_delta, success, hessian)
    """

    if round_idx == 0:
        aggr_moments = global_standardization(grid, context, propensity_model)

        global_surv_stats = compute_survival_statistics(
            grid, context, propensity_model, aggr_moments
        )

        risk_phi_stats_list, global_moments_bytes = compute_risk_phi_stats_list(
            grid, context, propensity_model, global_surv_stats
        )

        # Save to context
        context.state.config_records["stats"] = ConfigRecord(
            {
                "global-surv-stats-bytes": pickle.dumps(global_surv_stats),
                "risk-phi-stats-bytes": pickle.dumps(risk_phi_stats_list),
                "global-moments-bytes": global_moments_bytes,
            }
        )

    risk_phi_stats_list = pickle.loads(
        context.state.config_records["stats"]["risk-phi-stats-bytes"]
    )
    global_gradient_and_hessian = build_global_gradient_and_hessian(risk_phi_stats_list)

    (
        risk_phi_stats_list,
        cox_model_state_dict,
        past_ll,
        robust_stats,
        norm_delta,
        success,
    ) = train(
        grid,
        context,
        propensity_model,
        global_gradient_and_hessian,
    )

    # Update phi stats
    context.state.config_records["stats"]["risk-phi-stats-bytes"] = pickle.dumps(
        risk_phi_stats_list
    )
    return (
        cox_model_state_dict,
        past_ll,
        robust_stats,
        norm_delta,
        success,
        global_gradient_and_hessian["hessian"],
    )


def train(
    grid: Grid,
    context: Context,
    p_model: torch.nn.Module,
    glb_gradient_and_hessian: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Train the Cox model on all nodes using provided global gradient and hessian.

    Args:
        grid (Grid): The federated learning grid.
        context (Context): The server context.
        p_model (torch.nn.Module): The propensity model.
        glb_gradient_and_hessian (dict): Global gradient and hessian.
    Returns:
        Tuple: (risk_phi_stats_list, cox_model_state_dict, past_ll, robust_stats, norm_delta, success)
    """

    # Prepare messages
    params_record = ArrayRecord(p_model.state_dict())
    record = RecordDict(
        {
            "prop-model-params": params_record,
            "config": ConfigRecord(
                {
                    "cox-data-path": context.run_config["path-to-data"],
                }
            ),
            "global-survival-stats": ConfigRecord(
                {
                    "bytes": context.state.config_records["stats"][
                        "global-surv-stats-bytes"
                    ]
                }
            ),
            "global-gradient-and-hessian": ConfigRecord(
                {"bytes": pickle.dumps(glb_gradient_and_hessian)}
            ),
        }
    )

    replies = send_and_receive_all_nodes(grid, record, stage="train.train_web_disco")

    # Deserialize received local-phi-stats
    risk_phi_stats_list = [
        pickle.loads(
            msg.content.config_records["train-returns"]["local-phi-stats-bytes"]
        )
        for msg in replies
        if msg.has_content()
    ]

    # Get COX-model state_dict and more
    # ! we just take the first (all are the same)
    train_results = replies[0].content.config_records["train-returns"]
    past_ll = pickle.loads(train_results["past-ll-bytes"])
    robust_stats = pickle.loads(train_results["robust-stats"])
    cox_model_state_dict = pickle.loads(train_results["model-bytes"])
    norm_delta = train_results["norm-delta"]
    success = train_results["success"]

    return (
        risk_phi_stats_list,
        cox_model_state_dict,
        past_ll,
        robust_stats,
        norm_delta,
        success,
    )


def compute_risk_phi_stats_list(
    grid: Grid,
    context: Context,
    p_model: torch.nn.Module,
    glb_surv_stats: dict[str, Any],
) -> Tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Compute local phi statistics for all nodes and global moments.

    Args:
        grid (Grid): The federated learning grid.
        context (Context): The server context.
        p_model (torch.nn.Module): The propensity model.
        glb_surv_stats (dict): Global survival statistics.
    Returns:
        Tuple: (risk_phi_stats_list, global_moments_bytes)
    """

    # Prepare messages
    params_record = ArrayRecord(p_model.state_dict())
    record = RecordDict(
        {
            "prop-model-params": params_record,
            "config": ConfigRecord(
                {
                    "cox-data-path": context.run_config["path-to-data"],
                }
            ),
            "global-survival-stats": ConfigRecord(
                {"bytes": pickle.dumps(glb_surv_stats)}
            ),
        },
    )

    replies = send_and_receive_all_nodes(
        grid, record, stage="query.compute_local_phi_stats"
    )

    # Deserialize received local-phi-stats
    risk_phi_stats_list = [
        pickle.loads(msg.content.config_records["local-phi-stats"]["bytes"])
        for msg in replies
        if msg.has_content()
    ]

    # Return also global moments (taking the first is enough -- same in all clients)
    global_moments_bytes = replies[0].content.config_records["global-moments"]["bytes"]

    return risk_phi_stats_list, global_moments_bytes


def compute_survival_statistics(
    grid: Grid, context: Context, p_model: torch.nn.Module, aggr_moments: Any
):
    """
    Compute global survival statistics by aggregating local statistics from all nodes.

    Args:
        grid (Grid): The federated learning grid.
        context (Context): The server context.
        p_model (torch.nn.Module): The propensity model.
        aggr_moments (Any): Aggregated moments for survival statistics.
    Returns:
        dict: Global survival statistics.
    """

    # Prepare messages
    params_record = ArrayRecord(p_model.state_dict())
    record = RecordDict(
        {
            "prop-model-params": params_record,
            "config": ConfigRecord(
                {
                    "cox-data-path": context.run_config["path-to-data"],
                }
            ),
            "aggregated-moments": ConfigRecord({"bytes": pickle.dumps(aggr_moments)}),
        },
    )

    replies = send_and_receive_all_nodes(
        grid, record, stage="query.compute_local_constant_survival_statistics"
    )

    # Deserialize received local survival statistics
    local_surv_stats = [
        pickle.loads(msg.content.config_records["local-surv-stats"]["bytes"])
        for msg in replies
        if msg.has_content()
    ]

    return compute_global_survival_statistics(local_surv_stats)


def global_standardization(grid: Grid, context: Context, p_model: torch.nn.Module):
    """
    Perform global standardization by aggregating local uncentered moments from all nodes.

    Args:
        grid (Grid): The federated learning grid.
        context (Context): The server context.
        p_model (torch.nn.Module): The propensity model.
    Returns:
        Any: Aggregated global moments.
    """

    # Prepare messages
    params_record = ArrayRecord(p_model.state_dict())
    record = RecordDict(
        {
            "prop-model-params": params_record,
            "config": ConfigRecord(
                {
                    "cox-data-path": context.run_config["path-to-data"],
                }
            ),
        },
    )

    replies = send_and_receive_all_nodes(
        grid, record, stage="query.local_uncentered_moments"
    )

    # Deserialize received local uncentered moments
    local_moments = [
        pickle.loads(msg.content.config_records["uncentered-moments"]["bytes"])
        for msg in replies
        if msg.has_content()
    ]

    # Compute global/aggregated moments
    return compute_global_moments(local_moments)


def fed_newton_raphson(grid: Grid, context: Context) -> torch.nn.Module:
    """
    Implements FedNewtonRaphson (Algorithm 1) for federated logistic regression.

    Args:
        grid (Grid): The federated learning grid.
        context (Context): The server context.
    Returns:
        torch.nn.Module: Trained global propensity model.
    """

    # Init global propensity model
    global_model = LogisticRegressionTorch(ndim=10)
    damping_factor = 0.8  # as in FedECA
    iterations = context.run_config["nr-iterations"]

    for i in range(iterations):
        c_records = netwon_raphson_step(
            global_model, context.run_config["path-to-data"], grid
        )

        n_all_samples = sum([rec["n-samples"] for rec in c_records])

        for idx, c_record in enumerate(c_records):
            # Compute average coefficient of the hessian and of the gradients
            sample_coefficient = c_record["n-samples"] / n_all_samples

            gradients = pickle.loads(c_record["gradients"])
            hessian = pickle.loads(c_record["hessian"])

            if idx == 0:
                total_hessians = hessian * sample_coefficient
                total_gradient_one_d = (
                    np.concatenate([grad.reshape(-1) for grad in gradients])
                    * sample_coefficient
                )
            else:
                total_hessians += hessian * sample_coefficient
                total_gradient_one_d += (
                    np.concatenate([grad.reshape(-1) for grad in gradients])
                    * sample_coefficient
                )

        parameters_update = -damping_factor * np.linalg.solve(
            total_hessians, total_gradient_one_d
        )

        # unflatten array
        updated_parameters = []
        current_index = 0

        for array in gradients:
            num_params = len(array.ravel())
            updated_parameters.append(
                np.array(
                    parameters_update[
                        current_index : current_index + num_params
                    ].reshape(array.shape)
                )
            )
            current_index += num_params

        # apply updated_parameters to global model
        increment_parameters(
            model=global_model,
            updates=[torch.from_numpy(x).to("cpu") for x in updated_parameters],
            with_batch_norm_parameters=True,
        )

    return global_model


def netwon_raphson_step(p_model: torch.nn.Module, data_csv: str, grid: Grid):
    """
    Perform a single Newton-Raphson step for all nodes in the grid.

    Args:
        p_model (torch.nn.Module): The propensity model.
        data_csv (str): Path to client data.
        grid (Grid): The federated learning grid.
    Returns:
        list: List of records returned from all nodes.
    """

    # Prepare messages for NewtonRaphson stage
    params_record = ArrayRecord(p_model.state_dict())
    record = RecordDict(
        {
            "model-params": params_record,
            "config": ConfigRecord({"cox-data-path": data_csv}),
        },
    )
    replies = send_and_receive_all_nodes(grid, record, stage="train.newton_rapshon")

    return [
        rep.content.config_records["nr-returns"] for rep in replies if rep.has_content()
    ]


def run_robust_cox_variance_stage(
    grid: Grid,
    p_model: torch.nn.Module,
    path_to_data: str,
    beta,
    variance_matrix,
    global_survival_stats_bytes,
    global_robust_statistics,
):
    """
    Run the robust Cox variance computation stage for all nodes.

    Args:
        grid (Grid): The federated learning grid.
        p_model (torch.nn.Module): The propensity model.
        path_to_data (str): Path to client data.
        beta: Model coefficients.
        variance_matrix: Variance matrix for robust statistics.
        global_survival_stats_bytes: Global survival statistics (bytes).
        global_robust_statistics: Global robust statistics.
    Returns:
        list: List of robust Cox Qk statistics from all nodes.
    """

    # Prepare messages
    params_record = ArrayRecord(p_model.state_dict())
    record = RecordDict(
        {
            "prop-model-params": params_record,
            "config": ConfigRecord(
                {
                    "cox-data-path": path_to_data,
                    "beta-bytes": pickle.dumps(beta),
                    "variance-matrix-bytes": pickle.dumps(variance_matrix),
                }
            ),
            "global-robust-stats": ConfigRecord(
                {
                    "bytes": pickle.dumps(global_robust_statistics),
                }
            ),
            "global-survival-stats": ConfigRecord(
                {
                    "bytes": global_survival_stats_bytes,
                }
            ),
        }
    )

    replies = send_and_receive_all_nodes(
        grid, record, stage="query.robust_cox_variance"
    )

    # Deserialize and append to list
    Qk_list = []
    for r in replies:
        if r.has_content():
            Qk_list.append(pickle.loads(r.content["Qk"]["bytes"]))
    return Qk_list
