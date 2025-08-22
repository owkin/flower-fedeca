import pickle
from typing import Optional

import numpy as np
import pandas as pd
import torch
from flwr.client import ClientApp
from flwr.common import Array, ConfigRecord, Context, Message, RecordDict

from fedeca_flower.fedeca.disco import CoxPHModelTorch, WebDiscoTrainer
from fedeca_flower.fedeca.newton_raphson import (
    LogisticRegressionTorch,
    NewtonRapshonClientTrainer,
    train_with_newton_raphson,
)
from fedeca_flower.fedeca.robustcox import RobustCoxVarianceAlgo
from fedeca_flower.fedeca.utils import set_seeds


# Flower ClientApp
app = ClientApp()


def load_data(
    context: Context, data_path_template: str, seed: int = 42
) -> pd.DataFrame:
    """
    Load data for the client, with optional bootstrapping.

    If bootstrapping is enabled, resample with replacement (per-client) or use indices provided by the server (global).
    Args:
        context (Context): The node context containing configuration and state.
        data_path_template (str): Path template for the data files. For example: `"data/center{}/data.csv"`
        which will be filled with the appropriate center ID in this function.
        seed (int): Random seed for reproducibility.
    Returns:
        pd.DataFrame: The client data, possibly resampled.
    """

    # df_client = get_client_data(msg, context)
    # Load data and maybe resample with replacement
    partition_id = context.node_config["partition-id"]
    resample_fn = (
        context.run_config["bootstrap-fn"]
        if context.state["config"]["is-bootstrap"]
        else ""
    )

    data_path = data_path_template.format(partition_id)
    df_data = pd.read_csv(data_path)
    if resample_fn == "per-client":
        rng = np.random.default_rng(seed)
        df_client = df_data.sample(n=df_data.shape[0], replace=True, random_state=rng)
    elif resample_fn == "global":
        # use the indices provided by the `ServerApp`
        indices = context.state["config"]["global-indices-array"]
        df_client = df_data.iloc[[int(idx) for idx in indices]].copy()
    else:
        df_client = df_data

    return df_client


@app.query("reset_state_and_set_seed")
def reset(msg: Message, context: Context) -> Message:
    """
    Set seed and reset local context for each iteration of FedECA.

    This is particularly relevant when running with bootstrapping.
    Args:
        msg (Message): Incoming message containing configuration.
        context (Context): The node context to reset.
    Returns:
        Message: Reply message with empty content.
    """
    # Reset local context
    context.state = RecordDict()

    # Set seed and wether it's bootstrapping
    context.state["config"] = msg.content["config"]
    return Message(content=RecordDict(), reply_to=msg)


@app.query("get_data_size")
def get_data(msg: Message, context: Context) -> Message:
    """
    Reply with the number of data samples in the local dataset.

    Args:
        msg (Message): Incoming message.
        context (Context): The node context.
    Returns:
        Message: Reply message containing local data count.
    """

    # Load data for this client
    partition_id = context.node_config["partition-id"]
    data_path_template = msg.content["config"]["data-path"]
    data_path = data_path_template.format(partition_id)
    df_data = pd.read_csv(data_path)
    # Count rows
    num_rows = df_data.shape[0]
    return Message(
        content=RecordDict(
            {
                "local-data-count": ConfigRecord(
                    {"center": partition_id, "count": num_rows}
                )
            }
        ),
        reply_to=msg,
    )


@app.train("newton_rapshon")
def train(msg: Message, context: Context) -> Message:
    """
    Train the local model using Newton-Raphson for FedNewtonRaphson.

    Args:
        msg (Message): Incoming message containing model parameters.
        context (Context): The node context.
    Returns:
        Message: Reply message containing gradients, hessian, and sample count.
    """

    seed = context.state["config"]["clientapp-seed"]
    set_seeds(seed)

    # Load data and maybe resample with replacement
    data_path_template = msg.content["config"]["data-path"]
    df_client = load_data(context, data_path_template, seed)

    # Init local model
    model = LogisticRegressionTorch(ndim=10)
    # Apply params received
    p_record = msg.content.array_records["model-params"]
    model.load_state_dict(p_record.to_torch_state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = NewtonRapshonClientTrainer(model, torch.nn.BCELoss(), device=device)

    gradients, hessian, n_samples = train_with_newton_raphson(df_client, trainer)

    # Put each element as an Array inside a RecordDict
    configs_record = ConfigRecord(
        {
            "gradients": pickle.dumps(gradients),
            "hessian": pickle.dumps(hessian),
            "n-samples": n_samples,
        }
    )

    content = RecordDict({"nr-returns": configs_record})
    return Message(content=content, reply_to=msg)


def setup_data_and_disco_trainer(msg: Message, context: Context):
    """
    Utility function to set up client data and initialize the WebDiscoTrainer and models.

    Args:
        msg (Message): Incoming message containing model parameters.
        context (Context): The node context.
    Returns:
        Tuple: (df_client, disco_trainer, cox_model, prop_model)
    """

    seed = context.state["config"]["clientapp-seed"]
    set_seeds(seed)

    # Load data and maybe resample with replacement
    data_path_template = msg.content["config"]["data-path"]
    df_client = load_data(context, data_path_template, seed)

    cox_model = CoxPHModelTorch(ndim=1)

    # Build Propensity model
    prop_model = LogisticRegressionTorch(ndim=10)
    # Apply params received
    p_record = msg.content.array_records["prop-model-params"]
    prop_model.load_state_dict(p_record.to_torch_state_dict())

    disco_trainer = WebDiscoTrainer(
        model=cox_model,
        propensity_model=prop_model,
        duration_col="time",
        event_col="event",
        treated_col="treatment",
        standardize_data=True,
        robust=True,  # The ClientApp computes additional statistics for FedECA robust always
        # Then, they will get taken into account in the final COX model accordingly
    )
    return df_client, disco_trainer, cox_model, prop_model


@app.query("local_uncentered_moments")
def query(msg: Message, context: Context) -> Message:
    """
    Compute and return local uncentered moments for global standardization in WebDisco.

    Args:
        msg (Message): Incoming message.
        context (Context): The node context.
    Returns:
        Message: Reply message containing uncentered moments.
    """

    df_client, disco_trainer, _, _ = setup_data_and_disco_trainer(msg, context)
    response_content = RecordDict()
    uncentered_moments = disco_trainer.local_uncentered_moments(df_client)
    response_content.config_records["uncentered-moments"] = ConfigRecord(
        {"bytes": pickle.dumps(uncentered_moments)}
    )

    return Message(content=response_content, reply_to=msg)


@app.query("compute_local_constant_survival_statistics")
def query(msg: Message, context: Context) -> Message:
    """
    Compute local constant survival statistics for WebDisco.

    Args:
        msg (Message): Incoming message containing aggregated moments.
        context (Context): The node context.
    Returns:
        Message: Reply message containing local survival statistics.
    """
    df_client, disco_trainer, _, _ = setup_data_and_disco_trainer(msg, context)
    aggregated_modments = pickle.loads(
        msg.content.config_records["aggregated-moments"]["bytes"]
    )
    local_survival_statistics = (
        disco_trainer._compute_local_constant_survival_statistics(
            df_client, aggregated_modments
        )
    )
    response_content = RecordDict(
        {
            "local-surv-stats": ConfigRecord(
                {"bytes": pickle.dumps(local_survival_statistics)}
            )
        }
    )

    return Message(content=response_content, reply_to=msg)


@app.query("compute_local_phi_stats")
def query(msg: Message, context: Context) -> Message:
    """
    Compute local phi statistics for WebDisco.

    Args:
        msg (Message): Incoming message containing global survival statistics.
        context (Context): The node context.
    Returns:
        Message: Reply message containing local phi stats and global moments.
    """
    df_client, disco_trainer, _, _ = setup_data_and_disco_trainer(msg, context)
    global_survival_statistics = pickle.loads(
        msg.content.config_records["global-survival-stats"]["bytes"]
    )
    local_phi_stats = disco_trainer.compute_local_phi_stats(
        df_client, global_survival_statistics
    )

    # We need to preserve the global_moments in the context for the "train" stage
    global_moments_bytes = pickle.dumps(disco_trainer.global_moments)
    context.state.config_records["local-config"] = ConfigRecord(
        {"global-moments": global_moments_bytes}
    )

    response_content = RecordDict(
        {
            "local-phi-stats": ConfigRecord({"bytes": pickle.dumps(local_phi_stats)}),
            "global-moments": ConfigRecord({"bytes": global_moments_bytes}),
        }
    )

    return Message(content=response_content, reply_to=msg)


@app.train("train_web_disco")
def train(msg: Message, context: Context) -> Message:
    """
    Run the train() method in WebDisco for federated training.

    Args:
        msg (Message): Incoming message containing global statistics and model state.
        context (Context): The node context.
    Returns:
        Message: Reply message containing training results and updated model state.
    """

    df_client, disco_trainer, cox_model, _ = setup_data_and_disco_trainer(msg, context)
    global_survival_statistics = pickle.loads(
        msg.content.config_records["global-survival-stats"]["bytes"]
    )
    global_gradient_and_hessian = pickle.loads(
        msg.content.config_records["global-gradient-and-hessian"]["bytes"]
    )

    shared_state = {**global_gradient_and_hessian, **global_survival_statistics}

    # Load state of previously update Cox-model weights
    if cox_model_state_dict_bytes := context.state.config_records["local-config"].get(
        "cox-model-bytes"
    ):
        cox_model.load_state_dict(pickle.loads(cox_model_state_dict_bytes))

    # Restore from local context
    local_config = context.state.config_records["local-config"]
    if "count-iter" in local_config:
        count_iter = local_config["count-iter"]
        current_weights = pickle.loads(local_config["current-weights-bytes"])
    else:
        # must be first time we do train, therefore init count-iter to 1 and store in state
        count_iter = 1
        current_weights = np.zeros(global_gradient_and_hessian["gradient_shape"])

    disco_trainer.init_train_step(
        current_weights=current_weights, count_inter=count_iter
    )
    disco_trainer.global_moments = pickle.loads(local_config["global-moments"])

    local_phi_stats, converging, success, norm_delta, past_ll, robust_stats = (
        disco_trainer.train(df_client, shared_state)
    )
    model_bytes = pickle.dumps(disco_trainer.model.state_dict())
    response_content = RecordDict(
        {
            "train-returns": ConfigRecord(
                {
                    "local-phi-stats-bytes": pickle.dumps(local_phi_stats),
                    "past-ll-bytes": pickle.dumps(past_ll),
                    "robust-stats": pickle.dumps(robust_stats),
                    "model-bytes": model_bytes,
                    "norm-delta": norm_delta,
                    "converging": converging,
                    "success": success,
                }
            )
        }
    )

    # Update local context
    context.state.config_records["local-config"][
        "count-iter"
    ] = disco_trainer.count_iter
    context.state.config_records["local-config"]["current-weights-bytes"] = (
        pickle.dumps(disco_trainer.current_weights)
    )
    context.state.config_records["local-config"]["cox-model-bytes"] = model_bytes

    return Message(content=response_content, reply_to=msg)


@app.query("robust_cox_variance")
def query(msg: Message, context: Context) -> Message:
    """
    Compute robust COX variance statistics for the client.

    Args:
        msg (Message): Incoming message containing beta, variance matrix, and global statistics.
        context (Context): The node context.
    Returns:
        Message: Reply message containing robust COX Qk statistics.
    """
    df_client, _, _, prop_model = setup_data_and_disco_trainer(msg, context)

    beta = pickle.loads(msg.content["config"]["beta-bytes"])
    variance_matrix = pickle.loads(msg.content["config"]["variance-matrix-bytes"])
    global_robust_statistics = pickle.loads(
        msg.content.config_records["global-robust-stats"]["bytes"]
    )
    global_survival_statistics = pickle.loads(
        msg.content.config_records["global-survival-stats"]["bytes"]
    )
    robust_cox = RobustCoxVarianceAlgo(
        beta=beta,
        variance_matrix=variance_matrix,
        global_robust_statistics=global_robust_statistics,
        propensity_model=prop_model,
        duration_col="time",
        event_col="event",
        treated_col="treatment",
    )

    Qk = robust_cox.local_q_computation(
        data_from_opener=df_client,
        shared_state=global_survival_statistics["moments"],
    )
    response_content = RecordDict({"Qk": ConfigRecord({"bytes": pickle.dumps(Qk)})})

    return Message(content=response_content, reply_to=msg)
