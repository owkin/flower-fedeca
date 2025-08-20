from typing import Generator, Union

import numpy as np
import torch
from pandas.api.types import is_numeric_dtype


def set_seeds(seed: int = 42):
    """Set seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    # TODO: set cuda seed and deterministic algos


class SubstraflTorchDataset(torch.utils.data.Dataset):
    """Substra torch dataset class."""

    def __init__(
        self,
        data_from_opener,
        is_inference: bool,
        target_columns: Union[list, None] = None,
        columns_to_drop: Union[list, None] = None,
        fit_cols: Union[list, None] = None,
        dtype="float64",
        return_torch_tensors=False,
    ):
        """Initialize SubstraflTorchDataset class."""
        self.data = data_from_opener
        self.is_inference = is_inference
        self.target_columns = (
            target_columns if target_columns is not None else ["T", "E"]
        )
        columns_to_drop = columns_to_drop if columns_to_drop is not None else []
        self.columns_to_drop = list(set(columns_to_drop + self.target_columns))

        string_columns = [
            col for col in self.data.columns if not (is_numeric_dtype(self.data[col]))
        ]

        if fit_cols is not None:
            self.data = self.data[fit_cols + self.target_columns]

        self.x = (
            self.data.drop(
                columns=(
                    [
                        col
                        for col in (self.columns_to_drop + string_columns)
                        if col in self.data
                    ]
                )
            )
            .to_numpy()
            .astype(dtype)
        )

        self.y = self.data[self.target_columns].to_numpy().astype(dtype)
        self.return_torch_tensors = return_torch_tensors

    def __getitem__(self, idx):
        """Get item."""
        if self.is_inference:
            x = self.x[idx]
            if self.return_torch_tensors:
                x = torch.from_numpy(x)
            return x

        else:
            x, y = self.x[idx], self.y[idx]
            if self.return_torch_tensors:
                x, y = torch.from_numpy(x), torch.from_numpy(y)
            return x, y

    def __len__(self):
        """Get length."""
        return len(self.data.index)


def model_parameters(
    model: torch.nn.Module, with_batch_norm_parameters: bool
) -> torch.nn.parameter.Parameter:
    """A generator of the given model parameters. The returned generator yields references hence all modification done
    to the yielded object will be applied to the input model. If with_batch_norm_parameters is set to True, the running
    mean and the running variance of each batch norm layer will be added after the "classic" parameters.
    """

    def my_iterator():
        for p in model.parameters():
            yield p

        if with_batch_norm_parameters:
            for p in batch_norm_param(model):
                yield p

    return my_iterator


def increment_parameters(
    model: torch.nn.Module,
    updates: list[torch.nn.parameter.Parameter],
    *,
    with_batch_norm_parameters: bool,
    updates_multiplier: float = 1.0,
):
    """Add the given update to the model parameters. If with_batch_norm_parameters is set to True, the operation
    will include the running mean and the running variance of the batch norm layers (in this case, they must be
    included in the given update). This function modifies the given model internally and therefore returns nothing.
    """
    with torch.no_grad():
        # INFO: this is the faster way I found of checking that both model.parameters() and shared states has the
        # same length as model.parameters() is a generator.
        iter_params = model_parameters(
            model=model, with_batch_norm_parameters=with_batch_norm_parameters
        )
        n_parameters = len(list(iter_params()))
        assert n_parameters == len(
            updates
        ), "Length of model parameters and updates are unequal."

        for weights, update in zip(iter_params(), updates):
            assert update.data.shape == weights.data.shape, (
                f"The shape of the model weights ({weights.data.shape}) and of the update ({update.data.shape}) "
                "passed in the updates argument are unequal."
            )
            weights.data += updates_multiplier * update.data


def is_batchnorm_layer(layer: torch.nn.Module) -> bool:
    """Checks if the provided layer is a Batch Norm layer (either 1D, 2D or 3D)."""
    list_bn_layers = [
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.LazyBatchNorm1d,
        torch.nn.LazyBatchNorm2d,
        torch.nn.LazyBatchNorm3d,
    ]
    for bn_layer_class in list_bn_layers:
        if isinstance(layer, bn_layer_class):
            return True

    return False


def batch_norm_param(
    model: torch.nn.Module,
) -> Generator[torch.nn.parameter.Parameter, None, None]:
    """Generator of the internal parameters of the batch norm layers
    of the model. This yields references hence all modification done to the yielded object will
    be applied to the input model."""
    for _, module in model.named_modules():
        if is_batchnorm_layer(module):
            yield module.running_mean
            yield module.running_var
