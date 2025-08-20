from typing import Optional

import numpy as np
import pandas as pd
import torch

from .utils import SubstraflTorchDataset


class LogisticRegressionTorch(torch.nn.Module):
    """Pytorch logistic regression class."""

    def __init__(self, ndim, torch_dtype=torch.float64):
        """Initialize Logistic Regression model in PyTorch."""
        super(LogisticRegressionTorch, self).__init__()
        self.torch_dtype = torch_dtype
        self.ndim = ndim
        self.fc1 = torch.nn.Linear(self.ndim, 1).to(self.torch_dtype)
        # Zero-init as in sklearn
        self.fc1.weight.data.fill_(0.0)
        self.fc1.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.fc1(x)
        return torch.sigmoid(x)


class NewtonRapshonClientTrainer:

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        l2_coeff: float = 0,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None,
    ):

        self.model = model
        self.criterion = criterion
        self.l2_coeff = l2_coeff
        self.seed = seed
        self._device = device

        # initialized and used only in the train method
        self._final_gradients = None
        self._final_hessian = None
        self._n_samples_done = None

    def _initialize_gradients_and_hessian(self):
        """Initializes the gradients and hessian matrices."""
        number_of_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        n_samples_done = 0

        final_gradients = [torch.zeros_like(p).numpy() for p in self.model.parameters()]
        final_hessian = np.zeros(
            [number_of_trainable_params, number_of_trainable_params]
        )

        return final_gradients, final_hessian, n_samples_done

    def _update_gradients_and_hessian(
        self, loss: torch.Tensor, current_batch_size: int
    ):
        """Updates the gradients and hessian matrices."""

        gradients, hessian = self._compute_gradients_and_hessian(loss)

        self._n_samples_done += current_batch_size

        self._final_hessian += hessian.cpu().detach().numpy()
        self._final_gradients = [
            sum(final_grad, grad.cpu().detach().numpy())
            for final_grad, grad in zip(self._final_gradients, gradients)
        ]

    def _jacobian(
        self, tensor_y: torch.Tensor, create_graph: bool = False
    ) -> torch.Tensor:
        """Compute the Jacobian for each  the given tensor_y regarding the
        model parameters."""
        jacobian = []
        flat_y = torch.cat([t.reshape(-1) for t in tensor_y])
        for y in flat_y:
            for param in self.model.parameters():
                if param.requires_grad:
                    (gradient,) = torch.autograd.grad(
                        y, param, retain_graph=True, create_graph=create_graph
                    )
                    jacobian.append(gradient)

        return jacobian

    def _hessian_shape(
        self, second_order_derivative: list[torch.Tensor]
    ) -> torch.Tensor:
        """Reshape from the second order derivative to obtain the Hessian
        matrix."""
        hessian = torch.cat([t.reshape(-1) for t in second_order_derivative])
        return hessian.reshape(self._final_hessian.shape)

    def _compute_gradients_and_hessian(self, loss: torch.Tensor) -> tuple[torch.Tensor]:
        """The compute_gradients_and_hessian function compute the gradients and
        the Hessian matrix of the parameters regarding the given loss, and
        outputs them."""

        gradients = self._jacobian(loss[None], create_graph=True)
        second_order_derivative = self._jacobian(gradients)

        hessian = self._hessian_shape(second_order_derivative)

        hessian = 0.5 * hessian + 0.5 * hessian.T  # ensure the hessian is symmetric

        return gradients, hessian

    def _l2_reg(self) -> torch.Tensor:
        """Compute the l2 regularization regarding the model parameters."""
        # L2 regularization
        l2_reg = 0
        for param in self.model.parameters():
            l2_reg += self.l2_coeff * torch.sum(param**2) / 2
        return l2_reg

    def local_train(
        self,
        train_dataset: torch.utils.data.Dataset,
    ):
        """Local train method. Contains the local training loop."""
        self._final_gradients, self._final_hessian, self._n_samples_done = (
            self._initialize_gradients_and_hessian()
        )

        # As the parameters of the model don't change during the loop, the l2 regularization is constant and can be
        # calculated only once for all the batches.
        l2_reg = self._l2_reg()

        # Create torch dataloader
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=len(train_dataset)
        )
        assert len(train_data_loader) == 1
        for x_batch, y_batch in train_data_loader:
            x_batch = x_batch.to(self._device)
            y_batch = y_batch.to(self._device)

            # Forward pass
            y_pred = self.model(x_batch)

            # Compute Loss
            loss = self.criterion(y_pred, y_batch)

            # L2 regularization
            loss += l2_reg

            current_batch_size = len(x_batch)

            self._update_gradients_and_hessian(loss, current_batch_size)

        eigenvalues = np.linalg.eig(self._final_hessian)[0].real
        if not (eigenvalues >= 0).all():
            raise ValueError(
                "Hessian matrix is not positive semi-definite, either the problem is not convex or due to numerical"
                " instability. It is advised to try to increase the l2_coeff. "
                f"Calculated eigenvalues are {eigenvalues.tolist()} and considered l2_coeff is {self.l2_coeff}"
            )


def train_with_newton_raphson(
    dataframe: pd.DataFrame,
    trainer: NewtonRapshonClientTrainer,
):
    """Newton-Raphson training."""
    train_dataset = SubstraflTorchDataset(
        dataframe,
        is_inference=False,
        target_columns=["treatment"],
        columns_to_drop=["event", "time"],
        return_torch_tensors=True,
    )

    trainer.model.train()
    trainer.local_train(train_dataset)

    return trainer._final_gradients, trainer._final_hessian, trainer._n_samples_done
