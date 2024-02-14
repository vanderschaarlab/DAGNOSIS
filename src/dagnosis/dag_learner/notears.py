# from https://github.com/jeroenbe/d-struct

# stdlib
from typing import Any, Tuple

# third party
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

# dagnosis absolute
import dagnosis.dag_learner.utils as ut
from dagnosis.dag_learner.dsl import NotearsMLP, NotearsSobolev


class NOTEARS(nn.Module):
    def __init__(
        self,
        dim: int,  # Dims of system
        nonlinear_dims: list = [10, 10, 1],  # Dims for non-linear arch
        sem_type: str = "mlp",
        rho: float = 1.0,  # NOTEARS parameters
        alpha: float = 1.0,  # |
        lambda1: float = 0.0,  # |
        lambda2: float = 0.0,  # |
    ):
        super().__init__()

        self.dim = dim
        self.notears = (
            NotearsMLP(dims=[dim, *nonlinear_dims])
            if sem_type == "mlp"
            else NotearsSobolev(dim, 5)
        )

        self.rho = rho
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def _squared_loss(self, x, x_hat):
        n = x.shape[0]
        return 0.5 / n * torch.sum((x_hat - x) ** 2)

    def h_func(self):
        return self.notears.h_func()

    def loss(self, x, x_hat):
        loss = self._squared_loss(x, x_hat)
        h_val = self.notears.h_func()
        penalty = 0.5 * self.rho * h_val * h_val + self.alpha * h_val
        l2_reg = 0.5 * self.lambda2 * self.notears.l2_reg()
        l1_reg = self.lambda1 * self.notears.fc1_l1_reg()

        return loss + penalty + l2_reg + l1_reg

    def forward(self, x: torch.Tensor):
        x_hat = self.notears(x)
        loss = self.loss(x, x_hat)

        return x_hat, loss


class lit_NOTEARS(pl.LightningModule):
    def __init__(
        self,
        model: NOTEARS,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
        w_threshold: float = 0.3,
        n: int = 200,
        s: int = 9,
        K: int = 5,
        dag_type="ER",
        dim: int = 5,
        save_hyperparams: bool = True,
    ):
        super().__init__()

        self.model = model
        self.h = np.inf

        self.h_tol, self.rho_max = h_tol, rho_max
        self.w_threshold = w_threshold

        # We need a way to cope with NOTEARS dual
        #   ascent strategy.
        self.automatic_optimization = False

        if save_hyperparams:
            self.save_hyperparameters(ignore=["model"])

        if dag_type == "ER":
            dag = 1
        elif dag_type == "SF":
            dag = 2
        elif dag_type == "BP":
            dag = 3

        self.log_dict({"s": s, "dag_type": dag, "dim": dim, "n": n, "K": K})

    def _dual_ascent_step(self, x, optimizer: torch.optim.Optimizer) -> Tuple[float]:
        h_new = None

        while self.model.rho < self.rho_max:

            def closure():
                optimizer.zero_grad()
                _, loss = self.model(x)
                self.manual_backward(loss)
                return loss

            optimizer.step(closure)
            with torch.no_grad():
                h_new = self.model.h_func().item()
            if h_new > 0.25 * self.h:
                self.model.rho *= 10
            else:
                break
        self.model.alpha += self.model.rho * h_new
        return self.model.alpha, self.model.rho, h_new

    def training_step(self, batch, batch_idx) -> Any:
        opt = self.optimizers()

        (X,) = batch

        alpha, rho, h = self._dual_ascent_step(X, opt)
        self.h = h

        self.log("h", h, on_step=True, logger=True, prog_bar=True)
        self.log("rho", rho, on_step=True, logger=True, prog_bar=True)
        self.log("alpha", alpha, on_step=True, logger=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return ut.LBFGSBScipy(self.model.parameters())

    def A(self, grad: bool = False) -> np.ndarray:
        if grad:
            B_est = self.model.notears.fc1_to_adj_grad()

        else:
            B_est = self.model.notears.fc1_to_adj()
            B_est[np.abs(B_est) < self.w_threshold] = 0
            B_est[B_est > 0] = 1

        return B_est

    def test_step(self, batch, batch_idx) -> Any:
        B_est = self.A()
        B_true = self.trainer.datamodule.DAG

        self.log_dict(ut.count_accuracy(B_true, B_est))
