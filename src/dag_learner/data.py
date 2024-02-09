import copy
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import src.dag_learner.simulate as sm
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class Data(pl.LightningDataModule):
    def __init__(
        self,
        dim: int = 20,  # Amount of vars
        s0: int = 40,  # Expected amount of edges
        sem_type: str = "mim",  # SEM-type (
        #   'mim' -> index model,
        #   'mlp' -> multi layer perceptrion,
        #   'gp' -> gaussian process,
        #   'gp-add' -> additive gp)
        dag_type: str = "ER",  # Random graph type (
        #   'ER' -> Erdos-Renyi,
        #   'SF' -> Scale Free,
        #   'BP' -> BiPartite)
        n_train: int = 1000,
        n_test: int = 10000,
        seed: int = 42,
        batch_size: int = 32,
    ):

        super().__init__()
        self.prepare_data_per_node = False
        self.dim = dim
        self.s0 = s0
        self.sem_type = sem_type
        self.dag_type = dag_type

        self.n_train = n_train
        self.n_test = n_test
        self.N = self.n_train + self.n_test

        self.batch_size = (
            batch_size  # Will be used for NOTEARS training (DAG discovery)
        )

        self.seed = seed
        self.DAG = None

        self._simulate()
        self._generate_SEM()
        self._sample()

    def _simulate(self) -> None:
        self.DAG = sm.simulate_dag(self.dim, self.s0, self.dag_type)
        self._id = hash(
            self.DAG.__repr__() + self.DAG.__array_interface__["data"][0].__repr__()
        )

    def _generate_SEM(self):
        self.list_SEMs, self.list_parameters = sm.generate_list_SEM(
            self.DAG, self.sem_type
        )
        self.list_corrupted_SEMs = copy.deepcopy(self.list_SEMs)

    def _sample(self) -> None:
        assert self.DAG is not None, "No DAG simulated yet"
        assert self.list_SEMs is not None, "SEMs not simulated yet"
        self.X = sm.simulate_sem_by_list(
            self.DAG, self.N, self.list_SEMs, noise_scale=None
        )

    def setup(self, stage: Optional[str] = None) -> None:
        assert self.DAG is not None, "No DAG simulated yet"
        assert self.X is not None, "No SEM simulated yet"

        DX = TensorDataset(torch.from_numpy(self.X).float())

        self.train, self.test = random_split(
            DX,
            [self.n_train, self.n_test],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def resample(self) -> None:
        """
        Resamples a new DAG and SEM
        Resets the train and test sets

        """
        self._simulate()
        self._sample()
        self.setup()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=os.cpu_count() // 3
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test, batch_size=len(self.test), num_workers=os.cpu_count()
        )


class Subset(pl.LightningDataModule):
    def __init__(
        self, X: np.ndarray, train_size_ratio: float = 0.5, batch_size: int = 256
    ) -> None:
        super().__init__()

        self.train_size_ratio = train_size_ratio
        self.batch_size = batch_size

        self.X = X
        self.N = self.X.shape[0]

    def setup(self):
        DX = TensorDataset(torch.from_numpy(self.X))

        _train_size = np.floor(self.N * self.train_size_ratio)
        self.train, self.test = random_split(
            DX, [int(_train_size), int(self.N - _train_size)]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=os.cpu_count()
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=os.cpu_count()
        )
