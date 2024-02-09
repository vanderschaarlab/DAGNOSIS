import sys
from pathlib import Path

PROJECT_ROOT = Path().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import warnings

warnings.filterwarnings("ignore")


import glob
import logging
import random
from copy import deepcopy
from typing import Any

import dill
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from src.conformal_prediction.representation import representation_class_based
from src.utils.conformal import train_cp, train_cp_pca
from src.utils.dag import get_DAG

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> Any:

    PATH_LOAD = cfg.PATH_SAVE_DATA
    PATH_SAVE = cfg.PATH_SAVE_CP
    # if the folder does not exist, create it
    Path(PATH_SAVE).mkdir(parents=True, exist_ok=True)

    confidence = cfg.confidence
    cal_train_ratio = cfg.cal_train_ratio
    notears_type = cfg.notears_type

    for filename in glob.glob(PATH_LOAD + "*"):
        logger.info(f"Processing {filename}")

        with open(filename, "rb") as f:
            data = dill.load(f)

        d = data["d"]
        D_train = data["D_train"]
        X_train = data["X_train"]
        X_train_copy = deepcopy(X_train)

        alpha = confidence / d  # Bonferroni correction

        # Get the adjacency matrices for the different baselines
        logger.info("Extracting the GT DAG")
        A_gt = get_DAG(D_train, "gt")
        logger.info("Extracting the autoregressive DAG")
        A_auto = get_DAG(D_train, "autoregressive")
        D_train_copy = deepcopy(D_train)
        logger.info("Discovering a DAG with NOTEARS")
        A_notears = get_DAG(D_train_copy, notears_type)

        artifacts = {}

        ##########################################
        ####### GT DAG ###########################
        ##########################################
        
        artifacts["A_gt"] = A_gt
        list_features = np.arange(d)

        # Train the regressors for CQR
        logger.info("DAGNOSIS GT")
        # With GT DAG
        list_conf_gt = train_cp(
            X_train=X_train,
            DAG=A_gt,
            list_features=list_features,
            alpha=alpha,
            cal_size=cal_train_ratio,
        )

        ##########################################
        ####### DATA-SUITE #######################
        ##########################################
        logger.info("DATA SUITE")
        np.testing.assert_array_equal(X_train, X_train_copy)
        rep_dim = int(np.ceil(X_train.shape[1] / 2))
        pca_train, _, pca, scaler = representation_class_based(
            X_train, X_train, rep_dim, "pca"
        )
        
        #Get the list of conformal estimators
        list_conf_pca = train_cp_pca(
            pca_train, X_train, list_features, alpha, cal_train_ratio
        )

        #########################################
        ######## AUTOREGRESSIVE #################
        #########################################
        np.testing.assert_array_equal(X_train, X_train_copy)

        logger.info("DAGNOSIS Autoregressive")
        
        artifacts["A_auto"] = A_auto
        #Get the list of conformal estimators
        list_conf_autoregressive = train_cp(
            X_train,
            A_auto,
            list_features,
            alpha,
            cal_train_ratio,
        )

        ##########################################
        ########  NOTEARS ########################
        ##########################################
        np.testing.assert_array_equal(X_train, X_train_copy)
        logger.info("DAGNOSIS NOTEARS")
        artifacts["A_notears"] = A_notears
        
        #Get the list of conformal estimators
        list_conf_notears = train_cp(
            X_train,
            A_notears,
            list_features,
            alpha,
            cal_train_ratio,
        )

        artifacts["conf_gt"] = list_conf_gt
        artifacts["conf_autoregressive"] = list_conf_autoregressive
        artifacts["conf_notears"] = list_conf_notears
        artifacts["conf_pca"] = list_conf_pca
        artifacts["Dtrain"] = D_train
        artifacts["scaler"] = scaler
        artifacts["pca"] = pca

        id = filename.split("/")[-1]
        id += f"_alpha_{alpha}_calratio_{cal_train_ratio}"
        filehandler = open(PATH_SAVE + id, "wb")
        dill.dump(artifacts, filehandler)
        filehandler.close()

if __name__ == "__main__":
    main()