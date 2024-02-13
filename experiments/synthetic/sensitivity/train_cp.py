import sys
from pathlib import Path

PROJECT_ROOT = Path().resolve().parent.parent.parent
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

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="../conf", config_name="config_sensitivity")
def main(cfg: DictConfig) -> Any:
        
    PATH_LOAD = cfg.PATH_SAVE_DATA
    PATH_SAVE = cfg.PATH_SAVE_CP
    
    # if the folder does not exist, create it
    Path(PATH_SAVE).mkdir(parents=True, exist_ok=True)

    confidence = cfg.confidence
    cal_train_ratio = cfg.cal_train_ratio
    dagma_type = cfg.dagma_type

    for filename in glob.glob(PATH_LOAD +'*'):
        logger.info(f"Processing {filename}")
    
        with open(filename, 'rb') as f:
            data = dill.load(f)
        
        
        d = data["d"]
        D_train = data["D_train"]
        X_train = data["X_train"]

        alpha = confidence/d
        

        A_gt = get_DAG(D_train, "gt")
        D_train2 = deepcopy(D_train)
        A_dagma = get_DAG(D_train2, dagma_type)


        logger.info(f"alpha: {alpha} cal_size: {cal_train_ratio}")
        artifacts = {}


        logger.info("GT DAG")
        artifacts["A_gt"] = A_gt
        list_features = np.arange(d)
        list_conf_gt = train_cp(X_train = X_train, DAG= A_gt, list_features = list_features, alpha = alpha, cal_size = cal_train_ratio)
        
        logger.info("DAGNOSIS DAGMA")
        artifacts["A_dagma"] = A_dagma 
        list_conf_dagma = train_cp(X_train = X_train, DAG= A_dagma, list_features = list_features, alpha = alpha, cal_size = cal_train_ratio)
               

        #########################################
        ######## CORRUPTED DAGS #################
        #########################################
        logger.info("CORRUPTED DAGS")
        corrupted_dags_dict = data["corrupted_dags_dict"]
        for SHD in corrupted_dags_dict.keys():
            logger.info(f"SHD: {SHD}")
            artifacts[f"conf_SHD_{SHD}"] = []
            list_DAGS = corrupted_dags_dict[SHD]
            for corrupted_DAG in list_DAGS:
                artifacts[f"conf_SHD_{SHD}"].append(train_cp(X_train=X_train, DAG = corrupted_DAG, list_features=list_features, alpha= alpha, cal_size =  cal_train_ratio))


        ##########################################
        ####### DATA-SUITE #######################
        ##########################################
        logger.info("DATA SUITE")
        rep_dim = int(np.ceil(X_train.shape[1] / 2))
        pca_train, _, pca, scaler =  representation_class_based(X_train, X_train, rep_dim, "pca")
        list_conf_pca = train_cp_pca(pca_train = pca_train, X_train = X_train, list_features = list_features, alpha = alpha, cal_size = cal_train_ratio)




        
        artifacts["conf_gt"] = list_conf_gt
        artifacts["conf_dagma"] = list_conf_dagma
        artifacts["conf_pca"] = list_conf_pca
        artifacts["Dtrain"] = D_train
        artifacts["scaler"] = scaler
        artifacts["pca"] = pca

        id = filename.split("/")[-1]
        id += f"_alpha_{alpha}_calratio_{cal_train_ratio}"
        filehandler = open(PATH_SAVE + id,"wb")
        dill.dump(artifacts, filehandler)
        filehandler.close()

if __name__ == "__main__":
    main()
    