import sys
from pathlib import Path

PROJECT_ROOT = Path().resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


import warnings

warnings.filterwarnings("ignore")


import glob
import itertools
import logging
import random

import dill
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from src.dag_learner.data import Data
from src.utils.dag import sample_corruptions

logger = logging.getLogger()
logger.setLevel(logging.INFO)



np.random.seed(42)
random.seed(42)
torch.manual_seed(42)




@hydra.main(version_base=None, config_path="../conf", config_name="config_sensitivity")
def main(cfg: DictConfig):
    
    
    PATH_SAVE = cfg.PATH_SAVE_DATA
    Path(PATH_SAVE).mkdir(parents=True, exist_ok=True)

    d = cfg.d
    n_train = cfg.n_train
    n_test = cfg.n_test
    n_dags_per_SHD = cfg.n_dags_per_SHD
    sem_type = cfg.sem_type
    list_s = cfg.list_s
    n_repeat = cfg.n_trials
    list_repeat = np.arange(n_repeat)
    graph_type = cfg.graph_type
    batch_size = n_train
    list_SHD = cfg.list_SHD
        
    for  s, _ in itertools.product( list_s, list_repeat):
        
        data_config_train={
            "dim": d,
            "s0": s,
            "n_train": n_train,
            "n_test": n_test,
            "sem_type": sem_type,
            "dag_type": graph_type,
            "batch_size": batch_size,
        }


        data_dic = {}
        
        D_train = Data(**data_config_train)
        D_train.setup()
        X_train, X_test_clean = D_train.train.dataset[D_train.train.indices][0].numpy(), D_train.test.dataset[D_train.test.indices][0].numpy()
        
        data_dic["D_train"] = D_train
        data_dic["X_train"] = X_train
        data_dic["X_test_clean"] = X_test_clean 
        data_dic["d"] = d
        data_dic["s"] = s
        data_dic["n_train"] = n_train
        data_dic["n_test"] = n_test
        data_dic["DAG"] = D_train.DAG
        data_dic["dag_type"] = graph_type
        data_dic["sem_type"] = sem_type

        corrupted_dags_dict = {}
        for SHD in list_SHD:
            corrupted_dags_dict[SHD] = [sample_corruptions(D_train.DAG, SHD) for k in range(n_dags_per_SHD)]
        data_dic["corrupted_dags_dict"] = corrupted_dags_dict
        k= 0
        id = f"id_{k}_d_{d}_s_{s}_n_{n_train}_sem_{sem_type}"
        while (PATH_SAVE +  id) in glob.glob(PATH_SAVE + "*"):
            k += 1
            id = f"id_{k}_d_{d}_s_{s}_n_{n_train}_sem_{sem_type}"

        filehandler = open(PATH_SAVE + id,"wb")
        dill.dump(data_dic, filehandler)
        filehandler.close()
        
if __name__ == "__main__":
    main()