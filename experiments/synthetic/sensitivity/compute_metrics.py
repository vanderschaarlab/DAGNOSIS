import sys
from pathlib import Path

PROJECT_ROOT = Path().resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import glob
import pickle

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config_sensitivity")
def main(cfg: DictConfig) -> None:
    PATH_LOAD  = cfg.PATH_SAVE_METRIC
    dic_all = []
    for filename in sorted(glob.glob(PATH_LOAD + "*")):
        print(filename)
        with open(filename, "rb") as f:
            dic_results = pickle.load(f)
            
        
        f1_dict = dic_results["f1"]
        precision_dict = dic_results["precision"]
        recall_dict = dic_results["recall"]

    
        for method in f1_dict.keys(): 
            f1_list = f1_dict[method][0]
           

            precision_list = precision_dict[method]
            recall_list = recall_dict[method]

            for i in range(len(f1_list)):
                f1 = f1_list[i]
                precision = precision_list[i]
                recall = recall_list[i]
                dic_all.append({"method": method, "f1": f1, "precision": precision, "recall": recall })
                

    df = pd.DataFrame(dic_all)

    for method in ["pca", "dagma", "SHD_10", "SHD_20", "SHD_30", "SHD_40"]:
        print(method)
        print("#"*42)
        x = df[df["method"] == method]
        
        print("f1", round(np.mean(x["f1"]),2), round(1.96/np.sqrt(len(x))*np.std(x["f1"]),2))
        print("precision", round(np.mean(x["precision"]),2), round(1.96/np.sqrt(len(x))*np.std(x["precision"]), 2))
        print("recall", round(np.mean(x["recall"]), 2), round(1.96/np.sqrt(len(x))*np.std(x["recall"]), 2))

if __name__ == "__main__":
    main()
