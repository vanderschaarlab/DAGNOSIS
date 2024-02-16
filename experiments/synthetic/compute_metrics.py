# stdlib
import glob
import pickle
import sys
from pathlib import Path

# third party
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

PROJECT_ROOT = Path().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    PATH_LOAD = cfg.PATH_SAVE_METRIC

    dic_metrics = []
    for filename in sorted(glob.glob(PATH_LOAD + "*")):
        with open(filename, "rb") as f:
            dic_results = pickle.load(f)

        s = int(filename.split("/")[-1].split("_")[5])
        f1_dict = dic_results["f1"]
        precision_dict = dic_results["precision"]
        recall_dict = dic_results["recall"]
        for method in f1_dict.keys():
            f1 = f1_dict[method][0][0]
            precision = precision_dict[method][0]
            recall = recall_dict[method][0]
            dic_metrics.append(
                {
                    "number of edges": s,
                    "f1": f1,
                    "method": method,
                    "precision": precision,
                    "recall": recall,
                }
            )

    df = pd.DataFrame(dic_metrics)

    for method in ["auto", "pca", "gt", "notears"]:
        print(method)
        for s in [10, 20, 30, 40]:
            print("#" * 42)
            print(f"s: {s}")
            x = df[df["number of edges"] == s]
            x_gt = x[x["method"] == method]
            print(
                "f1",
                round(np.mean(x_gt["f1"]), 2),
                round(1.96 / np.sqrt(len(x_gt)) * np.std(x_gt["f1"]), 2),
            )
            print(
                "precision",
                round(np.mean(x_gt["precision"]), 2),
                round(1.96 / np.sqrt(len(x_gt)) * np.std(x_gt["precision"]), 2),
            )
            print(
                "recall",
                round(np.mean(x_gt["recall"]), 2),
                round(1.96 / np.sqrt(len(x_gt)) * np.std(x_gt["recall"]), 2),
            )


if __name__ == "__main__":
    main()
