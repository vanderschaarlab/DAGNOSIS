# stdlib
import glob
import logging
import os
import pickle
import random
import sys
import warnings
from pathlib import Path

# third party
import dill
import hydra
import numpy as np
from omegaconf import DictConfig

# dagnosis absolute
from dagnosis.utils.dag import sample_nodes_to_corrupt
from dagnosis.utils.data import sample_corrupted
from dagnosis.utils.utils import compute_TP, compute_TP_pca

PROJECT_ROOT = Path().resolve().parent.parent
print(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    PATH_LOAD = cfg.PATH_SAVE_CP
    PATH_SAVE_confdict = cfg.PATH_SAVE_CONFDICT
    PATH_SAVE_metric = cfg.PATH_SAVE_METRIC

    # create folder if does not exist
    if not os.path.exists(PATH_SAVE_confdict):
        os.makedirs(PATH_SAVE_confdict)
    if not os.path.exists(PATH_SAVE_metric):
        os.makedirs(PATH_SAVE_metric)

    np.random.seed(42)
    random.seed(42)

    n_runs = 1
    n_corrupted = cfg.n_corrupted

    for filename in glob.glob(PATH_LOAD + "*"):
        logger.info(filename)

        with open(filename, "rb") as f:
            dic_results = dill.load(f)

        list_conf_gt, list_conf_auto, list_conf_pca, list_conf_notears = (
            dic_results["conf_gt"],
            dic_results["conf_autoregressive"],
            dic_results["conf_pca"],
            dic_results["conf_notears"],
        )
        scaler, pca = dic_results["scaler"], dic_results["pca"]
        D_train = dic_results["Dtrain"]
        A_gt, A_auto, A_notears = (
            dic_results["A_gt"],
            dic_results["A_auto"],
            dic_results["A_notears"],
        )
        X_test_clean = D_train.test.dataset[D_train.test.indices][0].numpy()

        assert len(X_test_clean) == cfg.n_test

        dic_precision = {"gt": [], "auto": [], "pca": [], "notears": []}
        dic_recall = {"gt": [], "auto": [], "pca": [], "notears": []}
        dic_f1 = {"gt": [], "auto": [], "pca": [], "notears": []}

        FP_gt, _ = compute_TP(list_conf_gt, A_gt, X_test_clean)
        FP_auto, _ = compute_TP(list_conf_auto, A_auto, X_test_clean)
        FP_pca, _ = compute_TP_pca(list_conf_pca, scaler, pca, X_test_clean)
        FP_notears, _ = compute_TP(list_conf_notears, A_notears, X_test_clean)

        for run in range(n_runs):
            list_features_corruption = sample_nodes_to_corrupt(A_gt, 2)
            list_corruption_type = ["gaussian_noise"] * len(list_features_corruption)

            print("Corrupted nodes: {}".format(list_features_corruption))

            d = len(D_train.DAG)
            noise_mean_list = np.zeros(d)
            (
                X_test_corrupted,
                list_corrupted_SEMs,
                list_corrupted_parameters,
            ) = sample_corrupted(
                D_train,
                n_corrupted,
                list_features_corruption,
                list_corruption_type,
                noise_mean_list=noise_mean_list,
            )

            TP_gt, conf_dict_gt = compute_TP(list_conf_gt, A_gt, X_test_corrupted)
            TP_auto, conf_dict_auto = compute_TP(
                list_conf_auto, A_auto, X_test_corrupted
            )
            TP_notears, conf_dict_notears = compute_TP(
                list_conf_notears, A_notears, X_test_corrupted
            )
            TP_pca, conf_dict_pca = compute_TP_pca(
                list_conf_pca, scaler, pca, X_test_corrupted
            )

            dic_recall["gt"].append(TP_gt / len(X_test_corrupted))
            dic_recall["auto"].append(TP_auto / len(X_test_corrupted))
            dic_recall["notears"].append(TP_notears / len(X_test_corrupted))
            dic_recall["pca"].append(TP_pca / len(X_test_corrupted))

            dic_precision["gt"].append(TP_gt / (TP_gt + FP_gt))
            dic_precision["auto"].append(TP_auto / (TP_auto + FP_auto))
            dic_precision["notears"].append(TP_notears / (TP_notears + FP_notears))
            dic_precision["pca"].append(TP_pca / (TP_pca + FP_pca))

            id = filename.split("/")[-1]

            k = 0
            while str(os.path.join(PATH_SAVE_confdict, f"run_{k}_{id}")) in glob.glob(
                PATH_SAVE_confdict + "*"
            ):
                k += 1

            dic_conf = {
                "gt": conf_dict_gt,
                "auto": conf_dict_auto,
                "pca": conf_dict_pca,
                "notears": conf_dict_notears,
            }
            file_name = f"run_{k}_{id}"
            file_name = os.path.join(PATH_SAVE_confdict, file_name)
            with open(file_name, "wb") as handle:
                pickle.dump(dic_conf, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for method in dic_recall.keys():
            dic_f1[method].append(
                2
                / (
                    1 / np.array(dic_precision[method])
                    + 1 / np.array(dic_recall[method])
                )
            )

        dic_metrics = {"f1": dic_f1, "precision": dic_precision, "recall": dic_recall}
        id = filename.split("/")[-1]

        file_name = os.path.join(PATH_SAVE_metric, id)
        with open(file_name, "wb") as handle:
            pickle.dump(dic_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(dic_metrics)


if __name__ == "__main__":
    main()
