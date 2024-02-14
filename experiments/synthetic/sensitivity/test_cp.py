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
import torch
from omegaconf import DictConfig

# dagnosis absolute
from dagnosis.utils.dag import sample_nodes_to_corrupt
from dagnosis.utils.data import sample_corrupted
from dagnosis.utils.utils import compute_TP, compute_TP_pca

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path().resolve().parent.parent.parent
print(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


@hydra.main(version_base=None, config_path="../conf", config_name="config_sensitivity")
def main(cfg: DictConfig):

    PATH_LOAD = cfg.PATH_SAVE_CP
    PATH_LOAD_DATA = cfg.PATH_SAVE_DATA

    PATH_SAVE_confdict = cfg.PATH_SAVE_CONFDICT
    PATH_SAVE_metric = cfg.PATH_SAVE_METRIC

    # create folder if does not exist
    if not os.path.exists(PATH_SAVE_confdict):
        os.makedirs(PATH_SAVE_confdict)
    if not os.path.exists(PATH_SAVE_metric):
        os.makedirs(PATH_SAVE_metric)

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    n_runs = 1
    n_corrupted = cfg.n_corrupted
    list_SHD = cfg.list_SHD

    for filename in glob.glob(PATH_LOAD + "*"):

        logger.info(filename)

        with open(filename, "rb") as f:
            dic_results = dill.load(f)

        data_file_name = PATH_LOAD_DATA + (filename.split("/")[-1]).split("_alpha")[0]

        with open(data_file_name, "rb") as f:
            dic_data = dill.load(f)

        corrupted_dag_dict = dic_data[
            "corrupted_dags_dict"
        ]  # dictionary with key the SHD, and value the list of corrupted dags

        dic_SHD = (
            {}
        )  # dictionnary with key the SHD, and value the list of conformal estimators (one for each corrupted dag)
        list_conf_gt, list_conf_pca, list_conf_dagma = (
            dic_results["conf_gt"],
            dic_results["conf_pca"],
            dic_results["conf_dagma"],
        )

        for SHD in list_SHD:
            dic_SHD[SHD] = dic_results[f"conf_SHD_{SHD}"]

        scaler, pca = dic_results["scaler"], dic_results["pca"]
        D_train = dic_results["Dtrain"]
        A_gt, A_dagma = dic_results["A_gt"], dic_results["A_dagma"]

        X_test_clean = D_train.test.dataset[D_train.test.indices][0].numpy()

        dic_precision = {"gt": [], "auto": [], "pca": [], "dagma": []}
        dic_recall = {"gt": [], "auto": [], "pca": [], "dagma": []}
        dic_f1 = {"gt": [], "auto": [], "pca": [], "dagma": []}

        # Initialize the dictionnaries
        for SHD in list_SHD:
            dic_precision[f"SHD_{SHD}"] = []
            dic_recall[f"SHD_{SHD}"] = []
            dic_f1[f"SHD_{SHD}"] = []

        FP_gt, _ = compute_TP(list_conf_gt, A_gt, X_test_clean)
        FP_dagma, _ = compute_TP(list_conf_dagma, A_dagma, X_test_clean)
        FP_pca, _ = compute_TP_pca(list_conf_pca, scaler, pca, X_test_clean)

        dic_FP_corrupted_dags = {}
        for SHD in list_SHD:
            dic_FP_corrupted_dags[SHD] = []
            for i in range(len(dic_SHD[SHD])):
                list_conf_corrupted = dic_SHD[SHD][i]
                corrupted_dag = corrupted_dag_dict[SHD][i]
                FP, _ = compute_TP(list_conf_corrupted, corrupted_dag, X_test_clean)
                dic_FP_corrupted_dags[SHD].append(FP)

        for run in range(n_runs):
            list_features_corruption = sample_nodes_to_corrupt(A_gt, 2)
            list_corruption_type = ["gaussian_noise"] * len(list_features_corruption)

            print("Corrupted nodes: {}".format(list_features_corruption))

            d = len(D_train.DAG)
            noise_mean_list = np.zeros(d)
            X_test_corrupted, _, _ = sample_corrupted(
                D_train,
                n_corrupted,
                list_features_corruption,
                list_corruption_type,
                noise_mean_list=noise_mean_list,
                sample_last_layer=False,
                mean_mlp=0,
                std_mlp=0.3,
            )

            TP_gt, conf_dict_gt = compute_TP(list_conf_gt, A_gt, X_test_corrupted)
            TP_dagma, conf_dict_dagma = compute_TP(
                list_conf_dagma, A_dagma, X_test_corrupted
            )

            TP_pca, conf_dict_pca = compute_TP_pca(
                list_conf_pca, scaler, pca, X_test_corrupted
            )

            precision_gt = TP_gt / (TP_gt + FP_gt)
            recall_gt = TP_gt / len(X_test_corrupted)
            f1_gt = 2 / (1 / precision_gt + 1 / recall_gt)
            logger.info(f"f1 {f1_gt}, precision {precision_gt}, recall {recall_gt}")

            for SHD in list_SHD:

                for i in range(len(dic_SHD[SHD])):
                    list_conf_corrupted = dic_SHD[SHD][i]
                    corrupted_dag = corrupted_dag_dict[SHD][i]

                    TP, _ = compute_TP(
                        list_conf_corrupted, corrupted_dag, X_test_corrupted
                    )
                    FP = dic_FP_corrupted_dags[SHD][i]

                    dic_recall[f"SHD_{SHD}"].append(TP / len(X_test_corrupted))
                    dic_precision[f"SHD_{SHD}"].append(TP / (TP + FP))
                    # the f1 score is computed below

            dic_recall["gt"].append(TP_gt / len(X_test_corrupted))
            dic_recall["dagma"].append(TP_dagma / len(X_test_corrupted))
            dic_recall["pca"].append(TP_pca / len(X_test_corrupted))

            dic_precision["gt"].append(TP_gt / (TP_gt + FP_gt))
            dic_precision["dagma"].append(TP_dagma / (TP_dagma + FP_dagma))
            dic_precision["pca"].append(TP_pca / (TP_pca + FP_pca))

            id = filename.split("/")[-1]

            k = 0
            while str(os.path.join(PATH_SAVE_confdict, f"run_{k}_{id}")) in glob.glob(
                PATH_SAVE_confdict + "*"
            ):
                k += 1

            dic_conf = {
                "gt": conf_dict_gt,
                "pca": conf_dict_pca,
                "dagma": conf_dict_dagma,
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


if __name__ == "__main__":
    main()
