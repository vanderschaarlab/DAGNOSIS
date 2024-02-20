# stdlib
import logging
import random
import sys
import warnings
from pathlib import Path

# third party
import dill
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# dagnosis absolute
from dagnosis.conformal_prediction.representation import representation_class_based
from dagnosis.utils.conformal import (
    analyse_conformal_dict,
    predict_cp,
    predict_cp_pca,
    train_cp,
    train_cp_pca,
)
from dagnosis.utils.dag import get_adult_DAG
from dagnosis.utils.data_loader import load_adult, preprocess_adult

PROJECT_ROOT = Path().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

warnings.filterwarnings("ignore")


def complement(a1, a2):
    return list(set(a2) - set(a1))


def main(seed):
    ###################################################################
    ############### DATASET PREPROCESSING #############################
    ###################################################################
    random.seed(seed)
    np.random.seed(seed)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    dataset = load_adult()
    dataset = preprocess_adult(dataset)
    df_sex_1 = dataset.query("sex ==1")

    salary_1_idx = dataset.query("sex == 0 & income == 1")
    salary_0_idx = dataset.query("sex == 0 & income == 0")
    X = df_sex_1.drop(["income"], axis=1)
    y = df_sex_1["income"]

    name_features = dataset.columns.tolist()
    feature_to_index = {}
    for i in range(len(name_features)):
        feature_to_index[name_features[i]] = i

    list_edges_name = [
        ["age", "workclass"],
        ["age", "marital-status"],
        ["age", "occupation"],
        ["age", "relationship"],
        ["age", "capital-gain"],
        ["age", "capital-loss"],
        ["age", "hours-per-week"],
        ["marital-status", "occupation"],
        ["marital-status", "relationship"],
        ["marital-status", "hours-per-week"],
        ["race", "workclass"],
        ["race", "fnlwgt"],
        ["race", "marital-status"],
        ["race", "occupation"],
        ["race", "relationship"],
        ["sex", "workclass"],
        ["sex", "marital-status"],
        ["sex", "occupation"],
        ["sex", "relationship"],
        ["sex", "capital-gain"],
        ["native-country", "marital-status"],
        ["native-country", "occupation"],
        ["native-country", "relationship"],
    ]
    list_edges = [
        [feature_to_index[feat1], feature_to_index[feat2]]
        for (feat1, feat2) in list_edges_name
    ]

    cal_size = 0.2
    d = X.shape[1]
    alpha = 0.1 / d

    detection_on_sex_gt = []
    detection_on_sex_pca = []
    detection_on_sex_pc = []

    detection_all_gt = []
    detection_all_pca = []
    detection_all_pc = []

    detection_gt = []
    detection_pca = []
    detection_pc = []

    list_accuracy_gt = []
    list_accuracy_pca = []
    list_accuracy_pc = []
    list_accuracy_supervised = []

    for k in range(10):
        logger.info(f"k: {k}")

        ########################################################
        ################ DATA PROCESSING PART ##################
        ########################################################
        n_women_rich = 1000 * k
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, shuffle=True, random_state=seed
        )

        logger.info(f"Len test: {len(X_test)}")

        X_train = np.vstack([X_train, salary_0_idx.drop(["income"], axis=1)]).astype(
            np.float64
        )
        X_test = np.vstack(
            [X_test, salary_1_idx.drop(["income"], axis=1)[:n_women_rich]]
        ).astype(np.float64)

        y_train = np.hstack([y_train, salary_0_idx["income"]])
        y_test = np.hstack([y_test, salary_1_idx["income"][:n_women_rich]])

        ###############################
        ### SUPERVISED BASELINE #######
        ###############################

        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        clf.fit(X_train, y_train)
        acc_test_supervised = accuracy_score(y_test, clf.predict(X_test))
        list_accuracy_supervised.append(acc_test_supervised)

        ##################################
        ##### DAGNOSIS GT#################
        ##################################

        logger.info("GT DAG")
        A_gt = get_adult_DAG(name_features)

        list_features = np.arange(d)
        list_conf_gt = train_cp(X_train, A_gt, list_features, alpha, cal_size)

        sex_feature_index = 9

        conformal_dict_corrupted = predict_cp(X_test, list_conf_gt, np.arange(d), A_gt)
        df = analyse_conformal_dict(
            conformal_dict_corrupted
        )  # This is a dataframe where each column denotes if the sample is corrupted or not on the i-th feature

        inconsistent_samples_indices = df.index[
            df["inconsistent"] == 1
        ].tolist()  # indices of the samples which are inconsistent

        inconsistent_women_gt = np.intersect1d(
            df.index[df[sex_feature_index] == 1].tolist(),
            conformal_dict_corrupted[sex_feature_index]
            .index[conformal_dict_corrupted[sex_feature_index]["true_val"] == 0]
            .tolist(),
        )
        inconsistent_women_all_gt = np.intersect1d(
            df.index[df["inconsistent"] == 1].tolist(),
            conformal_dict_corrupted[sex_feature_index]
            .index[conformal_dict_corrupted[sex_feature_index]["true_val"] == 0]
            .tolist(),
        )

        consistent_indices = complement(
            inconsistent_samples_indices, np.arange(len(X_test))
        )

        y_pred = clf.predict(X_test[consistent_indices, :])
        accuracy_gt = accuracy_score(y_test[consistent_indices], y_pred)
        list_accuracy_gt.append(accuracy_gt)

        detection_on_sex_gt.append(len(inconsistent_women_gt))
        detection_all_gt.append(len(inconsistent_women_all_gt))
        detection_gt.append(len(inconsistent_samples_indices))

        #########################################
        ######## DAGNOSIS PC ####################
        #########################################

        logger.info("DAGNOSIS PC")

        A_gt = get_adult_DAG(name_features, list_edges=list_edges)
        list_features = np.arange(d)
        list_conf_gt = train_cp(X_train, A_gt, list_features, alpha, cal_size)

        conformal_dict_corrupted = predict_cp(X_test, list_conf_gt, np.arange(d), A_gt)
        df = analyse_conformal_dict(
            conformal_dict_corrupted
        )  # This is a dataframe where each column denotes if the sample is corrupted or not on the i-th feature
        inconsistent_samples_indices = df.index[df["inconsistent"] == 1].tolist()

        inconsistent_women_pc = np.intersect1d(
            df.index[df[sex_feature_index] == 1].tolist(),
            conformal_dict_corrupted[sex_feature_index]
            .index[conformal_dict_corrupted[sex_feature_index]["true_val"] == 0]
            .tolist(),
        )
        inconsistent_women_all_pc = np.intersect1d(
            df.index[df["inconsistent"] == 1].tolist(),
            conformal_dict_corrupted[sex_feature_index]
            .index[conformal_dict_corrupted[sex_feature_index]["true_val"] == 0]
            .tolist(),
        )

        consistent_indices = complement(
            inconsistent_samples_indices, np.arange(len(X_test))
        )
        y_pred = clf.predict(X_test[consistent_indices, :])
        accuracy_pc = accuracy_score(y_test[consistent_indices], y_pred)
        list_accuracy_pc.append(accuracy_pc)

        detection_on_sex_pc.append(len(inconsistent_women_pc))
        detection_all_pc.append(len(inconsistent_women_all_pc))
        detection_pc.append(len(inconsistent_samples_indices))

        #####################################
        ###### DATA SUITE ###################
        #####################################
        pca_train, _, pca, scaler = representation_class_based(
            X_train, X_train, 8, "pca"
        )

        list_features = np.arange(d)
        logger.info("Data-SUITE")
        sex_feature_index = 9
        list_conf_pca = train_cp_pca(pca_train, X_train, list_features, alpha, cal_size)
        conformal_dict_corrupted_ds = predict_cp_pca(
            X_test, list_conf_pca, np.arange(d), scaler, pca
        )
        df_ds = analyse_conformal_dict(
            conformal_dict_corrupted_ds
        )  # Dataframe of inconsistences for DS
        inconsistent_samples_indices_ds = df_ds.index[
            df_ds["inconsistent"] == 1
        ].tolist()
        inconsistent_women_ds = np.intersect1d(
            df_ds.index[df_ds[sex_feature_index] == 1].tolist(),
            conformal_dict_corrupted_ds[sex_feature_index]
            .index[conformal_dict_corrupted_ds[sex_feature_index]["true_val"] == 0]
            .tolist(),
        )

        inconsistent_women_all_ds = np.intersect1d(
            df_ds.index[df_ds["inconsistent"] == 1].tolist(),
            conformal_dict_corrupted_ds[sex_feature_index]
            .index[conformal_dict_corrupted_ds[sex_feature_index]["true_val"] == 0]
            .tolist(),
        )

        detection_on_sex_pca.append(len(inconsistent_women_ds))
        detection_all_pca.append(len(inconsistent_women_all_ds))
        detection_pca.append(len(inconsistent_samples_indices_ds))

        consistent_indices_ds = complement(
            inconsistent_samples_indices_ds, np.arange(len(X_test))
        )
        y_pred = clf.predict(X_test[consistent_indices_ds, :])
        accuracy_ds = accuracy_score(y_test[consistent_indices_ds], y_pred)
        list_accuracy_pca.append(accuracy_ds)

        list_accuracy = [
            list_accuracy_gt,
            list_accuracy_pc,
            list_accuracy_pca,
            list_accuracy_supervised,
        ]
        detection_all = [detection_all_gt, detection_all_pc, detection_all_pca]
        detection_on_sex = [
            detection_on_sex_gt,
            detection_on_sex_pc,
            detection_on_sex_pca,
        ]
        detection = [detection_gt, detection_pc, detection_pca]

        artifacts = {}
        artifacts["accuracy"] = list_accuracy
        artifacts["detection_all"] = detection_all
        artifacts["detection_on_sex"] = detection_on_sex
        artifacts["detection"] = detection

        print(artifacts)

        with open(f"artifacts_adult/artifacts_final_seed_{seed}", "wb") as f:
            dill.dump(artifacts, f)


if __name__ == "__main__":
    for i in range(5):
        seed = i
        main(seed)
