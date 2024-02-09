import logging

import numpy as np
import pandas as pd
from src.conformal_prediction.conformal import conformalized_quantile_regression
from src.utils.dag import find_markov_boundary
from tqdm import tqdm


def analyse_conformal_dict(conformal_dict):
    dictionnary_df = {}
    for feature in conformal_dict.keys():
        df = conformal_dict[feature]

        def func(truth, min_val, max_val, interval):

            return not ((truth >= min_val) & (truth <= max_val))

        df["outlier"] = df.apply(
            lambda x: func(
                x["true_val"],
                x["min"],
                x["max"],
                x["conf_interval"],
            ),
            axis=1,
        )

        dictionnary_df[feature] = df["outlier"].tolist()

    final_df = pd.DataFrame(dictionnary_df)
    isInconsistent = final_df.any(axis=1)
    final_df["inconsistent"] = isInconsistent
    return final_df


def predict_cp(X_test, list_conf, list_features, DAG):
    list_markov_boundary = find_markov_boundary(DAG)

    conformal_dict_corrupted = {}

    for feature in list_features:
        conf = list_conf[feature]
        
        conditioning_variables = list_markov_boundary[feature]

        y_test_corrupted = X_test[:, feature]

        if len(conditioning_variables) == 0:
            X_test_corrupted_cp = np.zeros((X_test.shape[0], 1))
        else:
            X_test_corrupted_cp = X_test[:, conditioning_variables]

        conformal_dict_corrupted[feature] = conf.predict(
            x_test=X_test_corrupted_cp, y_test=y_test_corrupted
        )

    return conformal_dict_corrupted


def predict_cp_pca(X_test, list_conf, list_features, scaler, pca, pca_test=None):
    if pca_test is None:
        X_test_scaled = scaler.transform(X_test)
        pca_test = pca.transform(X_test_scaled)

    conformal_dict_corrupted = {}

    for feature in list_features:
        conf = list_conf[feature]

        y_test_corrupted = X_test[:, feature]
        X_test_corrupted_cp = pca_test

        conformal_dict_corrupted[feature] = conf.predict(
            x_test=X_test_corrupted_cp, y_test=y_test_corrupted
        )

    return conformal_dict_corrupted




def train_cp(
    X_train,
    DAG,
    list_features,
    alpha,
    cal_size=0.2,
    n_search=100,
):
    """
    Train the feature wise conformal estimators.
    """
    
    list_conf = [
        None for i in range(X_train.shape[1])
    ]  # List which will contain the conformal predictors for each feature
    list_markov_boundary = find_markov_boundary(
        DAG
    )  # List of markov boundaries for each feature

    for k, feature in tqdm(enumerate(list_features)):

        logging.info(f"Processing feature {feature}")
        conditioning_variables = list_markov_boundary[feature]

        y_train_cp = X_train[:, feature]

        if len(conditioning_variables) == 0:

            # if the node is a root of the DAG, then just use a simple CI which does not depend on the input.

            X_train_cp = np.zeros((X_train.shape[0], 1))
        else:
            X_train_cp = X_train[:, conditioning_variables]

        conf = conformalized_quantile_regression(
            alpha=alpha,
            seed=0,
            cal_size=cal_size,
            n_search=n_search,
        )

        conf.fit(X_train_cp, y_train_cp)
        list_conf[feature] = conf
    return list_conf


def train_cp_pca(
    pca_train,
    X_train,
    list_features,
    alpha,
    cal_size=0.2,
    n_search=100,
):
    """Train the feature wise estimators for DATA-SUITE (PCA)
    """
    list_conf = [None for i in range(X_train.shape[1])]
    for k, feature in tqdm(enumerate(list_features)):
        logging.info(f"Processing feature {feature}")
        y_train_cp = X_train[
            :, feature
        ]  # The target variable is the feature we want to predict
        X_train_cp = pca_train  # Use the representation obtained via PCA
        conf = conformalized_quantile_regression(
            alpha=alpha,
            seed=42,
            cal_size=cal_size,
            n_search=n_search,
        )

        conf.fit(X_train_cp, y_train_cp)
        list_conf[feature] = conf
    return list_conf
