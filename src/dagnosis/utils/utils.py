# third party
import numpy as np

# dagnosis absolute
from dagnosis.utils.conformal import analyse_conformal_dict, predict_cp, predict_cp_pca


def compute_TP(list_conf, A, X_test):

    d = A.shape[0]
    conformal_dict_corrupted = predict_cp(X_test, list_conf, np.arange(d), A)
    df = analyse_conformal_dict(conformal_dict_corrupted)
    TP = len(df[df["inconsistent"] == True])  # noqa: E712
    return TP, conformal_dict_corrupted


def compute_TP_pca(list_conf, scaler, pca, X_test_corrupted):

    d = X_test_corrupted.shape[1]
    conformal_dict_corrupted = predict_cp_pca(
        X_test_corrupted, list_conf, np.arange(d), scaler, pca
    )
    df_pca = analyse_conformal_dict(conformal_dict_corrupted)
    TP_pca = len(df_pca[df_pca["inconsistent"] == True])  # noqa: E712

    return TP_pca, conformal_dict_corrupted
