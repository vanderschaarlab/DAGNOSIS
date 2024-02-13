import copy

import numpy as np
import src.dag_learner.simulate as sm
from scipy.special import expit as sigmoid


def sample_corrupted(
    D, n_samples, list_feature, list_corruption_type, noise_mean_list=None, mean_linear = 5, std_linear = 1, mean_mlp = 2, std_mlp = 1., sample_last_layer= True
):
    """Sample a corrupted dataset where some SEMs have been corrupted."""
    

    list_corrupted_SEMs = copy.deepcopy(D.list_SEMs)
    list_corrupted_parameters = copy.deepcopy(D.list_parameters)

    for feature, corruption_type in zip(list_feature, list_corruption_type):
        pa_size = sm.find_parent_size(D.DAG, feature)
        list_corrupted_SEMs, list_corrupted_parameters = sm.modify_single_sem(
            pa_size,
            corruption_type,
            feature,
            list_corrupted_SEMs,
            list_corrupted_parameters,
            mean_linear=mean_linear,
            std_linear=std_linear,
            mean_mlp=mean_mlp,
            std_mlp=std_mlp,
            sample_last_layer=sample_last_layer
        )

    X_test_corrupted = sm.simulate_sem_by_list(
        D.DAG,
        n_samples,
        list_corrupted_SEMs,
        noise_scale=None,
        noise_mean_list=noise_mean_list,
    )
    return X_test_corrupted, list_corrupted_SEMs, list_corrupted_parameters
