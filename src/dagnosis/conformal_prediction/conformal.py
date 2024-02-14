# third party
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mapie.quantile_regression import MapieQuantileRegressor
from scipy.stats import randint, uniform
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split


class conformalized_quantile_regression:
    def __init__(
        self, alpha=0.1, scale=False, seed=42, cal_size=0.2, n_search=100
    ) -> None:
        self.alpha = alpha
        self.scale = scale
        self.seed = seed
        self.cal_size = cal_size
        self.n_search = n_search

    def fit(self, x_train, y_train):

        x_train, y_train = np.array(x_train).astype(np.float64), np.array(
            y_train,
        ).astype(np.float64)
        self.x_train, self.y_train = x_train, y_train

        X_train, X_cal, y_train, y_cal = train_test_split(
            x_train, y_train, test_size=self.cal_size, random_state=self.seed
        )  # train vs cal
        X_train_sc, X_cal_sc, y_train_sc, y_cal_sc = X_train, X_cal, y_train, y_cal

        self.range_max = np.max(y_train_sc)
        self.range_min = np.min(y_train_sc)

        estimator = LGBMRegressor(
            objective="quantile", alpha=self.alpha / 2, random_state=self.seed, n_jobs=1
        )
        params_distributions = dict(
            num_leaves=randint(low=10, high=50),
            max_depth=randint(low=3, high=20),
            n_estimators=randint(low=50, high=300),
            learning_rate=uniform(),
        )

        optim_model = RandomizedSearchCV(
            estimator,
            param_distributions=params_distributions,
            n_jobs=-1,
            n_iter=self.n_search,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            verbose=-1,
            random_state=42,
        )

        optim_model.fit(X_train_sc, y_train_sc)
        estimator = optim_model.best_estimator_

        params = {"method": "quantile", "cv": "split", "alpha": self.alpha}
        mapie = MapieQuantileRegressor(estimator, **params)
        mapie.fit(X_train_sc, y_train_sc, X_calib=X_cal_sc, y_calib=y_cal_sc)

        self.cp_model = mapie

    def predict(self, x_test, y_test):
        X_test_sc = x_test
        y_test_sc = y_test

        y_pred, y_pis = self.cp_model.predict(X_test_sc)

        lower_bound = y_pis[:, 0, 0]
        upper_bound = y_pis[:, 1, 0]

        header = ["min", "max", "true_val", "conf_interval"]
        size = upper_bound - lower_bound

        table = np.vstack([lower_bound.T, upper_bound.T, y_test_sc, size.T]).T
        df = pd.DataFrame(table, columns=header)

        feature_range = self.range_max - self.range_min
        df["norm_interval"] = df["conf_interval"] / feature_range
        return df
