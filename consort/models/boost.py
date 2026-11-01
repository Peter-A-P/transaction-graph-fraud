"""Gradient boosting for arms R and G: one fit, one set of stated hyperparameters.

The hyperparameters are fixed and stated, not tuned per arm, so that the difference between R
and G is the features and nothing else. Early stopping on validation AUPRC is the only thing
fitted to the validation partition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb

PARAMS: dict[str, object] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
}
MAX_ROUNDS = 2000
EARLY_STOP = 100


@dataclass(frozen=True)
class Fitted:
    booster: xgb.Booster
    columns: list[str]
    best_iteration: int

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        d = xgb.DMatrix(X[self.columns])
        out = self.booster.predict(d, iteration_range=(0, self.best_iteration + 1))
        return np.asarray(out, dtype=np.float64)


def fit(
    X_train: pd.DataFrame,
    y_train: npt.ArrayLike,
    X_val: pd.DataFrame,
    y_val: npt.ArrayLike,
    seed: int,
    nthread: int = 0,
) -> Fitted:
    columns = list(X_train.columns)
    dtrain = xgb.DMatrix(X_train, label=np.asarray(y_train))
    dval = xgb.DMatrix(X_val[columns], label=np.asarray(y_val))
    params = {**PARAMS, "seed": seed, "nthread": nthread}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=False,
    )
    return Fitted(booster, columns, int(booster.best_iteration))
