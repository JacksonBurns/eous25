import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import joblib
import matplotlib
import optuna
from sklearn.metrics import auc
from astartes import train_val_test_split

matplotlib.use("Agg")

from common import (
    get_prf_pipe,
    PreviousModelTransformer,
)

# just a manual guess at the settings
_d = dict(stack_xgb=False, stack_knn=True, stack_ridge=False, stack_svr=False)

# from previous hpopt runs
KNOWN_PARAMS = {
    "F340450": _d,
    "T340": _d,
    "T450": _d,
    "F480": _d,
}

# these are in a specific order of which will be used to predict the others
TARGETS = [
    "F340450",
    "T340",
    "T450",
    "F480",
]
SMILES_COL = "clean_smiles"

TUNING_TRIALS = 64  # number of optuna trials for hyperparameter tuning


def define_by_run(trial):
    params = dict(
        morgan_radius=trial.suggest_categorical("morgan_radius", [2, 3, 4]),
        morgan_size=trial.suggest_categorical("morgan_size", [1024, 2048, 4096]),
        stack_chemprop=trial.suggest_categorical("stack_chemprop", [True, False]),
        stack_xgb=trial.suggest_categorical("stack_xgb", [True, False]),
        stack_knn=trial.suggest_categorical("stack_knn", [True, False]),
        stack_ridge=trial.suggest_categorical("stack_ridge", [True, False]),
        stack_svr=trial.suggest_categorical("stack_svr", [True, False]),
    )

    # Conditionally suggest hyperparameters for stacked models
    if params["stack_xgb"]:
        params["xgb_max_depth"] = trial.suggest_int("xgb_max_depth", 3, 10)
        params["xgb_learning_rate"] = trial.suggest_float("xgb_learning_rate", 1e-3, 0.3, log=True)
    else:
        # Provide a default value if not stacked, though common.py uses its own defaults
        # This is primarily to avoid Optuna errors if the parameter isn't defined.
        params["xgb_max_depth"] = 6
        params["xgb_learning_rate"] = 0.1

    if params["stack_knn"]:
        params["knn_n_neighbors"] = trial.suggest_int("knn_n_neighbors", 3, 30)
    else:
        params["knn_n_neighbors"] = 8

    if params["stack_svr"]:
        params["svr_C"] = trial.suggest_float("svr_C", 1e-1, 100.0, log=True)
        params["svr_gamma"] = trial.suggest_categorical("svr_gamma", ["scale", "auto"])
    else:
        params["svr_C"] = 1.0
        params["svr_gamma"] = "scale"
    
    if params["stack_chemprop"]:
        params["chemeleon_ffn_hidden_dim"] = trial.suggest_int("chemeleon_ffn_hidden_dim", 400, 3000, step=200)
        params["chemeleon_ffn_num_layers"] = trial.suggest_int("chemeleon_ffn_num_layers", 0, 3)

    return params


def train_one(
    df,
    train_idxs,
    val_idxs,
    target,
    subdir,
    extra_transformers,
    write_output=False,
    **kwargs,
):
    pipe = get_prf_pipe(
        extra_transformers=extra_transformers,
        **kwargs,
    )
    pipe.fit(df[SMILES_COL].iloc[train_idxs], df[target].iloc[train_idxs])
    val_pred = pipe.predict_proba(df[SMILES_COL].iloc[val_idxs])[:, 1].flatten()
    data = {"smiles": df[SMILES_COL].iloc[val_idxs].reset_index(drop=True)}
    data[f"true_{target}"] = df[target].iloc[val_idxs].reset_index(drop=True)
    data[f"pred_{target}"] = val_pred
    val_df = pd.DataFrame(data)
    if write_output:
        val_df.to_csv(Path(subdir) / "val_predictions.csv", index=False)
        joblib.dump(pipe, subdir / "validation_model.joblib")
    return auc(val_df[f"true_{target}"], val_df[f"pred_{target}"])


if __name__ == "__main__":
    try:
        outdir = Path(sys.argv[1])
    except:
        print("Usage: python training.py <output_directory>")
        exit(1)

    # timestamped output directory
    outdir /= datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir.mkdir(exist_ok=True)

    # get the data
    _df = pd.read_csv(Path("../data/training.csv"))

    # going to fit one model per target, re-using previous models outputs on subsequent models
    previous_model_paths = []
    for _target in TARGETS:
        df = _df.copy()
        target = _target

        # just in case
        df[target] = df[target].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[target])

        subdir = outdir / target.replace(" ", "_")
        subdir.mkdir(parents=True, exist_ok=True)

        extra_transformers = []
        if previous_model_paths:
            extra_transformers += [
                (
                    "previous_models",
                    PreviousModelTransformer(previous_model_paths, outdir / "cache.db"),
                )
            ]

        if _target in KNOWN_PARAMS:
            # mock the outcome of the study with known params
            study = SimpleNamespace()
            study.best_params = KNOWN_PARAMS[_target]
        else:
            *_, train_idxs, val_idxs, _ = train_val_test_split(
                np.arange(df.shape[0]),
                train_size=0.10,
                val_size=0.01,
                test_size=1.0 - (0.10 + 0.01),
                return_indices=True,
            )
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: train_one(
                    df,
                    train_idxs,
                    val_idxs,
                    target,
                    subdir,
                    extra_transformers,
                    write_output=False,
                    **define_by_run(trial),
                ),
                n_trials=TUNING_TRIALS,
            )
            with open(subdir / f"optuna_study_{target.replace(' ', '_')}.txt", "w") as f:
                f.write(f"Best hyperparameters for target {target}: {study.best_params}\n")
            study.trials_dataframe().to_csv(subdir / f"optuna_study_results_{target.replace(' ', '_')}.csv")

            # for reference, train and save the validation model with the optimal settings
            train_one(
                df,
                train_idxs,
                val_idxs,
                target,
                subdir,
                extra_transformers,
                write_output=True,
                **study.best_params,
            )

        # using the optimal settings, train a model on the entire dataset for actual submission
        pipe = get_prf_pipe(extra_transformers=extra_transformers, random_seed=42, **study.best_params)
        pipe.fit(df[SMILES_COL], df[target])
        outmodel = subdir / "final_model.joblib"
        joblib.dump(pipe, outmodel)
        previous_model_paths.append(outmodel)
