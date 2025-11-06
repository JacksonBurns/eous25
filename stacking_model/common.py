# adapted from here under the terms of the MIT license:
# https://github.com/JacksonBurns/chemeleon/blob/51e028a77a3cb4de87ff1e75a7ed18d4372606f4/models/rf_morgan_physchem/evaluate.py
from pathlib import Path
import sqlite3
from typing import Literal

import numpy as np
import joblib
from rdkit.Chem import MolToSmiles
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.descriptors import MolecularDescriptorTransformer
from scikit_mol.fingerprints import MorganFingerprintTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.svm import SVR
from sklearn.base import clone

from argparse import ArgumentParser, Namespace
from datetime import datetime
import logging
import os
from os import PathLike
from pathlib import Path
from typing import List, Literal, Optional, Sequence

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, StochasticWeightAveraging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
import torch
from torch.utils.data import DataLoader

from chemprop.cli.common import add_common_args, find_models
from chemprop.cli.train import add_train_args, build_model, normalize_inputs
from chemprop.cli.utils.parsing import make_datapoints, make_dataset, parse_csv
from chemprop.data.collate import (
    collate_batch,
    collate_mol_atom_bond_batch,
    collate_multicomponent,
)
from chemprop.data.datasets import MolAtomBondDataset, MulticomponentDataset
from chemprop.featurizers.molgraph.reaction import RxnMode
from chemprop.models import MPNN, MulticomponentMPNN, utils
from chemprop.nn.transforms import UnscaleTransform
from chemprop.data.datapoints import make_mol, MoleculeDatapoint
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np


NOW = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
CHEMPROP_TRAIN_DIR = Path(os.getenv("CHEMPROP_TRAIN_DIR", "chemprop_training"))


def add_train_defaults(args: Namespace) -> Namespace:
    parser = ArgumentParser()
    parser = add_common_args(parser)
    parser = add_train_args(parser)
    defaults = parser.parse_args([])
    for k, v in vars(defaults).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


class ChemeleonClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 64,
        output_dir: Optional[PathLike] = CHEMPROP_TRAIN_DIR / "sklearn_output" / NOW,
        ffn_hidden_dim: int = 2_048,
        ffn_num_layers: int = 1,
        accelerator: str = "auto",
        devices: str | int | Sequence[int] = "auto",
        epochs: int = 20,
    ):
        args = Namespace(
            num_workers=num_workers,
            batch_size=batch_size,
            output_dir=output_dir,
            ffn_hidden_dim=ffn_hidden_dim,
            ffn_num_layers=ffn_num_layers,
            accelerator=accelerator,
            devices=devices,
            epochs=epochs,
            from_foundation="chemeleon",
            task="classification",
        )
        self.args = add_train_defaults(args)
        self.model = None
        for name, value in locals().items():
            if name not in {"self", "args"}:
                setattr(self, name, value)
    def _build_dps(self, X: np.ndarray, Y: Optional[np.ndarray]):
        if Y is None:
            return [MoleculeDatapoint(mol=mol) for mol in X.flatten()]
        return [MoleculeDatapoint(mol=mol, y=[int(target)]) for mol, target in zip(X.flatten(), Y)]

    def __sklearn_is_fitted__(self):
        return hasattr(self, "classes_")

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        datapoints = self._build_dps(X, y)
        train_set = make_dataset(datapoints)

        if self.model is None:
            # no output normalization for classification
            self.model = build_model(self.args, train_set, None, [None] * 4)

        train_loader = DataLoader(
            train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=collate_batch,
        )

        trainer = Trainer(
            accelerator=self.args.accelerator,
            devices=self.args.devices,
            max_epochs=self.args.epochs,
            callbacks=[StochasticWeightAveraging(0.001, annealing_epochs=4, swa_epoch_start=0.6)],
            logger=False,
            enable_checkpointing=False,
            strategy="ddp_spawn",
        )
        trainer.fit(self.model, train_dataloaders=train_loader)
        return self

    def predict_proba(self, X):
        datapoints = self._build_dps(X, None)
        test_set = make_dataset(datapoints)
        dl = DataLoader(
            test_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=collate_batch,
        )

        eval_trainer = Trainer(
            accelerator=self.args.accelerator,
            devices=1,
            enable_progress_bar=False,
            logger=False,
        )

        preds = eval_trainer.predict(self.model, dataloaders=dl, return_predictions=True)
        probs = torch.cat(preds, dim=0).softmax(dim=1)  # ✅ convert logits → probabilities (chemprop might already do this)
        return probs.numpy(force=True)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def clean_smiles(
    smiles: str,
    remove_hs: bool = True,
    strip_stereochem: bool = False,
    strip_salts: bool = True,
) -> str:
    """Applies preprocessing to SMILES strings, seeking the 'parent' SMILES

    Note that this is different from simply _neutralizing_ the input SMILES - we attempt to get the parent molecule, analogous to a molecular skeleton.
    This is adapted in part from https://rdkit.org/docs/Cookbook.html#neutralizing-molecules

    Args:
        smiles (str): input SMILES
        remove_hs (bool, optional): Removes hydrogens. Defaults to True.
        strip_stereochem (bool, optional): Remove R/S and cis/trans stereochemistry. Defaults to False.
        strip_salts (bool, optional): Remove salt ions. Defaults to True.

    Returns:
        str: cleaned SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Could not parse SMILES {smiles}"
        if remove_hs:
            mol = Chem.RemoveHs(mol)
        if strip_stereochem:
            Chem.RemoveStereochemistry(mol)
        if strip_salts:
            remover = SaltRemover()  # use default saltremover
            mol = remover.StripMol(mol)  # strip salts

        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        out_smi = Chem.MolToSmiles(mol, kekuleSmiles=True)  # this also canonicalizes the input
        assert len(out_smi) > 0, f"Could not convert molecule to SMILES {smiles}"
        return out_smi
    except Exception as e:
        print(f"Failed to clean SMILES {smiles} due to {e}")
        return None


def get_prf_pipe(
    morgan_radius: int = 2,
    morgan_size: int = 2048,
    n_estimators: int = 500,
    random_seed: int = 42,
    extra_transformers: Optional[List] = None,
    stack_chemprop: bool = True,
    stack_xgb: bool = True,
    stack_knn: bool = True,
    stack_ridge: bool = True,
    stack_svr: bool = True,
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.1,
    knn_n_neighbors: int = 8,
    svr_C: float = 1.0,
    svr_gamma: str = "scale",
    chemeleon_ffn_hidden_dim: int = 2_048,
    chemeleon_ffn_num_layers: int = 1,
):
    if extra_transformers is None:
        extra_transformers = []

    # base feature pipeline (we will clone this for each base learner)
    base_feature_pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "morgan",
                            MorganFingerprintTransformer(
                                fpSize=morgan_size,
                                radius=morgan_radius,
                                useCounts=True,
                                n_jobs=-1,
                            ),
                        ),
                        (
                            "physchem",
                            MolecularDescriptorTransformer(
                                desc_list=[desc for desc in MolecularDescriptorTransformer().available_descriptors if desc != "Ipc"],
                                n_jobs=-1,
                            ),
                        ),
                    ]
                    + extra_transformers
                ),
            ),
            ("variance_filter", VarianceThreshold(0.0)),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    estimators = []

    # helper to create a full pipeline (cloned feature extractor + Classifier)
    def make_base_pipeline(name: str, estimator):
        fp_clone = clone(base_feature_pipeline)
        return (name, Pipeline([("feat", fp_clone), (name + "_est", estimator)]))

    # always include RF
    estimators.append(
        make_base_pipeline(
            "rf",
            RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed, n_jobs=-1),
        )
    )

    if stack_xgb:
        estimators.append(
            make_base_pipeline(
                "xgb",
                XGBClassifier(n_estimators=n_estimators, random_state=random_seed, n_jobs=-1, max_depth=xgb_max_depth, learning_rate=xgb_learning_rate),
            )
        )

    if stack_knn:
        estimators.append(make_base_pipeline("knn", KNeighborsClassifier(n_neighbors=knn_n_neighbors)))

    if stack_ridge:
        estimators.append(make_base_pipeline("ridge", RidgeClassifier(random_state=random_seed)))

    if stack_svr:
        estimators.append(make_base_pipeline("svr", make_pipeline(StandardScaler(), SVR(C=svr_C, gamma=svr_gamma))))

    if stack_chemprop:
        # note - no feature generator! Chemprop handles this internally
        estimators.append(("chemeleon", ChemeleonClassifier(ffn_hidden_dim=chemeleon_ffn_hidden_dim, ffn_num_layers=chemeleon_ffn_num_layers)))

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        passthrough=False,
        n_jobs=1,  # avoid parallel training of chemprop
        cv=5,  # default
    )

    pipe = Pipeline(
        [
            ("smiles2mol", SmilesToMolTransformer()),
            ("Classifier", model),
        ],
        verbose=True,
    )

    return pipe


class PreviousModelTransformer:
    def __init__(self, model_paths: list[Path], cache_db: Path = Path("model_cache.sqlite")):
        self.model_paths = model_paths
        self.cache_db = cache_db
        self._ensure_schema()

    def _ensure_schema(self):
        """Create cache table for classification if it doesn't exist."""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    model_path TEXT,
                    smiles TEXT,
                    prob_neg REAL,
                    prob_pos REAL,
                    PRIMARY KEY (model_path, smiles)
                )
                """
            )
            conn.commit()

    def _fetch_cached_predictions(self, conn, model_path, smiles_list):
        """Retrieve cached probability pairs for given model and SMILES list."""
        if not smiles_list:
            return {}
        placeholders = ",".join("?" * len(smiles_list))
        query = f"""
            SELECT smiles, prob_neg, prob_pos
            FROM predictions
            WHERE model_path = ? AND smiles IN ({placeholders})
        """
        cur = conn.execute(query, (str(model_path), *smiles_list))
        return {row[0]: (row[1], row[2]) for row in cur.fetchall()}

    def _insert_predictions(self, conn, model_path, smiles, probs):
        """Insert new (prob_neg, prob_pos) predictions into the cache."""
        conn.executemany(
            """
            INSERT OR REPLACE INTO predictions (model_path, smiles, prob_neg, prob_pos)
            VALUES (?, ?, ?, ?)
            """,
            [(str(model_path), s, float(p[0]), float(p[1])) for s, p in zip(smiles, probs)],
        )
        conn.commit()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        smis = [MolToSmiles(mol[0]) for mol in X]
        preds = []

        with sqlite3.connect(self.cache_db) as conn:
            for model_path in self.model_paths:
                # 1. Fetch cached
                cached = self._fetch_cached_predictions(conn, model_path, smis)
                missing = [s for s in smis if s not in cached]

                # 2. Compute missing using predict_proba
                new_probs = []
                if missing:
                    model = joblib.load(model_path)
                    # Convert SMILES back to molecules or features as your model expects
                    X_missing = missing  # adapt if model requires feature conversion
                    new_probs = model.predict_proba(X_missing)
                    self._insert_predictions(conn, model_path, missing, new_probs)
                    del model

                # 3. Combine cached + new
                all_probs = np.zeros((len(smis), 2))
                for i, s in enumerate(smis):
                    if s in cached:
                        all_probs[i] = cached[s]
                    else:
                        idx = missing.index(s)
                        all_probs[i] = new_probs[idx]
                preds.append(all_probs)

        # Return shape: (n_samples, n_models * 2)
        return np.concatenate(preds, axis=1)

def parity_plot(
    truth: np.ndarray,
    prediction: np.ndarray,
    title: str = "",
    quantity: str = "",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    style: Literal["hexbin", "scatter"] = "scatter",
) -> None:
    """Create a scatter parity plot with an inset pie chart."""
    if xlim is None:
        xlim = (min(truth.min(), prediction.min()), max(truth.max(), prediction.max()))
    if ylim is None:
        ylim = xlim

    x_label = "True"
    y_label = "Predicted"
    if quantity:
        x_label += f" {quantity}"
        y_label += f" {quantity}"

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    ax: Axes  # type hint

    if style == "hexbin":
        hb = ax.hexbin(
            truth,
            prediction,
            gridsize=80,
            cmap="viridis",
            mincnt=1,
        )
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Number of compounds")
    elif style == "scatter":
        ax.scatter(
            truth,
            prediction,
            s=10,
            alpha=0.15 if truth.shape[0] > 1_000 else 0.5,
            color="C0",  # Default Matplotlib blue
        )
    else:
        raise ValueError(f"Unknown style: {style}")

    mae = round(mean_absolute_error(truth, prediction), 2)

    # 1:1 line
    ax.plot(xlim, xlim, "r", linewidth=1)
    # ±mae lines
    ax.plot(xlim, (np.array(xlim) + mae), "r--", linewidth=0.5)
    ax.plot(xlim, (np.array(xlim) - mae), "r--", linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, which="major", axis="both")
    ax.set_axisbelow(True)

    # Text box with R² and MSE
    r = pearsonr(truth, prediction)[0]
    textstr = (
        f"$\\bf{{R^2}}:$ {r**2:.3f}\n"
        f"$\\bf{{r}}:$ {r:.3f}\n"
        f"$\\bf{{MAE}}:$ {mae:.2f}\n"
        f"$\\bf{{MSE}}:$ {mean_squared_error(truth, prediction):.2f}\n"
        f"$\\bf{{RMSE}}:$ {root_mean_squared_error(truth, prediction):.2f}\n"
        f"$\\bf{{Support}}:$ {truth.shape[0]:d}"
    )
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Inset pie chart: fraction within ±mae
    frac_within_mae = np.mean(np.abs(truth - prediction) < mae)
    sizes = [1 - frac_within_mae, frac_within_mae]
    ax_inset = ax.inset_axes([0.75, 0.025, 0.25, 0.25], transform=ax.transAxes)
    ax_inset.pie(
        sizes,
        colors=["#ae2b27", "#4073b2"],
        startangle=360 * (frac_within_mae - 0.5) / 2,
        wedgeprops={"edgecolor": "black"},
        autopct="%1.f%%",
        textprops=dict(color="w"),
    )
    ax_inset.axis("equal")
    ax_inset.set_title(f"$\\bf{{±{mae:.2f}}}$ {quantity}", fontsize=10)

    plt.tight_layout()
    return fig
