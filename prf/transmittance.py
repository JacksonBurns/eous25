# adapted from here under the terms of the MIT license:
# https://github.com/JacksonBurns/chemeleon/blob/51e028a77a3cb4de87ff1e75a7ed18d4372606f4/models/rf_morgan_physchem/evaluate.py
from typing import Literal

from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.descriptors import MolecularDescriptorTransformer
from scikit_mol.fingerprints import MorganFingerprintTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier


def get_prf_pipe(
    morgan_radius: int = 4,
    n_estimators: int = 500,
    random_seed: int = 42,
    stack_xgb: bool = True,
    stack_knn: bool = True,
    stack_elasticnet: bool = True,
    final_estimator: Literal["elasticnet", "hgb", "rf"] = "hgb",
):
    estimators = []
    # always include RF
    estimators.append(
        (
            "rf",
            RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed, n_jobs=-1),
        )
    )

    if stack_xgb:
        estimators.append(
            (
                "xgb",
                XGBClassifier(n_estimators=n_estimators, random_state=random_seed, n_jobs=-1),
            )
        )

    if stack_knn:
        estimators.append(("knn", KNeighborsClassifier(n_neighbors=8)))

    if stack_elasticnet:
        estimators.append(("elasticnet", ElasticNet(random_state=random_seed)))

    if final_estimator == "elasticnet":
        final_estimator_model = ElasticNet()
    elif final_estimator == "hgb":
        final_estimator_model = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=500,
            random_state=random_seed,
        )
    elif final_estimator == "rf":
        final_estimator_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown final_estimator: {final_estimator}")

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator_model,
        passthrough=False,
        cv=5,
    )

    return Pipeline(
        [
            ("smiles2mol", SmilesToMolTransformer()),
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "morgan",
                            MorganFingerprintTransformer(
                                fpSize=2048,
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
                ),
            ),
            ("variance_filter", VarianceThreshold(0.0)),
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ],
        verbose=True,
    )

if __name__ == "__main__":
    from pathlib import Path
    
    import joblib
    import pandas as pd
    
    # T450 is the weakest prediction from the deep model, so we will just train a prf model for that
    
    pretrained = Path("prf.joblib")
    
    if not pretrained.exists():
        train_df = pd.read_csv(Path("../data/training.csv"))
        # downsample slightly to reduce imbalance
        positives = train_df[train_df["T450"] == 1]
        negatives = train_df[train_df["T450"] != 1]
        negative_sampled = negatives.sample(frac=0.1, random_state=42)
        train_df = pd.concat([positives, negative_sampled], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        pipe = get_prf_pipe(
            morgan_radius=4,
            n_estimators=700,
            stack_xgb=True,
            stack_knn=False,
            stack_elasticnet=False,
            final_estimator="hgb",
        )
        pipe.fit(train_df["clean_smiles"], train_df["T450"])
        joblib.dump(pipe, pretrained)
    pipe = joblib.load(pretrained)
    
    test_df = pd.read_csv(Path("../data/testing.csv"))
    
    pred = pipe.predict_proba(test_df["clean_smiles"])
    pd.DataFrame(dict(
        T450=pred[:, 1],
    )).to_csv("t450_pred.csv", index=False)
