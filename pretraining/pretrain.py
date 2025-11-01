from pathlib import Path
from urllib.request import urlretrieve
from datetime import datetime

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.preprocessing import QuantileTransformer
from astartes import train_test_split
import pandas as pd
import numpy as np
import joblib

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import BondMessagePassing, RegressionFFN
from chemprop.models import MPNN
from chemprop.nn.agg import MeanAggregation

from random_masking_mse import RandomMaskingMSE

NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def train_one(outdir):
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    agg = MeanAggregation()
    ckpt_dir = Path().home() / ".chemprop"
    ckpt_dir.mkdir(exist_ok=True)
    mp_path = ckpt_dir / "chemeleon_mp.pt"
    chemeleon_mp = torch.load(mp_path, weights_only=True)
    mp = BondMessagePassing(**chemeleon_mp['hyper_parameters'])
    mp.load_state_dict(chemeleon_mp['state_dict'])

    train_df = pd.read_parquet(Path("../data/pretraining.parquet"))

    train_idxs, val_idxs = train_test_split(
        np.arange(train_df.shape[0]),
        train_size=0.95,
        test_size=0.05,
        random_state=42,
    )
    
    scaler = Path("scaler.joblib")
    if not scaler.exists():
        qt = QuantileTransformer(output_distribution="normal", random_state=42)
        qt.fit(train_df.iloc[train_idxs].values)
    else:
        qt = joblib.load(scaler)
    train_df.loc[:, :] = qt.transform(train_df.values)
        
    
    train_data = [
        MoleculeDatapoint.from_smi(smi, y)
        for smi, y in zip(
            train_df.iloc[train_idxs].index, train_df.values.astype(np.float32)
        )
    ]
    val_data = [
        MoleculeDatapoint.from_smi(smi, y)
        for smi, y in zip(
            train_df.iloc[val_idxs].index, train_df.values.astype(np.float32)
        )
    ]
    train_dataset = MoleculeDataset(train_data, featurizer)
    val_dataset = MoleculeDataset(val_data, featurizer)
    train_dataloader = build_dataloader(train_dataset, num_workers=1, batch_size=64)
    val_dataloader = build_dataloader(val_dataset, num_workers=1, shuffle=False, batch_size=64)
    fnn = RegressionFFN(
        n_tasks=train_df.shape[1],
        input_dim=mp.output_dim,
        hidden_dim=2_048,
        n_layers=0,
        dropout=0.0,
        activation="relu",
        # criterion=RandomMaskingMSE(),
    )

    model = MPNN(
        mp,
        agg,
        fnn,
        batch_norm=False,
        init_lr=1e-4,
        max_lr=1e-3,
        final_lr=1e-5,
        warmup_epochs=2,
        # metrics=[RandomMaskingMSE()]
    )
    tensorboard_logger = TensorBoardLogger(
        outdir,
        name="tensorboard_logs",
        default_hp_metric=False,
    )
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=False,
            patience=5,
        ),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            dirpath=outdir / "checkpoints",
        ),
    ]
    trainer = Trainer(
        max_epochs=30,
        logger=tensorboard_logger,
        log_every_n_steps=1,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    ckpt_path = trainer.checkpoint_callback.best_model_path
    print(f"Reloading best model from checkpoint file: {ckpt_path}")
    model = MPNN.load_from_checkpoint(ckpt_path)
    result = trainer.validate(model, val_dataloader)[0]["val_loss"]
    return result

if __name__ == "__main__":
    outdir = Path("output")
    outdir.mkdir(parents=True, exist_ok=True)
    # https://github.com/JacksonBurns/chemeleon/blob/237fa44d42fa503cecc095cf0aadf3a9eef52a95/chemeleon_fingerprint.py#L27C9-L39C55
    ckpt_dir = Path().home() / ".chemprop"
    ckpt_dir.mkdir(exist_ok=True)
    mp_path = ckpt_dir / "chemeleon_mp.pt"
    if not mp_path.exists():
        urlretrieve(
            r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
            mp_path,
        )
    train_one(outdir)
