from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from astartes import train_test_split
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer, MorganCountFeaturizer
from chemprop.models import MPNN, load_model
from chemprop.nn import BinaryClassificationFFN
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from rdkit.Chem import MolFromSmiles

NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def train_one(pretrained_ckpt, outdir):
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    ms = 2048
    mcf = MorganCountFeaturizer(radius=3, length=ms)

    train_df = pd.read_csv(Path("../data/training.csv"))
    train_df.set_index("clean_smiles", drop=True, inplace=True)
    train_df = train_df[["T340", "T450", "F340450", "F480"]]  # order matters for later inference!
    
    # downsample the abundant negatives
    negative_mask = (train_df.values.sum(axis=1) == 0)
    postitive_count = train_df.shape[0] - negative_mask.astype(int).sum()
    print(f"There are {train_df.shape[0]} samples, of which {postitive_count} have at least one positive")
    train_df = pd.concat((train_df[negative_mask].sample(n=postitive_count, random_state=42), train_df[~negative_mask]), axis=0)
    print(f"Downsampled to {train_df.shape[0]} samples")

    train_idxs, val_idxs = train_test_split(
        np.arange(train_df.shape[0]),
        train_size=0.90,
        test_size=0.10,
        random_state=42,
    )

    train_data = [
        MoleculeDatapoint.from_smi(smi, y, x_d=mcf(MolFromSmiles(smi))) for smi, y in zip(train_df.iloc[train_idxs].index, train_df.iloc[train_idxs].values.astype(np.float32))
    ]
    val_data = [
        MoleculeDatapoint.from_smi(smi, y, x_d=mcf(MolFromSmiles(smi))) for smi, y in zip(train_df.iloc[val_idxs].index, train_df.iloc[val_idxs].values.astype(np.float32))
    ]
    train_dataset = MoleculeDataset(train_data, featurizer)
    val_dataset = MoleculeDataset(val_data, featurizer)
    train_dataloader = build_dataloader(train_dataset, num_workers=1, batch_size=64)
    val_dataloader = build_dataloader(val_dataset, num_workers=1, shuffle=False, batch_size=64)

    pretrained_model = load_model(pretrained_ckpt)
    mp = pretrained_model.message_passing
    agg = pretrained_model.agg
    fnn = BinaryClassificationFFN(
        n_tasks=4,
        input_dim=mp.output_dim + ms,
        hidden_dim=2_048,
        n_layers=1,
        dropout=0.0,
        activation="relu",
    )

    model = MPNN(
        mp,
        agg,
        fnn,
        batch_norm=False,
        init_lr=1e-5,
        max_lr=1e-4,
        final_lr=1e-5,
        warmup_epochs=2,
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
        max_epochs=50,
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
    import sys

    try:
        ckpt_path = Path(sys.argv[1])
    except:
        print("USAGE: python train.py <path/to/pretrained.ckpt>")
        exit(1)
    outdir = Path("output")
    outdir.mkdir(parents=True, exist_ok=True)
    train_one(ckpt_path, outdir)
