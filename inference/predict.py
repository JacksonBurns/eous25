from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer, MorganCountFeaturizer
from chemprop.models import load_model
from lightning import Trainer
from rdkit.Chem import MolFromSmiles

NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def infer(pretrained_ckpt):
    featurizer = SimpleMoleculeMolGraphFeaturizer()

    df = pd.read_csv(Path("../data/testing.csv"))
    df.set_index("clean_smiles", drop=True, inplace=True)
    
    mcf = MorganCountFeaturizer(3, 2_048)

    data = [MoleculeDatapoint.from_smi(smi, x_d=mcf(MolFromSmiles(smi))) for smi in df.index]
    dataset = MoleculeDataset(data, featurizer)
    dataloader = build_dataloader(dataset, num_workers=1, batch_size=64)

    model = load_model(pretrained_ckpt)

    trainer = Trainer(
        logger=False,
        enable_checkpointing=False,
    )
    return torch.cat(trainer.predict(model, dataloader), dim=0).numpy(force=True)


if __name__ == "__main__":
    import sys

    try:
        ckpt_path = Path(sys.argv[1])
    except:
        print("USAGE: python predict.py <path/to/trained.ckpt>")
        exit(1)
    preds = infer(ckpt_path)
    pd.DataFrame(
        {
            "Transmittance(340)": preds[:, 0],
            "Transmittance(450)": preds[:, 1],
            "Fluorescence(340/480)": preds[:, 2],
            "Fluorescence(multiple)": preds[:, 3],
        }
    ).to_csv("submission.csv", index=False)
