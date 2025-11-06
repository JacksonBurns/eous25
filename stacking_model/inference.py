# load and run inference using the trained models
import sys
from datetime import datetime

import pandas as pd
import joblib
from pathlib import Path

from common import clean_smiles
from tqdm import tqdm

if __name__ == "__main__":
    try:
        model_dir = Path(sys.argv[1])
    except:
        print("Usage: python inference.py <model_directory>")
        exit(1)

    df = pd.read_csv(Path("../data/testing.csv"))

    test_smiles = df["clean_smiles"].to_list()

    out_data = {}

    targets = list(model_dir.glob("*"))
    pbar = tqdm(total=len(targets))
    for target in targets:
        if not target.is_dir():
            continue
        target_name = target.stem
        pbar.set_description(f"Predicting '{target_name}'")
        pipe = joblib.load(target / "final_model.joblib")
        pred = pipe.predict_proba(test_smiles)[:, 1].flatten()
        out_data[target_name] = pred
        pbar.update(1)
    pbar.close()

    # timestamped result file
    out_df = pd.DataFrame(out_data)
    out_df[["T340", "T450", "F340450", "F480"]].to_csv(  # correct order
        model_dir / f"test_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        index=False,
    )
