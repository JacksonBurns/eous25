from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    pred_df = pd.read_csv(Path("output/pred.csv"))
    # prf_pred_df = pd.read_csv(Path("../prf/t450_pred.csv"))

    pd.DataFrame({
        "Transmittance(340)": pred_df["T340"],
        "Transmittance(450)": pred_df["T450"],
        "Fluorescence(340/480)": pred_df["F340450"],
        "Fluorescence(multiple)": pred_df["F480"],
    }).to_csv("submission.csv", index=False)
