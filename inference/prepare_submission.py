from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    pred_transmittance_df = pd.read_csv(Path("output/pred_transmittance.csv"))
    pred_fluorescence_df = pd.read_csv(Path("output/pred_fluorescence.csv"))

    pd.DataFrame({
        "Transmittance(340)": pred_transmittance_df["T340"],
        "Transmittance(450)": pred_transmittance_df["T450"],
        "Fluorescence(340/480)": pred_fluorescence_df["F340450"],
        "Fluorescence(multiple)": pred_fluorescence_df["F480"],
    }).to_csv("submission.csv", index=False)
