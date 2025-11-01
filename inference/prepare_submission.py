from pathlib import Path

import pandas as pd
from scipy.special import expit

def tranformed_transmittance2probability(x):
    pred_t = 100 * expit(x)
    if pred_t <= 70:
        return 1 - pred_t / 70
    else:
        return (100 - pred_t) / 30

if __name__ == "__main__":
    pred_transmittance_df = pd.read_csv(Path("output/pred_transmittance.csv"))
    pred_transmittance_df["T340"] = pred_transmittance_df["T340_transform"].map(tranformed_transmittance2probability)
    pred_transmittance_df["T450"] = pred_transmittance_df["T450_transform"].map(tranformed_transmittance2probability)
    
    pred_fluorescence_df = pd.read_csv(Path("output/pred_fluorescence.csv"))
    
    pd.DataFrame({
        "Transmittance(340)": pred_transmittance_df["T340"],
        "Transmittance(450)": pred_transmittance_df["T450"],
        "Fluorescence(340/480)": pred_fluorescence_df["F340450"],
        "Fluorescence(multiple)": pred_fluorescence_df["F480"],
    }).to_csv("submission.csv", index=False)
