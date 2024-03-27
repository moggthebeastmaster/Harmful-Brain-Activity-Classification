"""
dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
from src.framework.eeg1dgru.config import *
from src.utils.data import *

class Eeg1dGRUDataset(Dataset):
    def __init__(self, meta_df: pd.DataFrame, eegs_dir: Path,mode: str = "train"):
        self.mode = mode
        self.ch2idx = {ch:idx for idx, ch in enumerate(RAW_EEG_CHANNEL_LIST)}

        self.data_df = meta_df
        self.eegs_dir = eegs_dir


    def __len__(self):
        return len(self.data_df)
    

    def __getitem__(self, index: int):
        row = self.data_df.iloc[index]
        eeg_id = int(row["eeg_id"])
        raw_eeg = np.load(self.eegs_dir.joinpath(f"{eeg_id}.npy")).astype(np.float32)
        
        X = self.data_generation(raw_eeg)
        
        if self.mode != "test":
            votes = row[TARGETS].values
            y_prob = (votes / votes.sum()).astype(np.float32)
        else:
            y_prob = np.zeros((len(TARGETS)), dtype=np.float32)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y_prob, dtype=torch.float32), eeg_id
    

    def data_generation(self, raw_eeg: np.ndarray):
        X = np.zeros((EEG_STEP, len(RAW_EEG_CHANNEL_LIST)), dtype=np.float32)

        X[:, 0] = raw_eeg[:, self.ch2idx["Fp1"]] - raw_eeg[:, self.ch2idx["T3"]]
        X[:, 1] = raw_eeg[:, self.ch2idx["T3"]] - raw_eeg[:, self.ch2idx["O1"]]

        X[:, 2] = raw_eeg[:, self.ch2idx["Fp1"]] - raw_eeg[:, self.ch2idx["C3"]]
        X[:, 3] = raw_eeg[:, self.ch2idx["C3"]] - raw_eeg[:, self.ch2idx["O1"]]

        X[:, 4] = raw_eeg[:, self.ch2idx["Fp2"]] - raw_eeg[:, self.ch2idx["C4"]]
        X[:, 5] = raw_eeg[:, self.ch2idx["C4"]] - raw_eeg[:, self.ch2idx["O2"]]

        X[:, 6] = raw_eeg[:, self.ch2idx["Fp2"]] - raw_eeg[:, self.ch2idx["T4"]]
        X[:, 7] = raw_eeg[:, self.ch2idx["Fp2"]] - raw_eeg[:, self.ch2idx["O2"]]

        X = np.clip(X, -1024, 1024)
        X = np.nan_to_num(X, nan=0) / 32.0

        X = butter_lowpass_filter(X)

        return X


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    data_root = Path(os.environ["kaggle_data_root"]).joinpath("hms-harmful-brain-activity-classification", "test")
    cfg = Eeg1dGRUConfig()
    eeg1dgru_dataset = Eeg1dGRUDataset(data_root=data_root, mode="train")

    X, y = next(iter(eeg1dgru_dataset))

    X = X.numpy()

    time = np.arange(0, 50, 50/X.shape[0])
    for ch in range(X.shape[1]):
        plt.plot(time, X[:, ch] + ch*(X[:, ch].max() - X[:, ch].min()), label=f"feature_{ch+1}")
    
    plt.legend()
    plt.yticks([])
    plt.show()

    dataloader = DataLoader(eeg1dgru_dataset, batch_size=16, num_workers=1, pin_memory=True, drop_last=True)

    X, y = next(iter(dataloader))

    print(X.shape)








    