import numpy as np
from torch.utils.data import Dataset
import sys
from pathlib import Path
import pandas as pd
import torchvision.transforms as transforms
import torch

root = Path(__file__).parents[4]
sys.path.append(str(root))

from src.framework.external_02.config import External02Config


USE_KAGGLE_SPECTROGRAMS = True
USE_EEG_SPECTROGRAMS = True
TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

TARS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
TARS2 = {x: y for y, x in TARS.items()}

class External02Dataset(Dataset):
    def __init__(self,
                 meta_df: pd.DataFrame,
                 eegs_dir:Path,
                 spectrograms_dir: Path,
                 config: External02Config,
                 with_label=False,
                 train_mode=False,
                 ):

        assert not with_label
        assert not train_mode

        self.spec_indexes = meta_df.spectrogram_id
        self.eeg_indexes = meta_df.eeg_id

        self.eeg_dir = eegs_dir
        self.spectrograms_dir = spectrograms_dir
        self.config = config

        self.eps=1e-6
        self.image_transform = transforms.Resize((512, 512))

    def __len__(self):
        return len(self.eeg_indexes)

    def __getitem__(self, index):
        spec_id = self.spec_indexes[index]
        spec_path = self.spectrograms_dir / f"{spec_id}.parquet"
        data = pd.read_parquet(spec_path)
        data = data.fillna(-1).values[:, 1:].T
        if data.shape[0]!=400:
            data = data.T
        data = data[:, 0:300]  # (400,300)
        data = np.clip(data, np.exp(-6), np.exp(10))
        data = np.log(data)

        data_mean = data.mean(axis=(0, 1))
        data_std = data.std(axis=(0, 1))
        data = (data - data_mean) / (data_std + self.eps)
        data_tensor = torch.unsqueeze(torch.Tensor(data), dim=0)
        data = self.image_transform(data_tensor)
        eeg_id =self.eeg_indexes[index]
        return data, eeg_id

