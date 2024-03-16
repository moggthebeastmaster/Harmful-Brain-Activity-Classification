import os

import pytorch_lightning as pl
from torchvision.models import efficientnet_b0
import torch
from torch import nn
import pandas as pd
from pathlib import Path
import numpy as np

from .config import External02Config
from .data import External02Dataset, TARGETS

import warnings
warnings.filterwarnings('ignore')
class External02Model:
    """
    https://www.kaggle.com/code/hideyukizushi/hms-blend-all-torch-publicmodel-simpleblend-lb-32
    Model2 のラップクラス
    """
    def __init__(self, config:External02Config):
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_model(self):
        self.model = None


    def train(self,
              train_df: pd.DataFrame,
              val_df: pd.DataFrame,
              eegs_dir: Path,
              spectrograms_dir: Path,
              output_dir: Path,early_stop:bool) -> dict:
        raise NotImplementedError

    def predict(self,
                test_df: pd.DataFrame,
                eegs_dir: Path,
                spectrograms_dir: Path) -> pd.DataFrame:

        self.model.to(self.device).eval()
        dataset = External02Dataset(test_df,
                                    eegs_dir=eegs_dir,
                                    spectrograms_dir=spectrograms_dir,
                                    config=self.config,
                                    )
        test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=16, num_workers=os.cpu_count()//2)
        predict_y = []
        eeg_id_list = []
        with torch.inference_mode():
            for test_batch in test_loader:
                x, eeg_id = test_batch
                pred = torch.softmax(self.model(x.to(self.device)), dim=1).cpu().numpy()
                predict_y.append(pred)
                eeg_id_list.append(eeg_id.cpu().detach().numpy())
        predict_y = np.concatenate(predict_y, axis=0)
        eeg_ids = np.concatenate(eeg_id_list)[:, np.newaxis]

        predicts_df = pd.DataFrame(np.concatenate([eeg_ids, predict_y], axis=1), columns=["eeg_id"] + TARGETS)
        return predicts_df


    def save(self):
        raise NotImplementedError


    def load(self, file_path: Path):
        self.model = torch.load(file_path)
