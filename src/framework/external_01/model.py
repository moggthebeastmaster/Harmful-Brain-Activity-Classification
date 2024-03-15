import pytorch_lightning as pl
from torchvision.models import efficientnet_b0
import torch
from torch import nn
import pandas as pd
from pathlib import Path
import numpy as np

from .config import External01Config
from .data import External01Dataset, TARGETS

class External01Model:
    """
    https://www.kaggle.com/code/hideyukizushi/hms-blend-all-torch-publicmodel-simpleblend-lb-32
    Model1 のラップクラス
    """
    def __init__(self, config:External01Config):
        self.config = config
        self.model = EEGEffnetB0()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_model(self):
        self.model = EEGEffnetB0()


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
        dataset = External01Dataset(test_df,
                                    eegs_dir=eegs_dir,
                                    spectrograms_dir=spectrograms_dir,
                                    config=self.config,
                                    )
        test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=64)
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
        self.model = EEGEffnetB0.load_from_checkpoint(file_path)



class EEGEffnetB0(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.base_model = efficientnet_b0()
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, 6, dtype=torch.float32)
        self.prob_out = nn.Softmax()

    def forward(self, x):
        x1 = [x[:, :, :, i:i + 1] for i in range(4)]
        x1 = torch.concat(x1, dim=1)
        x2 = [x[:, :, :, i + 4:i + 5] for i in range(4)]
        x2 = torch.concat(x2, dim=1)

        x = torch.concat([x1, x2], dim=2)
        x = torch.concat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)

        out = self.base_model(x)
        return out
