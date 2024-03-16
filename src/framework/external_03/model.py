import os

import pytorch_lightning as pl
from torchvision.models import efficientnet_b0
import torch
from torch import nn
import pandas as pd
from pathlib import Path
import numpy as np
import timm

from .config import External03Config
from .data import External03Dataset, TARGETS

import warnings
warnings.filterwarnings('ignore')

class External03Model:
    """
    https://www.kaggle.com/code/hideyukizushi/hms-blend-all-torch-publicmodel-simpleblend-lb-32
    Model3 のラップクラス
    """
    def __init__(self, config:External03Config):
        self.config = config
        self.model_resnet = timm.create_model('resnet34d', pretrained=False, num_classes=6, in_chans=1)
        self.model_effnet_b0 = timm.create_model('efficientnet_b0', pretrained=False, num_classes=6, in_chans=1)
        self.model_effnet_b1 = timm.create_model('efficientnet_b1', pretrained=False, num_classes=6, in_chans=1)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_model(self):
        self.model_resnet = timm.create_model('resnet34d', pretrained=False, num_classes=6, in_chans=1)
        self.model_effnet_b0 = timm.create_model('efficientnet_b0', pretrained=False, num_classes=6, in_chans=1)
        self.model_effnet_b1 = timm.create_model('efficientnet_b1', pretrained=False, num_classes=6, in_chans=1)


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

        weight_resnet34d = 0.25
        weight_effnetb0 = 0.42
        weight_effnetb1 = 0.33

        self.model_resnet.to(self.device).eval()
        self.model_effnet_b0.to(self.device).eval()
        self.model_effnet_b1.to(self.device).eval()
        dataset = External03Dataset(test_df,
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
                pred_resnet34d = torch.softmax(self.model_resnet(x.to(self.device)), dim=1).cpu().numpy()
                pred_effnetb0 = torch.softmax(self.model_effnet_b0(x.to(self.device)), dim=1).cpu().numpy()
                pred_effnetb1 = torch.softmax(self.model_effnet_b1(x.to(self.device)), dim=1).cpu().numpy()
                weighted_pred = weight_resnet34d * pred_resnet34d + \
                                weight_effnetb0 * pred_effnetb0 + \
                                weight_effnetb1 * pred_effnetb1
                predict_y.append(weighted_pred)
                eeg_id_list.append(eeg_id.cpu().detach().numpy())
        predict_y = np.concatenate(predict_y, axis=0)
        eeg_ids = np.concatenate(eeg_id_list)[:, np.newaxis]

        predicts_df = pd.DataFrame(np.concatenate([eeg_ids, predict_y], axis=1), columns=["eeg_id"] + TARGETS)
        return predicts_df


    def save(self):
        raise NotImplementedError


    def load(self, file_path: Path):
        self.model_resnet.load_state_dict(torch.load(file_path / "resnet34d.pth", map_location=torch.device('cpu')))
        self.model_effnet_b0.load_state_dict(torch.load(file_path / "efficientnet_b0.pth", map_location=torch.device('cpu')))
        self.model_effnet_b1.load_state_dict(torch.load(file_path / "efficientnet_b1.pth", map_location=torch.device('cpu')))