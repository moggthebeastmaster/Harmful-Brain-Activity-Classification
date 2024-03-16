import os

import pytorch_lightning as pl
from torchvision.models import efficientnet_b0
import torch
from torch import nn
import pandas as pd
from pathlib import Path
import numpy as np
import timm

from .config import External04Config
from .data import External04Dataset, TARGETS, CFG

import warnings

warnings.filterwarnings('ignore')


class External04Model:
    """
    https://www.kaggle.com/code/hideyukizushi/hms-blend-all-torch-publicmodel-simpleblend-lb-32
    Model3 のラップクラス
    """

    def __init__(self, config: External04Config):
        self.config = config
        self.model = EEGNet(
            kernels=CFG.kernels,
            in_channels=CFG.in_channels,
            fixed_kernel_size=CFG.fixed_kernel_size,
            num_classes=CFG.target_size,
            linear_layer_features=CFG.linear_layer_features,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_model(self):
        self.model = EEGNet(
            kernels=CFG.kernels,
            in_channels=CFG.in_channels,
            fixed_kernel_size=CFG.fixed_kernel_size,
            num_classes=CFG.target_size,
            linear_layer_features=CFG.linear_layer_features,
        )

    def train(self,
              train_df: pd.DataFrame,
              val_df: pd.DataFrame,
              eegs_dir: Path,
              spectrograms_dir: Path,
              output_dir: Path, early_stop: bool) -> dict:
        raise NotImplementedError

    def predict(self,
                test_df: pd.DataFrame,
                eegs_dir: Path,
                spectrograms_dir: Path) -> pd.DataFrame:
        self.model.to(self.device).eval()
        dataset = External04Dataset(test_df,
                                    eegs_dir=eegs_dir,
                                    spectrograms_dir=spectrograms_dir,
                                    config=self.config,
                                    )
        test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=32,
                                                  num_workers=os.cpu_count() // 2)
        softmax = nn.Softmax(dim=1)
        predict_y = []
        eeg_id_list = []
        with torch.inference_mode():
            for batch in test_loader:
                X = batch.pop("eeg").to(self.device)
                eeg_id = batch.pop("eeg_id")
                with torch.no_grad():
                    preds = self.model(X)
                predict_y.append(softmax(preds).to("cpu").detach().numpy())
                eeg_id_list.append(eeg_id.cpu().detach().numpy())
        predict_y = np.concatenate(predict_y, axis=0)
        eeg_ids = np.concatenate(eeg_id_list)[:, np.newaxis]

        predicts_df = pd.DataFrame(np.concatenate([eeg_ids, predict_y], axis=1), columns=["eeg_id"] + TARGETS)
        return predicts_df

    def save(self):
        raise NotImplementedError

    def load(self, file_path: Path):
        ckpt = torch.load(file_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(ckpt["model"])


class ResNet_1D_Block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            downsampling,
            dilation=1,
            groups=1,
            dropout=0.0,
    ):
        super(ResNet_1D_Block, self).__init__()

        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        # self.relu = nn.ReLU(inplace=False)
        # self.relu_1 = nn.PReLU()
        # self.relu_2 = nn.PReLU()
        self.relu_1 = nn.Hardswish()
        self.relu_2 = nn.Hardswish()

        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.maxpool = nn.MaxPool1d(
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=dilation,
        )
        self.downsampling = downsampling

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu_1(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.maxpool(out)
        identity = self.downsampling(x)

        out += identity
        return out


class EEGNet(nn.Module):
    def __init__(
            self,
            kernels,
            in_channels,
            fixed_kernel_size,
            num_classes,
            linear_layer_features,
            dilation=1,
            groups=1,
    ):
        super(EEGNet, self).__init__()
        self.kernels = kernels
        self.planes = 24
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels

        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.planes,
                kernel_size=(kernel_size),
                stride=1,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        # self.relu = nn.ReLU(inplace=False)
        # self.relu_1 = nn.ReLU()
        # self.relu_2 = nn.ReLU()
        self.relu_1 = nn.SiLU()
        self.relu_2 = nn.SiLU()

        self.conv1 = nn.Conv1d(
            in_channels=self.planes,
            out_channels=self.planes,
            kernel_size=fixed_kernel_size,
            stride=2,
            padding=2,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.block = self._make_resnet_layer(
            kernel_size=fixed_kernel_size,
            stride=1,
            dilation=dilation,
            groups=groups,
            padding=fixed_kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=6, stride=6, padding=2)

        self.rnn = nn.GRU(
            input_size=self.in_channels,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            # dropout=0.2,
        )

        self.fc = nn.Linear(in_features=linear_layer_features, out_features=num_classes)

    def _make_resnet_layer(
            self,
            kernel_size,
            stride,
            dilation=1,
            groups=1,
            blocks=9,
            padding=0,
            dropout=0.0,
    ):
        layers = []
        downsample = None
        base_width = self.planes

        for i in range(blocks):
            downsampling = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            )
            layers.append(
                ResNet_1D_Block(
                    in_channels=self.planes,
                    out_channels=self.planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    downsampling=downsampling,
                    dilation=dilation,
                    groups=groups,
                    dropout=dropout,
                )
            )
        return nn.Sequential(*layers)

    def extract_features(self, x):
        x = x.permute(0, 2, 1)
        out_sep = []

        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu_1(out)
        out = self.conv1(out)

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.avgpool(out)

        out = out.reshape(out.shape[0], -1)
        rnn_out, _ = self.rnn(x.permute(0, 2, 1))
        new_rnn_h = rnn_out[:, -1, :]  # <~~

        new_out = torch.cat([out, new_rnn_h], dim=1)
        return new_out

    def forward(self, x):
        new_out = self.extract_features(x)
        result = self.fc(new_out)
        return result
