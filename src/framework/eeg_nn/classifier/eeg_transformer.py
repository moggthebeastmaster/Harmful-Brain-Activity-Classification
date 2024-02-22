import os

import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import tqdm
from torch import nn
from pathlib import Path
import sys
import math

root = Path(__file__).parents[4]
sys.path.append(str(root))

from src.framework.eeg_nn.config import EEGNeuralNetConfig


class ResnetBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               stride=(1, stride), padding=(0, padding), bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               stride=(1, stride), padding=(0, padding), bias=False)

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.2, inplace=False)

        # 入力と出力のチャンネル数が異なる場合、x をダウンサンプリングする。
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1),
                                      stride=(1, stride), bias=False)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        identity = self.shortcut(x)

        out += identity
        return out


class TemporalEncoder(nn.Module):
    def __init__(self,
                 num_electrodes=20,
                 num_base_channels=24,
                 kernels=(3, 5, 7, 9)):
        super().__init__()
        self.kernels = kernels
        self.num_base_channels = num_base_channels
        self.num_channel = self.num_base_channels * len(self.kernels)
        self.num_electrodes = num_electrodes
        self.parallel_conv = nn.ModuleList()

        for i, kernel_size in enumerate(self.kernels):
            sep_conv = nn.Conv2d(in_channels=1, out_channels=self.num_base_channels, kernel_size=(1, kernel_size),
                                 stride=1, padding=(0, kernel_size // 2), padding_mode='replicate', bias=False)
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm2d(num_features=self.num_channel)
        self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(in_channels=self.num_channel, out_channels=self.num_channel, kernel_size=(1, 3),
                               stride=(1, 2), bias=False)
        self.block = self._make_resnet_layer(kernel_size=3, stride=1, padding=3 // 2)
        self.bn2 = nn.BatchNorm2d(num_features=self.num_channel)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(self.num_electrodes, 1))

    def _make_resnet_layer(self, kernel_size, stride, blocks=5, padding=0):
        layers = []
        for i in range(blocks):
            layers.append(
                ResnetBlock(in_channels=self.num_channel, out_channels=self.num_channel, kernel_size=kernel_size,
                            stride=stride, padding=padding))

        return nn.Sequential(*layers)

    def forward(self, x):
        out_sep = []

        for conv in self.parallel_conv:
            sep = conv(x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=1)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)

        out = out.squeeze(-1)
        return out


class ResnetBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               stride=(1, stride), padding=(0, padding), bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               stride=(1, stride), padding=(0, padding), bias=False)

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.2, inplace=False)

        # 入力と出力のチャンネル数が異なる場合、x をダウンサンプリングする。
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1),
                                      stride=(1, stride), bias=False)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        identity = self.shortcut(x)

        out += identity
        return out


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads=8, n_layers=4):
        super().__init__()
        self.pos_encoder = PositionalEncoding(in_channels)
        encoder_layers = TransformerEncoderLayer(in_channels, n_heads, dropout=0.5, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=n_layers)

    def forward(self, x):
        # [batch, channels, seq_len]
        x = self.pos_encoder(x)

        # [batch, channels, seq_len] -> [batch, seq_len, channels]
        x = x.permute(0, 2, 1)
        x = self.transformer_encoder(x)

        # [batch, seq_len, channels] -> [batch, channels, seq_len]
        x = x.permute(0, 2, 1)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, in_channels: int, max_len: int = 20):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(0).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, in_channels, 2) * (-math.log(10000.0) / in_channels)).unsqueeze(
            0).unsqueeze(2)
        # pe = torch.zeros(max_len, 1, d_model)
        pe = torch.zeros(1, in_channels, max_len)  # [batch, ch, len]
        # pe[:, 0, 0::2] = torch.sin(position * div_term)
        # pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe[0, 0::2] = torch.sin(position * div_term)
        pe[0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_encoder', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, in_channels, seq_len]``
        """
        return x + self.position_encoder[..., :x.size(2)]


class EEGTransformer(nn.Module):
    def __init__(self,
                 num_electrodes: int = 20,
                 num_base_channels=24,
                 num_classes: int = 6,
                 parallel_kernels=(3, 5, 7, 9),
                 ):
        super().__init__()

        self.temporal_encoder = TemporalEncoder(num_electrodes=num_electrodes,
                                                num_base_channels=num_base_channels,
                                                kernels=parallel_kernels,
                                                )
        self.spatial_encoder = SpatialTransformer(in_channels=int(num_base_channels) * len(parallel_kernels),
                                                  n_heads=num_base_channels,
                                                  n_layers=1)
        self.head = nn.Linear(in_features=num_base_channels*len(parallel_kernels)*num_electrodes, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): [batch, chunk_size, num_electrodes].

        Returns:
            torch.Tensor[batch, num_classes]
        '''

        x = x.permute(0, 2, 1).unsqueeze(dim=1)

        # [batch, 1, chunk_size, num_electrodes] -> [batch, ch, num_electrodes]
        x = self.temporal_encoder(x)

        # [batch, ch, num_electrodes] -> [batch, ch, num_electrodes]
        x = self.spatial_encoder(x)

        # [batch, ch*num_electrodes]
        x = x.flatten(start_dim=1)
        x = self.head(x)

        return x


if __name__ == '__main__':

    import pandas as pd

    config = EEGNeuralNetConfig(data_use_second=10)
    eegs_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_eegs")
    meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/train.csv"))
    meta_df = meta_df.iloc[:1000]

    from src.framework.eeg_nn.data.eeg_dataset import EEGDataset

    train_dataset = EEGDataset(meta_df, eegs_dir, config=config, with_label=True)
    print(len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2 ** 1, shuffle=True,
                                                   num_workers=os.cpu_count() // 2)

    model = EEGTransformer()
    model.to("cpu")
    model.train()

    for x, label, _, _ in tqdm.tqdm(train_dataloader):
        pred = model(x)

    print(pred)
