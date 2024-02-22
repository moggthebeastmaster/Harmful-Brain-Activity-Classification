import torch
from torch import nn
import tqdm
from torch import nn
from pathlib import Path
import sys

root = Path(__file__).parents[4]
sys.path.append(str(root))

from src.framework.eeg_nn.config import EEGNeuralNetConfig


class WaveBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_layers: int,
                 kernel_size: int,
                 ):

        super().__init__()
        self.n_layers = n_layers
        dilation_rates = [2 ** i for i in range(n_layers)]

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, kernel_size),
                               stride=(1, 1),
                               padding="same",
                               )

        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.wave_conv_tanh_list = nn.ModuleList()
        self.wave_conv_sigm_list = nn.ModuleList()
        self.wave_conf_1x1_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()


        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        for dilation_rate in dilation_rates:
            self.wave_conv_sigm_list.append(nn.Conv2d(in_channels=out_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=(1, kernel_size),
                                                      stride=(1, 1),
                                                      padding="same",
                                                      dilation=dilation_rate,
                                                      bias=False,
                                                      ))
            self.wave_conv_tanh_list.append(nn.Conv2d(in_channels=out_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=(1, kernel_size),
                                                      stride=(1, 1),
                                                      padding="same",
                                                      dilation=dilation_rate,
                                                      bias=False,
                                                      )
                                            )
            self.wave_conf_1x1_list.append(nn.Conv2d(in_channels=out_channels,
                                                     out_channels=out_channels,
                                                     kernel_size=(1, 1),
                                                     stride=(1, 1),
                                                     padding="same",
                                                     bias=False,
                                                     )
                                           )
            self.bn_list.append(torch.nn.BatchNorm2d(num_features=out_channels))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        for n in range(self.n_layers):
            res_x = x
            x_tanh = self.wave_conv_tanh_list[n](x)
            x_tanh = self.tanh(x_tanh)
            x_sigmoid = self.wave_conv_sigm_list[n](x)
            x_sigmoid = self.wave_conv_sigm_list[n](x_sigmoid)
            x = torch.multiply(x_tanh, x_sigmoid)
            x = self.wave_conf_1x1_list[n](x)
            x = self.relu(x)
            x = self.bn_list[n](x)
            x = torch.add(res_x, x)

        return x


class WaveNet(nn.Module):
    def __init__(self,
                 num_electrodes: int = 20,
                 num_base_channels:int =8,
                 num_classes: int = 6,
                 kernel_size: int = 3,
                 drop_out = 0.5):
        super().__init__()
        """
        WaveNet モデル
        試してみてわかったこと
        ・ WaveBlockが4層だと、途中でnanが出る場合があり->層が深すぎて勾配消失が起きている？
        ・ 最初のWaveBlock のn_layers が小さいと性能は低い->n_layer = 12 だと dillation=2**12=4096となり、かなりでかいがこれでいいらしい
        """

        nc = num_base_channels

        self.block1_1 = WaveBlock(1, nc * (2 ** 0), kernel_size, 12)
        self.block1_2 = WaveBlock(nc * (2 ** 0), nc * (2 ** 1), kernel_size, 6)
        # self.block1_3 = WaveBlock(nc * (2 ** 1), nc * (2 ** 2), kernel_size, 4)
        # self.block1_4 = WaveBlock(nc * (2 ** 2), nc * (2 ** 3), kernel_size, 2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(num_electrodes, 1))
        self.head = nn.Sequential(nn.Linear(in_features=(nc * (2 ** 2-1)) * num_electrodes, out_features=nc * (2**3), bias=False),
                                  nn.Dropout(drop_out, inplace=False),
                                  nn.ReLU(inplace=False),
                                  nn.Linear(in_features=nc * (2**3), out_features=num_classes, bias=False),
                                  )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): [batch, chunk_size, num_electrodes].

        Returns:
            torch.Tensor[batch, num_classes]
        '''

        x = x.permute(0, 2, 1).unsqueeze(dim=1)
        x_1 = self.block1_1(x)
        x_2 = self.block1_2(x_1)
        # x_3 = self.block1_3(x_2)
        # x_4 = self.block1_4(x_3)

        x = torch.concatenate([x_1, x_2], dim=1)
        del x_1, x_2,

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.head(x)



        return x


if __name__ == '__main__':

    import pandas as pd

    config = EEGNeuralNetConfig()
    eegs_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_eegs")
    meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/train.csv"))

    from src.framework.eeg_nn.data.eeg_dataset import EEGDataset, COLUMN_NAMES

    train_dataset = EEGDataset(meta_df, eegs_dir, config=config, with_label=True)
    print(len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2 ** 1, shuffle=True)

    model = WaveNet(num_electrodes=len(COLUMN_NAMES))
    model.to("cpu")
    model.train()

    for x, label, _, _ in tqdm.tqdm(train_dataloader):
        pred = model(x)

    print(pred)
