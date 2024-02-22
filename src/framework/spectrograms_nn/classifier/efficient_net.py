import os

import torchvision.models
import tqdm
from pathlib import Path
import sys
import torchvision.models as models


root = Path(__file__).parents[4]
sys.path.append(str(root))

from src.framework.spectrograms_nn.config.efficient_net_config import EfficientNetConfig, TARGETS_COLUMNS


import torch
from torch import nn

class EfficientNet(nn.Module):
    def __init__(self, config:EfficientNetConfig, pretrain:bool):
        super().__init__()
        self.config = config
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrain else None
        self.efficient_net = torchvision.models.efficientnet_b0(weights=weights)

        head = nn.Sequential(nn.Dropout(self.config.drop_out),
                             nn.Linear(self.efficient_net.classifier[1].in_features, len(TARGETS_COLUMNS),
                                       dtype=torch.float32)

        )


        self.efficient_net.classifier[1] = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): [batch, weights, heights].

        Returns:
            torch.Tensor[batch, num_classes]
        '''
        x = self.efficient_net(x)
        return x


if __name__ == '__main__':

    import pandas as pd

    config = EfficientNetConfig()
    spectrograms_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_spectrograms")
    meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/train.csv"))

    from src.framework.spectrograms_nn.data.spectrograms_dataset import SpectrogramsDataset

    train_dataset = SpectrogramsDataset(meta_df, spectrograms_dir, config=config, with_label=True)
    print(len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2 ** 1, shuffle=True, num_workers=os.cpu_count()//2)

    model = EfficientNet(config=config)
    model.to("cpu")
    model.train()

    for x, label, _, _ in tqdm.tqdm(train_dataloader):
        pred = model(x)

    print(pred)
