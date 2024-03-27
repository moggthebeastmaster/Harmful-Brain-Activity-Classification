"""
model
"""
import dataclasses
import transformers
import pandas as pd
import tqdm
import shutil
import numpy as np
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from src.framework.eeg1dgru.config import Eeg1dGRUConfig
from src.framework.eeg1dgru.data import Eeg1dGRUDataset
from src.kaggle_score import kaggle_score

class ResNet_1D_Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsampling, drop_out: float = 0.):
        super(ResNet_1D_Block, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=drop_out, inplace=False)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsampling = downsampling

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.maxpool(out)
        identity = self.downsampling(x)

        out += identity
        return out


class ResNet1dEncoder(nn.Module):
    def __init__(self, cfg:Eeg1dGRUConfig):
        super(ResNet1dEncoder, self).__init__()

        self.cfg = cfg
        self.kernels = cfg.kernels
        self.planes = cfg.planes
        self.parallel_conv = nn.ModuleList()
        self.in_channels = cfg.in_channels
        out_features = cfg.resnet_out_features
        self.drop_out = cfg.drop_out

        fixed_kernel_size = cfg.fixed_kernel_size

        # kernel size ごとの 1dconv
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.planes, kernel_size=(kernel_size),
                               stride=1, padding=0, bias=False,)
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=self.planes, out_channels=self.planes, kernel_size=fixed_kernel_size,
                               stride=2, padding=2, bias=False)
        self.block = self._make_resnet_layer(kernel_size=fixed_kernel_size, stride=1, padding=fixed_kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=6, stride=6, padding=2)
        self.linear = nn.Linear(in_features=168, out_features=out_features)
        self.last_dropout = nn.Dropout(p=self.drop_out)


    def _make_resnet_layer(self, kernel_size, stride, blocks=9, padding=0):
        layers = []
        downsample = None
        base_width = self.planes

        for i in range(blocks):
            downsampling = nn.Sequential(
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                )
            layers.append(ResNet_1D_Block(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsampling=downsampling, drop_out=self.drop_out))

        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor)->torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): (B, L, C) B:batch, L: time length, C:channel

        Returns:
            torch.Tensor: _description_
        """
        x = x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        out_sep = []

        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2) # [(B, C, L), ...] -> (B, C, L*#kernels)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)  

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)  
        
        out = out.reshape(out.shape[0], -1)
        out = self.linear(out)
        out = self.last_dropout(out)
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ConvTransEncoder(nn.Module):
    def __init__(self, cfg: Eeg1dGRUConfig):
        super().__init__()
        self.cfg = cfg
        self.kernels = cfg.kernels
        self.in_channels = cfg.in_channels
        self.planes = cfg.planes
        d_model = 128

        self.parallel_conv = nn.ModuleList()
        # kernel size ごとの 1dconv
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.planes, kernel_size=(kernel_size),
                               stride=1, padding="same", bias=False,)
            self.parallel_conv.append(sep_conv)
        self.conv1 = nn.Conv1d(in_channels=self.planes*len(self.kernels), out_channels=d_model, kernel_size=self.kernels[-1], stride=self.kernels[-1], padding=0, bias=False)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        self.rnn = nn.GRU(input_size=d_model, hidden_size=128, num_layers=2, bidirectional=True)
        

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)

        out_sep = []

        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=1) # [(B, C, L), ...] -> (B, plane*#kernels, L)
        out = self.conv1(out)
        out = out.permute(0, 2, 1)  # (B, plane*#kernels, L) -> (B, L, plane*#kernels)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)

        rnn_out, _ = self.rnn(out)
        rnn_h = rnn_out[:, -1, :]
        return rnn_h


class GRUEncoder(nn.Module):
    def __init__(self, cfg: Eeg1dGRUConfig):
        super(GRUEncoder, self).__init__()
        self.cfg = cfg
        self.in_channels = cfg.in_channels

        self.rnn = nn.GRU(input_size=self.in_channels, hidden_size=128, num_layers=1, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.rnn(x)    #(B, L, C)
        new_rnn_h = rnn_out[:, -1, :]  # (B, 256)

        return new_rnn_h


class EegNet(nn.Module):

    def __init__(self, cfg: Eeg1dGRUConfig):
        super(EegNet, self).__init__()
        self.cfg = cfg
        
        num_classes = self.cfg.num_classes
        self.res_encoder = ResNet1dEncoder(cfg=cfg)
        self.gru_encoder = GRUEncoder(cfg=cfg)

        in_features = cfg.resnet_out_features + cfg.gru_out_features*2
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes)
        

    def forward(self, x):
        res_out = self.res_encoder(x)
        gru_out = self.gru_encoder(x)

        out = torch.cat([res_out, gru_out], dim=1)
        result = self.fc(out)  

        return result
    
class EegTransNet(nn.Module):

    def __init__(self, cfg: Eeg1dGRUConfig):
        super(EegTransNet, self).__init__()
        self.cfg = cfg
        
        num_classes = self.cfg.num_classes
        self.res_encoder = ResNet1dEncoder(cfg=cfg)
        self.gru_encoder = ConvTransEncoder(cfg=cfg)

        in_features = cfg.resnet_out_features + cfg.gru_out_features*2
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes)
        

    def forward(self, x):
        res_out = self.res_encoder(x)
        gru_out = self.gru_encoder(x)

        out = torch.cat([res_out, gru_out], dim=1)
        result = self.fc(out)  

        return result


class EegDataModule(LightningDataModule):
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset, config: Eeg1dGRUConfig):
        super().__init__()
        self.train_dataset: Dataset = train_dataset
        self.val_dataset: Dataset = val_dataset
        self.config: Eeg1dGRUConfig = config


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_worker,
            persistent_workers=self.config.num_worker > 1,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_worker,
            persistent_workers=self.config.num_worker > 1,
            pin_memory=True,
        )


class Eeg1dGRUModel():
    def __init__(self, config: Eeg1dGRUConfig):
        self.config = config
        self.target_columns = config.target_columns

        self.dataset_class = Eeg1dGRUDataset
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = None
        self.initialize_model()


    def initialize_model(self):
        self.model = EegLightningModel(self.config)
        self.model.to(self.device)

    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, eegs_dir: Path, spectrograms_dir: Path,
              output_dir: Path) -> dict:
        self.initialize_model()

        train_dataset = self.dataset_class(meta_df=train_df, eegs_dir=eegs_dir)
        val_dataset = self.dataset_class(meta_df=val_df, eegs_dir=eegs_dir)

        data_module = EegDataModule(train_dataset=train_dataset, val_dataset=val_dataset, config=self.config)

        callbacks = [TQDMProgressBar()]
        if self.config.early_stop:
            early_stop_callback = EarlyStopping(monitor="val_score", min_delta=0.01, patience=1, verbose=False,
                                                mode="min")
            callbacks.append(early_stop_callback)

        trainer = Trainer(max_epochs=self.config.max_epoch,
                          callbacks=callbacks,
                          logger=CSVLogger(save_dir=str(output_dir)),
                          # log_every_n_steps=1,
                          precision="16-mixed",
                          check_val_every_n_epoch=1,
                          accumulate_grad_batches=self.config.accumulate_grad_batches,
                          enable_checkpointing=False,
                          )

        trainer.fit(self.model, datamodule=data_module)

        self.model.eval()
        self.model.to(self.device)
        predict_y = []
        eeg_id_list = []
        labels_list = []
        val_dataloader = data_module.val_dataloader()
        for batch in tqdm.tqdm(val_dataloader, desc="predict val dataset"):
            input_data, labels, eeg_id = batch
            with torch.no_grad():
                predicts_logit = self.model(input_data.to(self.device))
                predict_y.append(torch.softmax(predicts_logit, dim=1).cpu().detach().numpy())
                eeg_id_list.append(eeg_id.cpu().detach().numpy())
                labels_list.append(labels.cpu().detach().numpy())
        predict_y = np.concatenate(predict_y, axis=0)
        eeg_ids = np.concatenate(eeg_id_list)[:, np.newaxis]
        label = np.concatenate(labels_list)

        label_df = pd.DataFrame(np.concatenate([eeg_ids, label], axis=1), columns=["eeg_id"] + self.target_columns)
        predicts_df = pd.DataFrame(np.concatenate([eeg_ids, predict_y], axis=1), columns=["eeg_id"] + self.target_columns)
        score = kaggle_score(label_df.copy(), predicts_df.copy(), "eeg_id")
        score_df = pd.DataFrame([score], index=["kaggle_score"])

        # 保存処理
        output_dir.mkdir(exist_ok=True, parents=True)
        self.save(output_dir.joinpath("model.pt"))
        score_df.to_csv(output_dir.joinpath("score.csv"))
        label_df.to_csv(output_dir.joinpath("label.csv"))
        predicts_df.to_csv(output_dir.joinpath("predicts.csv"))
        shutil.copyfile(Path(self.model.trainer.log_dir) / "hparams.yaml", output_dir / "hparams.yaml")

        return {"kaggle_score": score}


    def predict(self, test_df: pd.DataFrame, eegs_dir: Path, spectrograms_dir: Path) -> pd.DataFrame:
        test_dataset = self.dataset_class(meta_df=test_df, eegs_dir=eegs_dir, config=self.config, with_label=False,
                                  train_mode=False)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=self.config.batch_size,
                                     shuffle=False,
                                     num_workers=os.cpu_count()//2,
                                     )

        harf = False
        float_type = torch.float16 if harf else torch.float32
        if harf:
            self.model.half()
        self.model.eval()
        self.model.to(self.device)
        predict_y = []
        eeg_id_list = []
        for batch in tqdm.tqdm(test_dataloader, desc="predict val dataset"):
            input_data, labels, eeg_id = batch
            with torch.no_grad():
                predicts_logit = self.model(input_data.to(self.device).to(float_type))
                predict_y.append(torch.softmax(predicts_logit, dim=1).cpu().detach().numpy())
                eeg_id_list.append(eeg_id.cpu().detach().numpy())
        predict_y = np.concatenate(predict_y, axis=0)
        eeg_ids = np.concatenate(eeg_id_list)[:, np.newaxis]

        # 評価
        predicts_df = pd.DataFrame(np.concatenate([eeg_ids, predict_y], axis=1), columns=["eeg_id"] + self.target_columns)

        return predicts_df


    def save(self, file_path: Path):
        torch.save(self.model.eeg_classifier.state_dict(), file_path)


    def load(self, file_path: Path):
        self.model.eeg_classifier.load_state_dict(state_dict=torch.load(file_path, map_location=torch.device('cpu')))



class EegLightningModel(LightningModule):
    def __init__(self, config: Eeg1dGRUConfig):
        super().__init__()

        self.config = config
        
        if config.model_framework == "resnet_1d_gru":
            self.eeg_classifier = EegNet(config)
        elif config.model_framework == "resnet_1d_convtrans":
            self.eeg_classifier = EegTransNet(config)
        else:
            raise ValueError()
        
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

        self.save_hyperparameters(dataclasses.asdict(self.config))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.eeg_classifier(x)
        return logits
    

    def _loss_func(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.kl_div_loss(F.log_softmax(logits, dim=1), targets)
        return loss
    
    def training_step(self, batch, batch_idx) -> dict[str, torch.Tensor]:
        X, y, _ = batch
        X, y = self.mixup(X, y)
        logits = self(X)
        loss = self._loss_func(logits, y)
        loss = loss.mean()

        self.log("train loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=False, logger=True, on_step=False, on_epoch=True)

        return {"loss": loss}
    

    def validation_step(self, batch, batch_idx) -> dict[str, torch.Tensor]:
        X, y, eeg_id = batch
        with torch.no_grad():
            logits = self(X)
        loss = self._loss_func(logits, y)
        loss = loss.mean()

        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {"loss": loss.detach().cpu()}


    def _get_total_steps(self) -> int:
        if not hasattr(self, "_total_steps"):
            train_loader = self.trainer.datamodule.train_dataloader()
            accum = max(1, self.trainer.num_devices) * self.trainer.accumulate_grad_batches
            self._total_steps = len(train_loader) // accum * self.trainer.max_epochs
            del train_loader
        return self._total_steps


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.config.learning_rate,
                                      weight_decay=self.config.weight_decay)
        total_steps = self._get_total_steps()
        warmup_steps = round(total_steps * self.config.warmup_steps_ratio)
        print(f"lr warmup step: {warmup_steps} / {total_steps}")
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        # scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, warmup_steps, total_steps, num_cycles=2)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def mixup(self, X: torch.Tensor, y: torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        """mini batch内でmixupを行う

        Args:
            X (torch.Tensor): features
            y (torch.Tensor): targets

        Returns:
            tuple[torch.Tensor, torch.Tensor]: mixuped (X, y)
        """


        rand_idx = torch.randperm(X.shape[0])
        rand_X = X[rand_idx]
        rand_y = y[rand_idx]

        p_1 = torch.rand(size=(X.shape[0],)).to(X.device)
        mask = torch.full(size=(X.shape[0],), fill_value=self.config.mixup_rate)
        mask = torch.bernoulli(mask).to(X.device)

        p_1 = p_1 * mask
        p_2 = 1 - p_1

        X = X * p_1[:, None, None] + rand_X * p_2[:, None, None]
        y = y * p_1[:, None] + rand_y * p_2[:, None]

        return X, y


if __name__ == "__main__":
    from pathlib import Path
    import os
    from src.framework.eeg1dgru.data import Eeg1dGRUDataset
    from torch.utils.data import DataLoader

    cfg = Eeg1dGRUConfig()
    model = ConvTransEncoder(cfg)

    data_root = Path(os.environ["kaggle_data_root"]).joinpath("hms-harmful-brain-activity-classification", "test")
    cfg = Eeg1dGRUConfig()
    meta_df = pd.read_csv(data_root.joinpath("train.csv"))
    eegs_dir = data_root.joinpath("raw_eeg")
    eeg1dgru_dataset = Eeg1dGRUDataset(meta_df=meta_df, eegs_dir=eegs_dir, mode="train")

    dataloader = DataLoader(eeg1dgru_dataset, batch_size=16, num_workers=1, pin_memory=True, drop_last=True)

    X, y, _ = next(iter(dataloader))

    out = model(X)

    print(out.shape)