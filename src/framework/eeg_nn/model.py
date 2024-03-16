import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pathlib import Path
import torch
import dataclasses
import transformers
import sys
import numpy as np
import tqdm
import shutil
import os

from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

root = Path(__file__).parents[1]
sys.path.append(str(root))

from src.framework.eeg_nn.config import EEGNeuralNetConfig, EEGResnetGRUConfig
from src.framework.eeg_nn.classifier.resnet_gru import ResnetGRU
from src.framework.eeg_nn.classifier.wave_net import WaveNet
from src.kaggle_score import kaggle_score


class EEGDataModule(LightningDataModule):
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset, config: EEGNeuralNetConfig):
        super().__init__()
        self.train_dataset: Dataset = train_dataset
        self.val_dataset: Dataset = val_dataset
        self.config: EEGNeuralNetConfig = config

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


class EEGNeuralNetModel():
    def __init__(self, config: EEGNeuralNetConfig):
        self.config = config

        if self.config.model_framework == "WaveNet":
            from src.framework.eeg_nn.data.eeg_dataset import TARGETS_COLUMNS, FREQUENCY, EEGDataset, COLUMN_NAMES
            self.dataset_class = EEGDataset
            self.target_columns = TARGETS_COLUMNS
        elif self.config.model_framework == "ResnetGRU":
            from src.framework.eeg_nn.data.eeg_resnetgru_dataset import TARGETS_COLUMNS, FREQUENCY, EEGResnetGRUDataset, COLUMN_NAMES
            self.dataset_class = EEGResnetGRUDataset
            self.target_columns = TARGETS_COLUMNS

        self.num_classes = len(self.target_columns)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        self.model = LightningModel(config=self.config)
        self.model.to(self.device)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, eegs_dir: Path, spectrograms_dir: Path,
              output_dir: Path) -> dict:

        self.initialize_model()

        train_dataset = self.dataset_class(meta_df=train_df, eegs_dir=eegs_dir, config=self.config, with_label=True,
                                   train_mode=True)
        val_dataset = self.dataset_class(meta_df=val_df, eegs_dir=eegs_dir, config=self.config, with_label=True,
                                 train_mode=False)
        data_module = EEGDataModule(train_dataset=train_dataset, val_dataset=val_dataset, config=self.config)

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
            input_data, labels, eeg_id, offset_num = batch
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
            input_data, labels, eeg_id, offset_num = batch
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
        torch.save(self.model.egg_classifier.state_dict(), file_path)

    def load(self, file_path: Path):
        self.model.egg_classifier.load_state_dict(state_dict=torch.load(file_path, map_location=torch.device('cpu')))


class LightningModel(LightningModule):
    def __init__(self, config: EEGNeuralNetConfig):
        super().__init__()


        self.config = config
        if self.config.model_framework == "ResnetGRU":
            from src.framework.eeg_nn.data.eeg_resnetgru_dataset import TARGETS_COLUMNS, COLUMN_NAMES
            self.egg_classifier = ResnetGRU(drop_out=self.config.drop_out)
            self.target_columns = TARGETS_COLUMNS
        elif self.config.model_framework == "WaveNet":
            from src.framework.eeg_nn.data.eeg_dataset import TARGETS_COLUMNS,  COLUMN_NAMES
            self.egg_classifier = WaveNet(num_electrodes=len(COLUMN_NAMES), num_base_channels=config.num_base_channels,
                                          drop_out=config.drop_out)
            self.target_columns = TARGETS_COLUMNS

        self.kl_loss_function = KLDivLossWithLogits()

        self.save_hyperparameters(dataclasses.asdict(config))

        #
        self.validation_step_eeg_ids = []
        self.validation_step_predicts = []
        self.validation_step_labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # softmaxをかける前のlogit の状態を返す
        return self.egg_classifier(x)

    def _loss_func(self, predicts_logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        kl_loss = self.kl_loss_function(predicts_logit, label)
        return kl_loss

    def training_step(self, batch, batch_idx):
        input_data, labels, _, _ = batch
        predicts_logit = self(input_data)
        loss = self._loss_func(predicts_logit, labels)
        loss = loss.mean()
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("learning_rate", self.lr_schedulers().get_last_lr()[0], prog_bar=False, logger=True, on_step=False,
                 on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input_data, labels, eeg_id, offset_num = batch
        with torch.no_grad():
            predicts_logit = self(input_data)

        self.validation_step_eeg_ids.append(eeg_id.cpu().detach().numpy())
        self.validation_step_predicts.append(torch.softmax(predicts_logit, dim=1).cpu().detach().numpy())
        self.validation_step_labels.append(labels.cpu().detach().numpy())

        loss = self._loss_func(predicts_logit, labels)
        loss = loss.mean()
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {
            "loss": loss.detach().cpu(),
        }

    def on_validation_epoch_end(self) -> None:
        if self.global_step == 0:
            return
        eeg_ids = np.concatenate(self.validation_step_eeg_ids, axis=0)[:, np.newaxis]
        labels = np.concatenate(self.validation_step_labels, axis=0)
        predicts = np.concatenate(self.validation_step_predicts, axis=0)
        label_df = pd.DataFrame(np.concatenate([eeg_ids, labels], axis=1), columns=["eeg_id"] + self.target_columns)
        predicts_df = pd.DataFrame(np.concatenate([eeg_ids, predicts], axis=1), columns=["eeg_id"] + self.target_columns)
        score = kaggle_score(label_df, predicts_df, row_id_column_name="eeg_id")
        self.log("val_score", score, prog_bar=False, logger=True, on_step=False, on_epoch=True)

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


class KLDivLossWithLogits(torch.nn.KLDivLoss):
    """

    """

    def __init__(self):
        super().__init__(reduction='batchmean')

    def forward(self, y, t):
        pred = torch.nn.functional.log_softmax(y, dim=1)
        loss = super().forward(pred, t)
        return loss


if __name__ == '__main__':
    #
    loss = KLDivLossWithLogits()
    label = torch.Tensor([[0., 1., 0], [1., 0., 0], [1., 0, 0.], [0., 0.5, 0.5]])
    target = torch.Tensor([[0.1, 0.9, 0., ], [0.9, 0., 0.1], [0.8, 0., 0.2], [0., 0.5, 0.5]])

    z = loss(target, label)
    z_ = loss(label, target)
    zz = loss(label, label)

    print(z)
