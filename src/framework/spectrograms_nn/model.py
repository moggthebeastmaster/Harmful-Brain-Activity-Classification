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

from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

root = Path(__file__).parents[1]
sys.path.append(str(root))

from src.framework.spectrograms_nn.config.efficient_net_config import EfficientNetConfig, TARGETS_COLUMNS
from src.framework.spectrograms_nn.classifier.efficient_net import EfficientNet
from src.framework.spectrograms_nn.data.spectrograms_dataset import SpectrogramsDataset
from src.framework.spectrograms_nn.data.spectrograms_eeg_dataset import SpectrogramsEEGDataset
from src.kaggle_score import kaggle_score


class SpectrogramsDataModule(LightningDataModule):
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset, config: EfficientNetConfig):
        super().__init__()
        self.train_dataset: Dataset = train_dataset
        self.val_dataset: Dataset = val_dataset
        self.config: EfficientNetConfig = config

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


class SpectrogramsModel():
    def __init__(self, config: EfficientNetConfig):
        self.config = config
        self.num_classes = len(TARGETS_COLUMNS)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # self.device =  torch.device("cpu")
        self.model = None
        self.initialize_model()

        if self.config.model_framework in ["efficientnet_b0"]:
            self.dataset = SpectrogramsDataset
        elif self.config.model_framework in ["eeg_efficientnet_b0"]:
            self.dataset = SpectrogramsEEGDataset
        else:
            raise NotImplementedError


    def initialize_model(self, pretrain=False):
        self.model = _LightningModel(config=self.config, pretrain=pretrain)
        self.model.to(self.device)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, eegs_dir: Path, spectrograms_dir: Path,
              output_dir: Path) -> dict:

        self.initialize_model(pretrain=True)

        train_dataset = SpectrogramsDataset(meta_df=train_df,
                                            spectrograms_dir=spectrograms_dir,
                                            config=self.config,
                                            with_label=True,
                                            train_mode=True)
        val_dataset = SpectrogramsDataset(meta_df=val_df,
                                          spectrograms_dir=spectrograms_dir,
                                          config=self.config,
                                          with_label=True,
                                          train_mode=False)
        data_module = SpectrogramsDataModule(train_dataset=train_dataset, val_dataset=val_dataset, config=self.config)

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
                          deterministic=True,
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

        # 評価

        label_df = pd.DataFrame(np.concatenate([eeg_ids, label], axis=1), columns=["eeg_id"] + TARGETS_COLUMNS)
        predicts_df = pd.DataFrame(np.concatenate([eeg_ids, predict_y], axis=1), columns=["eeg_id"] + TARGETS_COLUMNS)
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
        test_dataset = SpectrogramsDataset(meta_df=test_df,
                                           spectrograms_dir=spectrograms_dir,
                                           config=self.config,
                                           with_label=False,
                                           train_mode=False)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=self.config.batch_size,
                                     shuffle=False,
                                     num_workers=0,
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
        predicts_df = pd.DataFrame(np.concatenate([eeg_ids, predict_y], axis=1), columns=["eeg_id"] + TARGETS_COLUMNS)
        return predicts_df

    def save(self, file_path: Path):
        torch.save(self.model.egg_classifier.state_dict(), file_path)

    def load(self, file_path: Path):
        self.model.egg_classifier.load_state_dict(state_dict=torch.load(file_path, map_location=torch.device('cpu')))


class _LightningModel(LightningModule):
    def __init__(self, config: EfficientNetConfig, pretrain: bool):
        super().__init__()

        self.config = config
        if self.config.model_framework in ["efficientnet_b0", "eeg_efficientnet_b0"]:
            self.egg_classifier = EfficientNet(self.config, pretrain=pretrain)
        else:
            raise NotImplementedError

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
        images, labels, _, _ = batch
        predicts_logit = self(images)
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
        label_df = pd.DataFrame(np.concatenate([eeg_ids, labels], axis=1), columns=["eeg_id"] + TARGETS_COLUMNS)
        predicts_df = pd.DataFrame(np.concatenate([eeg_ids, predicts], axis=1), columns=["eeg_id"] + TARGETS_COLUMNS)
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
