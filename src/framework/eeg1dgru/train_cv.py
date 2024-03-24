"""
train
"""
import pandas as pd

from pathlib import Path
from sklearn.model_selection import GroupKFold

import torch
from torch.utils.data import Subset

from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer

from src.framework.eeg1dgru.config import Eeg1dGRUConfig
from src.framework.eeg1dgru.data import Eeg1dGRUDataset
from src.framework.eeg1dgru.model import EEGLightningModel, EEGDataModule


def cv_main(data_root: Path, output_dir: Path, n_cv: int=5):
    config = Eeg1dGRUConfig()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    dataset = Eeg1dGRUDataset(data_root=data_root, mode="train")

    gkf = GroupKFold(n_splits=n_cv)
    patient_ids = dataset.data_df["patient_id"].to_list()
    
    for fold, (train_index, valid_index) in enumerate(gkf.split(range(len(dataset)), groups=patient_ids)):
        print("="*50)
        print(f"Fold: {fold + 1}")
        model = EEGLightningModel(config)

        train_dataset = Subset(dataset, train_index)
        valid_dataset = Subset(dataset, valid_index)

        data_module = EEGDataModule(train_dataset, valid_dataset, config)

        callbacks = [TQDMProgressBar()]

        trainer = Trainer(max_epochs=config.max_epoch,
                          callbacks=callbacks,
                          logger=CSVLogger(save_dir=str(output_dir)),
                          # log_every_n_steps=1,
                          precision="16-mixed",
                          check_val_every_n_epoch=1,
                          accumulate_grad_batches=config.accumulate_grad_batches,
                          enable_checkpointing=False,
                          )
        
        trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    import os
    data_root = Path(os.environ["kaggle_data_root"]).joinpath("hms-harmful-brain-activity-classification", "test")
    
    cv_main(data_root=data_root, output_dir=Path("./outputs/cv_test"))


