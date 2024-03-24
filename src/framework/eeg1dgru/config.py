"""
config
"""
import dataclasses
import yaml
from pathlib import Path
from typing import Optional
from src.utils.data import *

@dataclasses.dataclass
class ConfigBase():
    def save(self, path: Path):
        with open(path, mode="w") as f:
            yaml.safe_dump(dataclasses.asdict(self), f)


    @classmethod
    def load(cls, path: Path):
        with open(path, mode="r") as f:
            obj = yaml.safe_load(f)

        return cls(**obj)


@dataclasses.dataclass
class Eeg1dGRUConfig(ConfigBase):
    model_framework: str = "resnet_1d_gru"
    target_columns = TARGETS
    num_classes: int = len(TARGETS)
    fixed_kernel_size: int = 5
    in_channels: int = len(RAW_EEG_CHANNEL_LIST)
    kernels: list[int] = dataclasses.field(default_factory=lambda: [3, 5, 7, 9])
    planes: int = 24
    resnet_out_features: int = 128
    gru_out_features: int = 128

    batch_size: int = 16
    num_worker: int = 2
    max_epoch: int = 5
    accumulate_grad_batches: int = 2 ** 0
    mixup_rate: float = 0.2
    learning_rate: float = 1e-4
    weight_decay: float = 0.5
    warmup_steps_ratio: float = 0.1
    early_stop: bool = False

if __name__ == "__main__":
    data_cfg = Eeg1dGRUConfig()

    data_cfg.save("test.yaml")
    data_cfg = Eeg1dGRUConfig.load("test.yaml")

    data_cfg
