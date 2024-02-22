from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

TARGETS_COLUMNS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

@dataclass
class EfficientNetConfig:
    model_framework: str = "efficientnet_b0"
    data_use_second: int = 600
    batch_size: int = 2 ** 3
    accumulate_grad_batches: int = 2 ** 0  # この回数だけロスを蓄積してパラメータを更新する。
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    warmup_steps_ratio: float = 0.1
    max_epoch: int = 10
    num_worker: int = 0
    early_stop: bool = True

    frequency_mask_range:int=40
    time_mask_range:int= 100
    drop_out: float = 0.1


    def save(self, output_path: Path):
        with open(output_path, 'w') as f:
            yaml.dump(asdict(self), f)
