from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

TARGETS_COLUMNS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

@dataclass
class EfficientNetConfig:
    # 2024/2/23 最適化
    model_framework: str = "efficientnet_b0"
    data_use_second: int = 600
    batch_size: int = 2 ** 3
    accumulate_grad_batches: int = 2 ** 0  # この回数だけロスを蓄積してパラメータを更新する。
    learning_rate: float = 0.0015020898211300792
    weight_decay: float = 0.08641866451903213
    warmup_steps_ratio: float = 0.1
    mix_up_alpha:float = 0.
    max_epoch: int = 10
    num_worker: int = 0
    early_stop: bool = False

    frequency_mask_range:int=58
    time_mask_range:int= 103
    drop_out: float = 0.6082302930971167


    def save(self, output_path: Path):
        with open(output_path, 'w') as f:
            yaml.dump(asdict(self), f)
