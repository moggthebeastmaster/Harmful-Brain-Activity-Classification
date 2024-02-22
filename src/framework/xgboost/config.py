from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

@dataclass
class XGBoostModelConfig:
    model_framework: str = "XGBoost"
    data_use_second: int = 50

    # 2024/2/17 fold0 kaggle_score = 1.0524149454152802
    n_estimators:int= 1000
    max_depth:int = 6
    learning_rate:float = 0.007267677483744385
    min_child_weight:float = 6.02465737725677
    subsample:float = 0.6505792273371327


    def save(self, output_path:Path):
        with open(output_path, 'w') as f:
            yaml.dump(asdict(self), f)

