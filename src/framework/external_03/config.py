from dataclasses import dataclass, asdict
from pathlib import Path
import yaml


@dataclass
class External03Config:
    model_framework: str = "External03Model"

    def save(self, output_path: Path):
        with open(output_path, 'w') as f:
            yaml.dump(asdict(self), f)
