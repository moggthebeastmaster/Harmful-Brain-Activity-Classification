import os
from pathlib import Path
import pandas as pd

from src.runner import Runner,RunnerConfig

if __name__ == '__main__':
    root = Path(__file__).parents[0]
    
    # モデルを用意
    from src.framework.spectrograms_nn.model import SpectrogramsModel,EfficientNetConfig
    from src.framework.eeg1dgru.model import Eeg1dGRUModel
    from src.framework.eeg1dgru.config import Eeg1dGRUConfig

    data_root = Path(os.environ["kaggle_data_root"]).joinpath("hms-harmful-brain-activity-classification/test")
    eegs_dir = data_root.joinpath("raw_eeg")
    spectrograms_dir = data_root.joinpath("spectrogram")
    meta_df = pd.read_csv(data_root.joinpath("train.csv"))
    output_dir = root.joinpath("outputs", "runner", "eeg1dgru", "20240324")


    config = Eeg1dGRUConfig(
        model_framework="resnet_1d_gru",
        batch_size=2**4,
        num_worker=os.cpu_count()//2,
        max_epoch=20
    )

    model = Eeg1dGRUModel(config)

    # runner を実行
    runner_config = RunnerConfig()
    runner = Runner(output_dir=output_dir, model=model, runner_config=runner_config)
    score_dict = runner.run_cv(meta_df, eegs_dir=eegs_dir, spectrograms_dir=spectrograms_dir)
