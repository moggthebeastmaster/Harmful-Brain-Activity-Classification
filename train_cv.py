import os
from pathlib import Path
import pandas as pd

from src.runner import Runner,RunnerConfig

if __name__ == '__main__':
    root = Path(__file__).parents[0]
    eegs_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_eegs")
    spectrograms_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_spectrograms")
    meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/train.csv"))
    output_dir = root.joinpath("outputs", "runner", "eeg_spectrograms_nn", "efficientnet_b0", "20240323")

    # 時間短縮用に真ん中の波形のみ使用
    meta_df = meta_df.groupby('eeg_id').agg(lambda s: s.iloc[len(s) // 2]).reset_index(drop=False)

    # モデルを用意
    from src.framework.spectrograms_nn.model import SpectrogramsModel,EfficientNetConfig

    config = EfficientNetConfig(model_framework="eeg_efficientnet_b0",
                                batch_size=2**4,
                                num_worker=os.cpu_count()//2,
                                max_epoch=20,
                                early_stop=True,
                                )

    model = SpectrogramsModel(config=config)

    # runner を実行
    runner_config = RunnerConfig()
    runner = Runner(output_dir=output_dir, model=model, runner_config=runner_config)
    score_dict = runner.run_cv(meta_df, eegs_dir=eegs_dir, spectrograms_dir=spectrograms_dir)
