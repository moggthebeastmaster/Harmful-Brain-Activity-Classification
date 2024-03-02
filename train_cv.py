import os
from pathlib import Path
import pandas as pd

from src.runner import Runner,RunnerConfig

if __name__ == '__main__':
    root = Path(__file__).parents[0]
    eegs_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_eegs")
    spectrograms_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_spectrograms")
    meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/train.csv"))
    output_dir = root.joinpath("outputs", "runner", "xgboost", "xgboost", "20240302")

    # 時間短縮用に真ん中の波形のみ使用
    meta_df = meta_df.groupby('eeg_id').agg(lambda s: s.iloc[len(s) // 2]).reset_index(drop=False)

    # モデルを用意
    from src.framework.xgboost.model import XGBoostModel, XGBoostModelConfig

    config = XGBoostModelConfig(model_framework="xgboost",
                                )
    model = XGBoostModel(config=config)

    # runner を実行
    runner_config = RunnerConfig()
    runner = Runner(output_dir=output_dir, model=model, runner_config=runner_config)
    score_dict = runner.run_cv(meta_df, eegs_dir=eegs_dir, spectrograms_dir=spectrograms_dir)

    # 単体提出用のモデル
    #from sklearn.model_selection import train_test_split
    #train_df, val_df= train_test_split(
    #    meta_df, test_size=0.1, random_state=runner_config.data_split_seed)
    #
    # runner.run_train(train_df, val_df=val_df, eegs_dir=eegs_dir, spectrograms_dir=spectrograms_dir)
