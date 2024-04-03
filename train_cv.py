import os
from pathlib import Path
import pandas as pd

from src.runner import Runner, RunnerConfig

if __name__ == '__main__':
    root = Path(__file__).parents[0]
    data_root = Path(os.environ["kaggle_data_root"])

    # モデルを用意
    from src.framework.spectrograms_nn.model import SpectrogramsModel, EfficientNetConfig
    from src.framework.eeg_nn.model import EEGResnetGRUConfig, EEGNeuralNetConfig, EEGNeuralNetModel

    # eegs_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_eegs")
    # spectrograms_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_spectrograms")
    # meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/train.csv"))
    # output_dir = root.joinpath("outputs", "runner", "spectrograms_nn", "eeg_efficientnet_b0_v2", "20240328")

    eegs_dir = data_root.joinpath("hms-harmful-brain-activity-classification/train_eegs")
    spectrograms_dir = data_root.joinpath("hms-harmful-brain-activity-classification/train_spectrograms")
    meta_df = pd.read_csv(data_root.joinpath("hms-harmful-brain-activity-classification/train.csv"))
    output_dir = root.joinpath("outputs", "runner", "WaveNet", "baseline_over_10")


    # 時間短縮用に真ん中の波形のみ使用
    TARGETS_COLUMNS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    meta_df = meta_df.groupby('eeg_id').agg(lambda s: s.iloc[len(s) // 2]).reset_index(drop=False)
    meta_df = meta_df.loc[meta_df[TARGETS_COLUMNS].sum(axis=1) >= 10].reset_index(drop=False)


    # config = EfficientNetConfig(
    #     model_framework="eeg_efficientnet_b0_v2",
    #     batch_size=2 ** 4,
    #     num_worker=os.cpu_count(),
    #     max_epoch=10,
    #     early_stop=False,
    # )
    # config = EEGResnetGRUConfig(num_worker=os.cpu_count(),
    #                             max_epoch=10*2,
    #                             early_stop=False)
    
    config = EEGNeuralNetConfig(num_worker=os.cpu_count(),
                                batch_size=2 ** 4,
                                max_epoch=10*2,
                                early_stop=False)

    model = EEGNeuralNetModel(config)#SpectrogramsModel(config)

    # runner を実行
    runner_config = RunnerConfig()
    runner = Runner(output_dir=output_dir, model=model, runner_config=runner_config)
    score_dict = runner.run_cv(meta_df, eegs_dir=eegs_dir, spectrograms_dir=spectrograms_dir)
    #score_dict = runner.run_2_stage_cv(meta_df, eegs_dir=eegs_dir, spectrograms_dir=spectrograms_dir)
