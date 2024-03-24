from pathlib import Path
import pandas as pd
import optuna
import os
from sklearn.model_selection import GroupShuffleSplit
from src.runner import Runner,RunnerConfig

root = Path(__file__).parents[0]

def objective(trial:optuna.Trial):
    eegs_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_eegs")
    spectrograms_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_spectrograms")
    meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/train.csv"))

    # 時間短縮用に真ん中の波形のみ使用
    meta_df = meta_df.groupby('eeg_id').agg(lambda s: s.iloc[len(s) // 2]).reset_index(drop=False)

    # モデルを用意
    from src.framework.spectrograms_nn.model import SpectrogramsModel, EfficientNetConfig

    # config 設定
    config = EfficientNetConfig(model_framework="eeg_efficientnet_b0",
                                batch_size = 2 ** 4,
                                learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.01, log=True),
                                weight_decay=trial.suggest_float("weight_decay", 0.001, 0.9, log=True),
                                frequency_mask_range=trial.suggest_int("frequency_mask_range", 1, 60),
                                time_mask_range=trial.suggest_int("time_mask_range", 1, 150),
                                drop_out=trial.suggest_float("drop_out", 0., 1.),
                                mix_up_alpha = trial.suggest_float("mix_up_alpha", 0., 2.),
                                num_worker=os.cpu_count()//2,
                                max_epoch=5,
                                early_stop=False,
                                )
    model = SpectrogramsModel(config=config)

    # runner を実行
    runner_config = RunnerConfig()
    runner = Runner(output_dir=output_dir, model=model, runner_config=runner_config)

    # fold0のみで実験
    gkf = GroupShuffleSplit(n_splits=runner_config.fold_num, random_state=runner_config.data_split_seed)
    for fold, (train_indexs, val_indexs) in enumerate(
            gkf.split(meta_df, None, meta_df.patient_id)):
        train_df = meta_df.loc[train_indexs]
        val_df = meta_df.loc[val_indexs]
        score_dict = runner.run_train(train_df, val_df, eegs_dir=eegs_dir, spectrograms_dir=spectrograms_dir)
        break

    return score_dict["kaggle_score"]

if __name__ == '__main__':

    frame_work = "spectrograms_nn"
    model_name = "eeg_efficientnet_b0"
    date = "20240324"


    output_dir = root.joinpath("outputs", "optuna", frame_work, model_name, date)

    study_name = '-'.join([frame_work, model_name, date])
    storage = 'sqlite:///optuna_results.db'
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=100,)

    study.best_params