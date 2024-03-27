from pathlib import Path
import pandas as pd
import optuna
import os
from sklearn.model_selection import GroupShuffleSplit
from src.runner import Runner,RunnerConfig

root = Path(__file__).parents[0]
data_root = Path(os.environ["kaggle_data_root"]).joinpath("hms-harmful-brain-activity-classification", "test")

def objective(trial:optuna.Trial):
    eegs_dir = data_root.joinpath("raw_eeg")
    spectrograms_dir = data_root.joinpath("spectrogram")
    meta_df = pd.read_csv(data_root.joinpath("train.csv"))

    # モデルを用意
    from src.framework.eeg1dgru.config import Eeg1dGRUConfig
    from src.framework.eeg1dgru.model import Eeg1dGRUModel

    # config 設定
    config = Eeg1dGRUConfig(model_framework="resnet_1d_convtrans",
                                learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.01, log=True),
                                weight_decay=trial.suggest_float("weight_decay", 0.001, 0.9, log=True),
                                mixup_rate = trial.suggest_float("mixup_rate", 0., 1.),
                                drop_out = trial.suggest_float("drop_out", 0., 1.),
                                num_worker=os.cpu_count()//2,
                                max_epoch=20,
                                )
    model = Eeg1dGRUModel(config=config)

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

    frame_work = "resnet_1d_convtrans"
    model_name = "resnet_gru"
    date = "20240326"


    output_dir = root.joinpath("outputs", "optuna", frame_work, date)

    study_name = '-'.join([frame_work, date])
    storage = 'sqlite:///optuna_results.db'
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=100,)

    study.best_params