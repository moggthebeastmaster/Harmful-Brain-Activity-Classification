from pathlib import Path
import pandas as pd
import sys
from yaml import safe_load
import numpy as np


class SubmissionRunner:

    def __init__(self,
                 trained_dir_list: list[Path],
                 meta_df: pd.DataFrame,
                 eegs_dir: Path,
                 spectrograms_dir: Path,
                 fold_num=5,
                 ):
        self.trained_dir_list = trained_dir_list
        self.meta_df = meta_df
        self.eegs_dir = eegs_dir
        self.spectrograms_dir = spectrograms_dir
        self.fold_num = fold_num

    def get_predicted_df(self, trained_dir: Path):
        # hparams.yaml から model_framework を読み取る
        with open(trained_dir / "hparams.yaml") as f:
            config_dict = safe_load(f)
            model_framework = config_dict["model_framework"]

        # model_framework　によって読み込むクラスを変更する
        if model_framework in ["WaveNet"]:
            from src.framework.eeg_nn.model import EEGNeuralNetModel
            from src.framework.eeg_nn.model import EEGNeuralNetConfig
            # config 設定
            config = EEGNeuralNetConfig(**config_dict)
            model = EEGNeuralNetModel(config=config)
            model.load(trained_dir / "model.pt")
        elif model_framework in ["XGBoost"]:
            from src.framework.xgboost.model import XGBoostModelConfig, XGBoostModel

            # config 設定
            config = XGBoostModelConfig(**config_dict)
            model = XGBoostModel(config=config)
            model.load(trained_dir / "model.json")

        elif model_framework in ["efficientnet_b0"]:
            from src.framework.spectrograms_nn.model import SpectrogramsModel, EfficientNetConfig

            # config 設定
            config = EfficientNetConfig(**config_dict)
            model = SpectrogramsModel(config=config)
            model.load(trained_dir / "model.pt")

        else:
            raise NotImplementedError

        # 予測
        predicted_df = model.predict(test_df=self.meta_df,
                                     eegs_dir=self.eegs_dir,
                                     spectrograms_dir=self.spectrograms_dir,
                                     )

        return predicted_df

    def predict_one(self, trained_dir: Path):
        # fold 別の結果を平均する

        predicted_df_list = []
        for n in range(self.fold_num):
            fold_path = trained_dir.joinpath(f"fold_{n}")
            df = self.get_predicted_df(trained_dir=fold_path)
            predicted_df_list.append(df)


        eeg_id = predicted_df_list[0]["eeg_id"].to_numpy()[:, np.newaxis]
        targets_columns = list(predicted_df_list[0].columns[1:])

        values = [predicted_df.iloc[:, 1:].to_numpy() for predicted_df in predicted_df_list]
        values = np.sum(np.stack(values), axis=0)
        predict_y = values / values.sum(axis=1, keepdims=True)
        predicted_df = pd.DataFrame(np.concatenate([eeg_id, predict_y], axis=1), columns=["eeg_id"] + targets_columns)
        return predicted_df


    def predict_blend(self):
        predicted_df_list = []
        for trained_dir in self.trained_dir_list:
            predicted_df_list.append(self.predict_one(trained_dir=trained_dir))

        eeg_id = predicted_df_list[0]["eeg_id"].to_numpy()[:, np.newaxis]
        targets_columns = list(predicted_df_list[0].columns[1:])
        values = [predicted_df.iloc[:, 1:].to_numpy() for predicted_df in predicted_df_list]
        values = np.sum(np.stack(values), axis=0)
        predict_y = values / values.sum(axis=1, keepdims=True)
        predicted_df = pd.DataFrame(np.concatenate([eeg_id, predict_y], axis=1), columns=["eeg_id"] + targets_columns)
        return predicted_df




