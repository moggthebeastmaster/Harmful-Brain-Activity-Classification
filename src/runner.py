from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from dataclasses import dataclass


@dataclass
class RunnerConfig:
    data_split_seed: int = 0  # CVのデータ分けのシード値。基本的には0で固定すること
    random_seed: int = 42
    fold_num:int = 5


class InterfaceModel:
    """
    ダミー用クラス
    """

    def initialize_model(self):
        """
        モデル初期化
        """
        pass

    def train(self,
              train_df: pd.DataFrame,
              val_df: pd.DataFrame,
              eegs_dir: Path,
              spectrograms_dir: Path,
              output_dir: Path,
              early_stop:bool) -> dict:
        score_dict = {"kaggle_score": 0.0}
        return score_dict

    def predict(self, test_df: pd.DataFrame, eegs_dir: Path, spectrograms_dir: Path) -> pd.DataFrame:
        predicted_df:pd.DataFrame = test_df.copy()
        return predicted_df

    def save(self):
        pass

    def load(self):
        pass


class Runner():
    def __init__(self, output_dir: Path, model: InterfaceModel, runner_config: RunnerConfig):
        """
        学習用のランナー
        modelを渡す。
        model は InterfaceModel のもつメソッドが必要
        作成例としては EEGNeuralNetModel など
        Args:
            output_dir:
            model:
            runner_config: 基本的に固定
        """
        self.output_dir = output_dir
        self.model = model
        self.runner_config = runner_config

    def run_cv(self, meta_df: pd.DataFrame, eegs_dir: Path, spectrograms_dir: Path):
        # 5-fold 分け
        gkf = GroupShuffleSplit(n_splits=self.runner_config.fold_num, random_state=self.runner_config.data_split_seed)
        score_dict = {"kaggle_score":0}

        for fold, (train_indexs, val_indexs) in enumerate(
                gkf.split(meta_df, None, meta_df.patient_id)):

            self.model.initialize_model()
            train_df = meta_df.loc[train_indexs]
            val_df = meta_df.loc[val_indexs]
            score_dict_inner = self.model.train(train_df, val_df, eegs_dir, spectrograms_dir,
                                                output_dir=self.output_dir.joinpath(f"fold_{fold}"),
                                                )

            print(score_dict_inner)
            score_dict["kaggle_score"] += score_dict_inner["kaggle_score"] / self.runner_config.fold_num
            score_dict[f"kaggle_score_fold{fold}"] = score_dict_inner["kaggle_score"]

        score_df = pd.DataFrame(score_dict, index=["index"])
        score_df.to_csv(self.output_dir.joinpath("kaggle_score.csv"))

        return score_dict

    def run_train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None, eegs_dir: Path, spectrograms_dir: Path):
        return self.model.train(train_df, val_df, eegs_dir, spectrograms_dir, output_dir=self.output_dir)

