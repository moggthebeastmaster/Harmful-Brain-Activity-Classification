from pathlib import Path
import pandas as pd
from yaml import safe_load
import numpy as np
import tqdm
from xgboost import XGBRegressor

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
        elif model_framework in ["xgboost"]:
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

        elif model_framework in ["eeg_efficientnet_b0"]:
            from src.framework.spectrograms_nn.model import SpectrogramsModel, EfficientNetConfig

            # config 設定
            config = EfficientNetConfig(**config_dict)
            model = SpectrogramsModel(config=config)
            model.load(trained_dir / "model.pt")

        elif model_framework in ['ResnetGRU']:
            from src.framework.eeg_nn.model import EEGNeuralNetModel
            from src.framework.eeg_nn.config import EEGResnetGRUConfig
            # config 設定
            config = EEGResnetGRUConfig(**config_dict)
            model = EEGNeuralNetModel(config=config)
            model.load(trained_dir / "model.pt")

        elif model_framework in ['External01Model']:
            from src.framework.external_01.model import External01Model
            from src.framework.external_01.config import External01Config
            # config 設定
            config = External01Config(**config_dict)
            model = External01Model(config=config)
            model.load(trained_dir / "model.ckpt")

        elif model_framework in ['External02Model']:
            from src.framework.external_02.model import External02Model
            from src.framework.external_02.config import External02Config
            # config 設定
            config = External02Config(**config_dict)
            model = External02Model(config=config)
            model.load(trained_dir / "model.pth")
        elif model_framework in ['External03Model']:
            from src.framework.external_03.model import External03Model
            from src.framework.external_03.config import External03Config
            # config 設定
            config = External03Config(**config_dict)
            model = External03Model(config=config)
            model.load(trained_dir / "model")

        elif model_framework in ['External04Model']:
            from src.framework.external_04.model import External04Model
            from src.framework.external_04.config import External04Config
            # config 設定
            config = External04Config(**config_dict)
            model = External04Model(config=config)
            model.load(trained_dir / "model.pth")
        else:
            raise NotImplementedError(model_framework)

        # 予測
        predicted_df = model.predict(test_df=self.meta_df,
                                     eegs_dir=self.eegs_dir,
                                     spectrograms_dir=self.spectrograms_dir,
                                     )

        del model
        return predicted_df

    def predict_one(self, trained_dir: Path, use_one_model: bool = False):
        if use_one_model:
            return self.get_predicted_df(trained_dir=trained_dir)

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
        with tqdm.tqdm(self.trained_dir_list) as pbar:
            for trained_dir in pbar:
                pbar.set_description(f"predict_model:{trained_dir}")
                predicted_df_list.append(self.predict_one(trained_dir=trained_dir, use_one_model=False))

        eeg_id = predicted_df_list[0]["eeg_id"].to_numpy()[:, np.newaxis]
        targets_columns = list(predicted_df_list[0].columns[1:])
        values = [predicted_df.iloc[:, 1:].to_numpy() for predicted_df in predicted_df_list]
        values = np.sum(np.stack(values), axis=0)
        predict_y = values / values.sum(axis=1, keepdims=True)
        predicted_df = pd.DataFrame(np.concatenate([eeg_id, predict_y], axis=1), columns=["eeg_id"] + targets_columns)
        return predicted_df

    def predict_ensemble(self, method="linear"):

        predicted_df_list = []
        model_framework_list = []
        with tqdm.tqdm(self.trained_dir_list) as pbar:
            for trained_dir in pbar:
                pbar.set_description(f"predict_model:{trained_dir}")
                predicted_df_list.append(self.predict_one(trained_dir=trained_dir))
                with open(trained_dir / "fold_0" / "hparams.yaml") as f:
                    config_dict = safe_load(f)
                    model_framework = config_dict["model_framework"]
                    model_framework_list.append(model_framework)

        # ensemble model を作る
        external_df_list = []
        train_x_df = None
        train_y_df = None
        test_x_df = None

        for index in range(len(model_framework_list)):
            model_framework = model_framework_list[index]
            if model_framework.startswith("External"):
                external_df_list.append(predicted_df_list[index])
            else:
                fold_pred_df = pd.concat([pd.read_csv(self.trained_dir_list[index]/f"fold_{n}"/"predicts.csv", index_col=0) for n in range(self.fold_num)])
                fold_label_df = pd.concat([pd.read_csv(self.trained_dir_list[index]/f"fold_{n}"/"label.csv", index_col=0) for n in range(self.fold_num)])
                fold_pred_df = fold_pred_df.astype({"eeg_id":int}).reset_index(drop=True)
                fold_label_df = fold_label_df.astype({"eeg_id":int}).reset_index(drop=True)
                if train_x_df is None:
                    train_x_df = fold_pred_df
                    train_y_df = fold_label_df
                else:
                    train_x_df = train_x_df.join(fold_pred_df.drop("eeg_id", axis=1), how='inner', rsuffix='_right')

                test_x = predicted_df_list[index].astype({"eeg_id":int}).reset_index(drop=True)
                if test_x_df is None:
                    test_x_df = test_x
                else:
                    test_x_df = test_x_df.join(test_x.drop("eeg_id", axis=1),how='inner', rsuffix='_right')

        eeg_id = predicted_df_list[0]["eeg_id"].to_numpy()[:, np.newaxis]
        targets_columns = list(predicted_df_list[0].columns[1:])

        if train_x_df is not None:
            from sklearn.multioutput import RegressorChain
            from sklearn.linear_model import LinearRegression
            base_estimater = LinearRegression()
            chain = RegressorChain(base_estimator=base_estimater, random_state=0)
            x = train_x_df.values[:, 1:]
            y = train_y_df.values[:, 1:]
            chain.fit(x,y)

            pred = chain.predict(test_x_df.values[:, 1:])
            pred = np.clip(pred, 0., 1.,)
            predict_y = pred / pred.sum(axis=1, keepdims=True)

            our_predicted_df = pd.DataFrame(np.concatenate([eeg_id, predict_y], axis=1), columns=["eeg_id"] + targets_columns)
        else:
            our_predicted_df = None

        # external グループ内で平均をとる
        if len(external_df_list)>0:
            values = [ex_predicted_df.iloc[:, 1:].to_numpy() for ex_predicted_df in external_df_list]
            values = np.sum(np.stack(values), axis=0)
            predict_y = values / values.sum(axis=1, keepdims=True)
            ex_predicted_df = pd.DataFrame(np.concatenate([eeg_id, predict_y], axis=1), columns=["eeg_id"] + targets_columns)
        else:
            ex_predicted_df = None

        # 二つの平均をとる
        if our_predicted_df is None:
            predicted_df = ex_predicted_df
        elif ex_predicted_df is None:
            predicted_df = our_predicted_df
        else:
            values = np.sum(np.stack([our_predicted_df.iloc[:, 1:].to_numpy(), ex_predicted_df.iloc[:, 1:].to_numpy()]), axis=0)
            predict_y = values / values.sum(axis=1, keepdims=True)
            predicted_df = pd.DataFrame(np.concatenate([eeg_id, predict_y], axis=1), columns=["eeg_id"] + targets_columns)

        return predicted_df












        eeg_id = predicted_df_list[0]["eeg_id"].to_numpy()[:, np.newaxis]
        targets_columns = list(predicted_df_list[0].columns[1:])
        values = [predicted_df.iloc[:, 1:].to_numpy() for predicted_df in predicted_df_list]
        values = np.sum(np.stack(values), axis=0)
        predict_y = values / values.sum(axis=1, keepdims=True)
        predicted_df = pd.DataFrame(np.concatenate([eeg_id, predict_y], axis=1), columns=["eeg_id"] + targets_columns)
        return predicted_df
