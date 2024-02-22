"""
kaggle 投稿用の submission.csv を作成する
"""

from pathlib import Path
import pandas as pd
import sys
from yaml import safe_load



# github のソースコードへのパスを通す
root = Path(__file__).parents[0]
sys.path.append(str(root))

# データフォルダへのパス
data_dir = root.joinpath("data")
eegs_dir = data_dir.joinpath("hms-harmful-brain-activity-classification/test_eegs")
spectrograms_dir = data_dir.joinpath("hms-harmful-brain-activity-classification/test_spectrograms")
meta_df = pd.read_csv(data_dir.joinpath("hms-harmful-brain-activity-classification/test.csv"))




def get_submission_df(trained_dir:Path):
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

    else:
        raise NotImplementedError


    # 予測
    predicted_df = model.predict(test_df=meta_df, eegs_dir=eegs_dir, spectrograms_dir=spectrograms_dir)

    return predicted_df

# 学習済みモデルのあるフォルダ (kaggle Dataset をアップロードして、そこに通す)
trained_dir = root.joinpath("outputs/runner/xgboost/20240218")
predicted_df1 = get_submission_df(trained_dir)
print(predicted_df1)

trained_dir = root.joinpath("outputs/runner/nn_model/wave_net/20240215")
predicted_df2 = get_submission_df(trained_dir)
print(predicted_df2)

# とりあえずブレンド(後日スタックモデルを作りたい)
predicted_df = (predicted_df1 + predicted_df2) / 2
print(predicted_df)



