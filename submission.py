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
eegs_dir = data_dir.joinpath("hms-harmful-brain-activity-classification/train_eegs")
spectrograms_dir = data_dir.joinpath("hms-harmful-brain-activity-classification/train_spectrograms")
meta_df = pd.read_csv(data_dir.joinpath("hms-harmful-brain-activity-classification/train.csv"))[:100]

from src.submission_runner import SubmissionRunner

trained_dir_list = [# root.joinpath("outputs/runner/xgboost/20240218"),
                    # root.joinpath("outputs/runner/nn_model/wave_net/20240215"),
                    root.joinpath("outputs/runner/spectrograms_nn/efficientnet_b0/20240223")
                    ]

submission_runner = SubmissionRunner(trained_dir_list=trained_dir_list,
                                     meta_df=meta_df,
                                     eegs_dir=eegs_dir,
                                     spectrograms_dir=spectrograms_dir)


predicted_df = submission_runner.predict_blend()

print(predicted_df.head())


