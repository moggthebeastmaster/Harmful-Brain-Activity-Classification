"""
kaggle 投稿用の submission.csv を作成する
"""

from pathlib import Path
import pandas as pd
import sys



# github のソースコードへのパスを通す
root = Path(__file__).parents[0]
sys.path.append(str(root))


# データフォルダへのパス
data_dir = root.joinpath("data")
eegs_dir = data_dir.joinpath("hms-harmful-brain-activity-classification/test_eegs")
spectrograms_dir = data_dir.joinpath("hms-harmful-brain-activity-classification/test_spectrograms")
meta_df = pd.read_csv(data_dir.joinpath("hms-harmful-brain-activity-classification/test.csv"))

from src.submission_runner import SubmissionRunner

trained_dir_list = [root.joinpath("outputs/runner/nn_model/wave_net/20240215"),
                    root.joinpath("outputs/runner/spectrograms_nn/efficientnet_b0/20240223"),
                    # root.joinpath("outputs/runner/xgboost/xgboost/20240302"),
                    root.joinpath("outputs/runner/external01"),
                    root.joinpath("outputs/runner/eeg_nn/resnet_gru/20240314_no_early"),
                    ]







submission_runner = SubmissionRunner(trained_dir_list=trained_dir_list,
                                     meta_df=meta_df,
                                     eegs_dir=eegs_dir,
                                     spectrograms_dir=spectrograms_dir)


predicted_df = submission_runner.predict_blend(use_one_model=False)

print(predicted_df.head())


