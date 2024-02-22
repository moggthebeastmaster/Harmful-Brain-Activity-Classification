from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from pathlib import Path
import sys

root = Path(__file__).parents[3]
sys.path.append(str(root))

from src.kaggle_score import kaggle_score
from src.framework.xgboost.config import XGBoostModelConfig
from src.framework.xgboost.data import XGBoostDataset

TARGETS_COLUMNS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']


class XGBoostModel():
    def __init__(self, config: XGBoostModelConfig):
        self.config = config
        self.num_classes = len(TARGETS_COLUMNS)
        self.xgb_classifier = None

    def initialize_model(self):
        self.xgb_classifier = XGBClassifier(n_estimators=self.config.n_estimators,
                                            max_depth=self.config.max_depth,
                                            min_child_weight=self.config.min_child_weight,
                                            subsample=self.config.subsample,
                                            learning_rate=self.config.learning_rate,
                                            objective='multi:softmax',
                                            num_class=self.num_classes,
                                            early_stopping_rounds=100,
                                            verbose=100,
                                            tree_method='hist',
                                            device="cpu")

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, eegs_dir: Path, spectrograms_dir: Path,
              output_dir: Path):
        assert val_df is not None, "このモデルは val_df が必須"

        self.initialize_model()

        train_dataset = XGBoostDataset(meta_df=train_df, eegs_dir=eegs_dir, config=self.config, with_label=True)
        val_dataset = XGBoostDataset(meta_df=val_df, eegs_dir=eegs_dir, config=self.config, with_label=True)

        self.xgb_classifier.fit(train_dataset.x, train_dataset.y, eval_set=[(val_dataset.x, val_dataset.y)],
                                verbose=100)
        predict_y = self.xgb_classifier.predict_proba(val_dataset.x)

        # 評価
        label = val_dataset.meta_label_prob
        eed_id = val_dataset.meta_eeg_id[:, np.newaxis]

        label_df = pd.DataFrame(np.concatenate([eed_id, label], axis=1), columns=["eeg_id"] + TARGETS_COLUMNS)
        predicts_df = pd.DataFrame(np.concatenate([eed_id, predict_y], axis=1), columns=["eeg_id"] + TARGETS_COLUMNS)
        score = kaggle_score(label_df.copy(), predicts_df.copy(), "eeg_id")
        score_df = pd.DataFrame([score], index=["kaggle_score"])

        # 保存処理
        output_dir.mkdir(exist_ok=True, parents=True)
        self.save(output_dir.joinpath("model.json"))
        score_df.to_csv(output_dir.joinpath("score.csv"))
        label_df.to_csv(output_dir.joinpath("label.csv"))
        predicts_df.to_csv(output_dir.joinpath("predicts.csv"))
        self.config.save(output_dir.joinpath("hparams.yaml"))
        return {"kaggle_score": score}

    def predict(self, test_df: pd.DataFrame, eegs_dir: Path, spectrograms_dir: Path):
        test_dataset = XGBoostDataset(meta_df=test_df, eegs_dir=eegs_dir, config=self.config, with_label=False)
        predict_y = self.xgb_classifier.predict_proba(test_dataset.x)
        eed_id = test_dataset.meta_eeg_id[:, np.newaxis]
        predicts_df = pd.DataFrame(np.concatenate([eed_id, predict_y], axis=1), columns=["eeg_id"] + TARGETS_COLUMNS)

        return predicts_df

    def save(self, file_path: Path):
        self.xgb_classifier.save_model(file_path)

    def load(self, file_path: Path):
        self.initialize_model()
        self.xgb_classifier.load_model(file_path)


if __name__ == '__main__':
    from sklearn.model_selection import GroupShuffleSplit

    root = Path(__file__).parents[3]

    output_dir = root.joinpath("outputs", "simple_model", "20240206")

    config = XGBoostModelConfig(data_use_second=50)
    eegs_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_eegs")
    spectrograms_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_spectrograms")
    meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/train.csv"))

    # 5-flod 分け
    gkf = GroupShuffleSplit(n_splits=5, random_state=0)
    meta_df['fold'] = 0
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(meta_df, meta_df.expert_consensus, meta_df.patient_id)):
        meta_df.loc[val_idx, 'fold'] = fold
    simple_model = XGBoostModel(config=config)

    train_df = meta_df[meta_df["fold"] != 0]
    val_df = meta_df[meta_df["fold"] == 0]

    # サブデータの多くは重複しているので最初の波形のみ使用
    train_df = train_df[train_df["eeg_sub_id"] == 0].reset_index()
    simple_model.train(train_df, val_df, eegs_dir, spectrograms_dir, output_dir=output_dir.joinpath("fold_0"))
