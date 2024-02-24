import scipy.stats
import numpy as np
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import tqdm
import sys
import matplotlib.pyplot as plt
root = Path(__file__).parents[3]
sys.path.append(str(root))

from src.framework.xgboost.config import XGBoostModelConfig


TARGETS = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}
TARGETS_INV = {x:y for y,x in TARGETS.items()}
TARGETS_NAME = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
FREQUENCY = 200
ONE_DATA_TIME = 50
COLUMN_NAMES =['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG']
COLUMN_NAMES =['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']

class XGBoostDataset():
    def __init__(self, meta_df:pd.DataFrame, eegs_dir:Path, config:XGBoostModelConfig, with_label=False):
        self.meta_df = meta_df
        self.eegs_dir = eegs_dir
        self.config = config
        self.with_label = with_label

        self.meta_eeg_id = self.meta_df.eeg_id.values
        self.meta_eeg_sub_id = self.meta_df.eeg_sub_id.values if with_label else np.zeros_like(self.meta_eeg_id)
        self.meta_eeg_label_offset_seconds = self.meta_df.eeg_label_offset_seconds.values if with_label else np.zeros_like(self.meta_eeg_id)

        self.meta_spectrogram_id = self.meta_df.spectrogram_id.values
        self.meta_spectrogram_sub_id = self.meta_df.spectrogram_sub_id.values if with_label else np.zeros_like(self.meta_spectrogram_id)
        self.meta_spectrogram_label_offset_seconds = self.meta_df.spectrogram_label_offset_seconds.values if with_label else np.zeros_like(self.meta_spectrogram_id)

        self.meta_patient_id = self.meta_df.patient_id.values

        if self.with_label:
            self.meta_label = self.meta_df.expert_consensus
            y = self.meta_df[TARGETS_NAME].values
            self.meta_label_prob = y / y.sum(axis=1, keepdims=True)

        eeg_array = np.zeros(shape=(len(self.meta_eeg_id), int(FREQUENCY * self.config.data_use_second), len(COLUMN_NAMES)))
        for index in tqdm.tqdm(range(len(self.meta_eeg_id)), desc="load parquets"):
            eeg_id = self.meta_eeg_id[index]
            eeg_path = self.eegs_dir.joinpath(f"{eeg_id}.parquet")

            # 対象データの中心時間。 基本的には offset + 25sec
            target_center_time = self.meta_eeg_label_offset_seconds[index] + ONE_DATA_TIME//2
            # 対象データの切り出し範囲
            target_start_point = int(FREQUENCY*(target_center_time - self.config.data_use_second//2))
            target_end_point = int(FREQUENCY*(target_center_time + self.config.data_use_second//2))

            # 読み込み
            eeg_table = pq.read_table(eeg_path, memory_map=True, columns=COLUMN_NAMES)

            # 対象データの切り出し
            eeg_table = eeg_table[target_start_point:target_end_point]
            inner_array = np.asarray(eeg_table)
            if inner_array.shape[0]==len(COLUMN_NAMES):
                inner_array = inner_array.T
            eeg_array[index] = inner_array

            del eeg_table

        self.x = extract_eeg_features(eeg_array)

        mean_values = np.nanmean(self.x, axis=0)
        nan_indices = np.isnan(self.x)
        self.x[nan_indices] = np.take(mean_values, nan_indices.nonzero()[1])

        # ラベル情報
        self.y = np.argmax(self.meta_label_prob, axis=1) if self.with_label else np.zeros(shape=(len(self.meta_eeg_id), len(TARGETS, )), dtype=float)

def extract_eeg_features(eeg_data):
    """
    Extract features from EEG data, including LL Spec, LP Spec, RP Spec, RL Spec

    Parameters:
    - eeg_data: EEG data with shape (number of samples, time points, number of electrodes)

    Returns:
    - Feature matrix with shape (number of samples, number of features)
    """
    # Initialize the feature matrix
    num_samples, num_time_points, num_electrodes = eeg_data.shape
    num_features = 8 * num_electrodes + 4 * 8  # Mean, standard deviation, minimum, maximum, skewness, kurtosis, energy, entropy for each electrode + 4 global features
    features = np.zeros((num_samples, num_features))

    # normalize
    normal_eeg_data = (eeg_data - eeg_data.mean(axis=1, keepdims=True) ) / (eeg_data.std(axis=1, keepdims=True) + 1e-5)

    # Extract features
    for sample_idx in tqdm.tqdm(range(num_samples), desc="extract_eeg_features"):
        for electrode_idx in range(num_electrodes):
            electrode_data = normal_eeg_data[sample_idx, :, electrode_idx]
            # electrode_data = eeg_data[sample_idx, :, electrode_idx]
            feature_idx = electrode_idx * 8

            # Mean
            features[sample_idx, feature_idx] = np.mean(electrode_data)
            # Standard deviation
            features[sample_idx, feature_idx + 1] = np.std(electrode_data)
            # Minimum
            features[sample_idx, feature_idx + 2] = np.min(electrode_data)
            # Maximum
            features[sample_idx, feature_idx + 3] = np.max(electrode_data)
            # Skewness
            features[sample_idx, feature_idx + 4] = scipy.stats.skew(electrode_data)
            # Kurtosis
            features[sample_idx, feature_idx + 5] = scipy.stats.kurtosis(electrode_data)
            # Energy
            features[sample_idx, feature_idx + 6] = np.sum(electrode_data ** 2) / len(electrode_data)
            # Entropy
            features[sample_idx, feature_idx + 7] = scipy.stats.entropy(np.abs(electrode_data))
        # Additional features: LL Spec, LP Spec, RP Spec, RL Spec
        ll_spec = (np.abs(np.fft.fft(eeg_data[sample_idx, :, 0] - eeg_data[sample_idx, :, 4]))**2 +
                   np.abs(np.fft.fft(eeg_data[sample_idx, :, 4] - eeg_data[sample_idx, :, 5]))**2 +
                   np.abs(np.fft.fft(eeg_data[sample_idx, :, 5] - eeg_data[sample_idx, :, 6]))**2 +
                   np.abs(np.fft.fft(eeg_data[sample_idx, :, 6] - eeg_data[sample_idx, :, 7]))**2) / 4
        features[sample_idx, -4*8] = np.mean(ll_spec)
        features[sample_idx, -4*8 + 1] = np.std(ll_spec)
        features[sample_idx, -4*8 + 2] = np.min(ll_spec)
        features[sample_idx, -4*8 + 3] = np.max(ll_spec)
        features[sample_idx, -4*8 + 4] = scipy.stats.skew(ll_spec)
        features[sample_idx, -4*8 + 5] = scipy.stats.kurtosis(ll_spec)
        features[sample_idx, -4*8 + 6] = np.sum(ll_spec ** 2) / len(ll_spec)
        features[sample_idx, -4*8 + 7] = scipy.stats.entropy(np.abs(ll_spec))

        lp_spec = (np.abs(np.fft.fft(eeg_data[sample_idx, :, 0] - eeg_data[sample_idx, :, 1]))**2 +
                   np.abs(np.fft.fft(eeg_data[sample_idx, :, 1] - eeg_data[sample_idx, :, 2]))**2 +
                   np.abs(np.fft.fft(eeg_data[sample_idx, :, 2] - eeg_data[sample_idx, :, 3]))**2 +
                   np.abs(np.fft.fft(eeg_data[sample_idx, :, 3] - eeg_data[sample_idx, :, 7]))**2) / 4
        features[sample_idx, -3 * 8] = np.mean(lp_spec)
        features[sample_idx, -3 * 8 + 1] = np.std(lp_spec)
        features[sample_idx, -3 * 8 + 2] = np.min(lp_spec)
        features[sample_idx, -3 * 8 + 3] = np.max(lp_spec)
        features[sample_idx, -3 * 8 + 4] = scipy.stats.skew(lp_spec)
        features[sample_idx, -3 * 8 + 5] = scipy.stats.kurtosis(lp_spec)
        features[sample_idx, -3 * 8 + 6] = np.sum(lp_spec ** 2) / len(lp_spec)
        features[sample_idx, -3 * 8 + 7] = scipy.stats.entropy(np.abs(lp_spec))


        rp_spec = (np.abs(np.fft.fft(eeg_data[sample_idx, :, 8] - eeg_data[sample_idx, :, 9]))**2 +
                   np.abs(np.fft.fft(eeg_data[sample_idx, :, 9] - eeg_data[sample_idx, :, 10]))**2 +
                   np.abs(np.fft.fft(eeg_data[sample_idx, :, 10] - eeg_data[sample_idx, :, 11]))**2 +
                   np.abs(np.fft.fft(eeg_data[sample_idx, :, 11] - eeg_data[sample_idx, :, 15]))**2) / 4
        features[sample_idx, -2] = np.mean(rp_spec)
        features[sample_idx, -2 * 8] = np.mean(rp_spec)
        features[sample_idx, -2 * 8 + 1] = np.std(rp_spec)
        features[sample_idx, -2 * 8 + 2] = np.min(rp_spec)
        features[sample_idx, -2 * 8 + 3] = np.max(rp_spec)
        features[sample_idx, -2 * 8 + 4] = scipy.stats.skew(rp_spec)
        features[sample_idx, -2 * 8 + 5] = scipy.stats.kurtosis(rp_spec)
        features[sample_idx, -2 * 8 + 6] = np.sum(rp_spec ** 2) / len(rp_spec)
        features[sample_idx, -2 * 8 + 7] = scipy.stats.entropy(np.abs(rp_spec))

        rl_spec = (np.abs(np.fft.fft(eeg_data[sample_idx, :, 8] - eeg_data[sample_idx, :, 12]))**2 +
                   np.abs(np.fft.fft(eeg_data[sample_idx, :, 12] - eeg_data[sample_idx, :, 13]))**2 +
                   np.abs(np.fft.fft(eeg_data[sample_idx, :, 13] - eeg_data[sample_idx, :, 14]))**2 +
                   np.abs(np.fft.fft(eeg_data[sample_idx, :, 14] - eeg_data[sample_idx, :, 15]))**2) / 4
        features[sample_idx, -1] = np.mean(rl_spec)
        features[sample_idx, -2] = np.mean(rl_spec)
        features[sample_idx, -3 * 8] = np.mean(rl_spec)
        features[sample_idx, -3 * 8 + 1] = np.std(rl_spec)
        features[sample_idx, -3 * 8 + 2] = np.min(rl_spec)
        features[sample_idx, -3 * 8 + 3] = np.max(rl_spec)
        features[sample_idx, -3 * 8 + 4] = scipy.stats.skew(rl_spec)
        features[sample_idx, -3 * 8 + 5] = scipy.stats.kurtosis(rl_spec)
        features[sample_idx, -3 * 8 + 6] = np.sum(rl_spec ** 2) / len(rl_spec)
        features[sample_idx, -3 * 8 + 7] = scipy.stats.entropy(np.abs(rl_spec))


    return features

def main():
    root = Path(__file__).parents[3]
    import tqdm

    train = True
    config = XGBoostModelConfig(data_use_second=50)
    if train:
        eegs_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_eegs")
        meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/train.csv"))
        meta_df = meta_df[meta_df["eeg_sub_id"] == 0].sort_values("eeg_id").reset_index(drop=True)[:1000]
    else:
        meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/test.csv"))
        eegs_dir = root.joinpath("data/hms-harmful-brain-activity-classification/test_eegs")



    dataset = XGBoostDataset(meta_df, eegs_dir, config, with_label=train)

    for x, y in tqdm.tqdm(zip(dataset.x, dataset.y)):
        pass
        # print(data_one)


if __name__ == '__main__':
    import line_profiler
    import atexit
    profile = line_profiler.LineProfiler()
    profile.add_module(XGBoostDataset)

    # デバッグ処理を行いたい場合は、以下を通常のmainに変更する
    # profile.runcall(main)
    main()

    atexit.register(profile.print_stats)
