import scipy.stats
import numpy as np
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import tqdm
import sys
import librosa
import matplotlib.pyplot as plt
import time

root = Path(__file__).parents[3]
sys.path.append(str(root))

from src.framework.xgboost.config import XGBoostModelConfig

TARGETS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
TARGETS_INV = {x: y for y, x in TARGETS.items()}
TARGETS_NAME = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
FREQUENCY = 200
ONE_DATA_TIME = 50
COLUMN_NAMES = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4',
                'T6', 'O2', 'EKG']

COLUMN_NAMES_INV = {val: key for key, val in enumerate(COLUMN_NAMES)}

CHAIN_FEATURES = [['Fp1', 'F7', 'T3', 'T5', 'O1'],
                  ['Fp1', 'F3', 'C3', 'P3', 'O1'],
                  ['Fp2', 'F8', 'T4', 'T6', 'O2'],
                  ['Fp2', 'F4', 'C4', 'P4', 'O2']]


class XGBoostDataset():
    def __init__(self, meta_df: pd.DataFrame, eegs_dir: Path, config: XGBoostModelConfig, with_label=False,
                 save_load_dir:Path|None=None):
        self.meta_df = meta_df
        self.eegs_dir = eegs_dir
        self.config = config
        self.with_label = with_label

        self.meta_eeg_id = self.meta_df.eeg_id.values
        self.meta_eeg_sub_id = self.meta_df.eeg_sub_id.values if with_label else np.zeros_like(self.meta_eeg_id)
        self.meta_eeg_label_offset_seconds = self.meta_df.eeg_label_offset_seconds.values if with_label else np.zeros_like(
            self.meta_eeg_id)

        self.meta_spectrogram_id = self.meta_df.spectrogram_id.values
        self.meta_spectrogram_sub_id = self.meta_df.spectrogram_sub_id.values if with_label else np.zeros_like(
            self.meta_spectrogram_id)
        self.meta_eeg_label_offset_seconds = self.meta_df.eeg_label_offset_seconds.values if "eeg_label_offset_seconds" in self.meta_df.columns else np.zeros_like(
            self.meta_eeg_id)
        self.meta_patient_id = self.meta_df.patient_id.values

        if self.with_label:
            self.meta_label = self.meta_df.expert_consensus
            y = self.meta_df[TARGETS_NAME].values
            self.meta_label_prob = y / y.sum(axis=1, keepdims=True)

        if (save_load_dir is None) or (not save_load_dir.exists()):
            eeg_array = np.zeros(
                shape=(len(self.meta_eeg_id), int(FREQUENCY * self.config.data_use_second), len(COLUMN_NAMES)))
            for index in tqdm.tqdm(range(len(self.meta_eeg_id)), desc="load parquets"):
                eeg_id = self.meta_eeg_id[index]
                eeg_path = self.eegs_dir.joinpath(f"{eeg_id}.parquet")

                # 対象データの中心時間。 基本的には offset + 25sec
                target_center_time = self.meta_eeg_label_offset_seconds[index] + ONE_DATA_TIME // 2
                # 対象データの切り出し範囲
                target_start_point = int(FREQUENCY * (target_center_time - self.config.data_use_second // 2))
                target_end_point = int(FREQUENCY * (target_center_time + self.config.data_use_second // 2))

                # 読み込み
                eeg_table = pq.read_table(eeg_path, memory_map=True, columns=COLUMN_NAMES)

                # 対象データの切り出し
                eeg_table = eeg_table[target_start_point:target_end_point]
                inner_array = np.asarray(eeg_table)
                if inner_array.shape[0] == len(COLUMN_NAMES):
                    inner_array = inner_array.T
                eeg_array[index] = inner_array

                del eeg_table

            mean_values = np.nanmean(eeg_array, axis=0)
            nan_indices = np.isnan(eeg_array)
            eeg_array[nan_indices] = np.take(mean_values, nan_indices.nonzero()[1])

            self.x = extract_eeg_features(eeg_array)

            mean_values = np.nanmean(self.x, axis=0)
            nan_indices = np.isnan(self.x)
            self.x[nan_indices] = np.take(mean_values, nan_indices.nonzero()[1])


            # ラベル情報
            self.y = np.argmax(self.meta_label_prob, axis=1) if self.with_label else np.zeros(
                shape=(len(self.meta_eeg_id), len(TARGETS, )), dtype=float)

            if save_load_dir is not None:
                print(f"save:{save_load_dir}")
                save_load_dir.mkdir(parents=True, exist_ok=True)
                np.save(save_load_dir / "x.npy", self.x)
                np.save(save_load_dir / "y.npy", self.y)

        else:
            print(f"load:{save_load_dir}")
            self.x = np.load(save_load_dir / "x.npy")
            self.y = np.load(save_load_dir / "y.npy")


def normalize(eeg_array: np.array, eps=1e-5, axis=(0, 1)):
    """
    eeg_array.shape = [time, ch]
    """
    eeg_array = (eeg_array - np.nanmean(eeg_array, axis=axis, keepdims=True)) / (
            np.nanstd(eeg_array, axis=axis, keepdims=True) + eps)
    return np.nan_to_num(eeg_array, nan=0)


def extract_eeg_features(eeg_data):
    """
    Extract features from EEG data, including LL Spec, LP Spec, RP Spec, RL Spec

    Parameters:
    - eeg_data: EEG data with shape (number of samples, time points, number of electrodes)

    Returns:
    - Feature matrix with shape (number of samples, number of features)
    """

    MEAN = True
    STD = True
    STFT_SPEC = True
    MEL_SPEC = True

    data_num = eeg_data.shape[0]

    features = []

    # LL-LP-RP-RL
    lr_eeg_data = np.zeros(shape=(data_num, eeg_data.shape[1], 4))
    for k, names in enumerate(CHAIN_FEATURES):
        for i in range(len(names) - 1):
            lr_eeg_data[..., k] += eeg_data[..., COLUMN_NAMES_INV[names[k + 1]]] - eeg_data[
                ..., COLUMN_NAMES_INV[names[k]]]

    s = time.perf_counter()
    if MEAN:
        mean_array = np.mean(lr_eeg_data, axis=1)
        features.append(mean_array)
        print(f"MEAN:time={time.perf_counter() - s:.4f}sec")

    s = time.perf_counter()
    if STD:
        std_array = np.std(lr_eeg_data, axis=1)
        features.append(std_array)
        print(f"STD:time={time.perf_counter() - s:.4f}sec")

    # skew_array = scipy.stats.skew(eeg_data, axis=1)
    # kurtosis_array = scipy.stats.kurtosis(eeg_data, axis=1)

    if STFT_SPEC:
        s = time.perf_counter()

        def stft(x):
            stft_spec = librosa.stft(y=x, hop_length=x.shape[-1] // 256,
                                     n_fft=255, win_length=128)
            rev = librosa.power_to_db(stft_spec, ref=np.max).astype(np.float32)

            return rev

        stft_spectrogram = np.zeros(shape=(data_num, 128, 257, 4))
        for data_index in tqdm.tqdm(range(data_num), desc="extract stft"):
            for k, names in enumerate(CHAIN_FEATURES):
                for i in range(len(names) - 1):
                    stft_spectrogram[data_index, ..., k] += stft(
                        eeg_data[data_index, ..., COLUMN_NAMES_INV[names[k + 1]]] - eeg_data[
                            data_index, ..., COLUMN_NAMES_INV[names[k]]])
        stft_spectrogram = normalize(stft_spectrogram, axis=(1, 2))

        # 60Hz まで 10Hz ずつ平均・標準偏差をとる
        mergin = 10
        for i in range(0, mergin*6, mergin):
            features.append(np.mean(stft_spectrogram[:, i:i + mergin], axis=(1, 2)))
            features.append(np.std(stft_spectrogram[:, i:i + mergin], axis=(1, 2)))
        del stft_spectrogram
        print(f"STFT:time={time.perf_counter() - s:.4f}sec")

    if MEL_SPEC:
        s = time.perf_counter()

        def mel_spec_func(x):
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=x.shape[-1] // 256,
                                                      n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)
            rev = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
            return rev

        mel_spectrogram = np.zeros(shape=(data_num, 128, 257, 4))
        for data_index in tqdm.tqdm(range(data_num), desc="extract mel"):
            for k, names in enumerate(CHAIN_FEATURES):
                for i in range(len(names) - 1):
                    mel_spectrogram[data_index, ..., k] += mel_spec_func(
                        eeg_data[data_index, ..., COLUMN_NAMES_INV[names[k + 1]]] - eeg_data[
                            data_index, ..., COLUMN_NAMES_INV[names[k]]])
        mel_spectrogram = normalize(mel_spectrogram, axis=(1, 2))

        # 60Hz まで 10Hz ずつ平均・標準偏差をとる
        mergin = 10
        for i in range(0, mergin*6, mergin):
            features.append(np.mean(mel_spectrogram[:, i:i + mergin], axis=(1, 2)))
            features.append(np.std(mel_spectrogram[:, i:i + mergin], axis=(1, 2)))
        del mel_spectrogram
        print(f"MEL:time={time.perf_counter() - s:.4f}sec")
    return np.concatenate(features, axis=1)

def main():
    root = Path(__file__).parents[3]
    import tqdm

    train = True
    config = XGBoostModelConfig(data_use_second=50, load_preprocess_data=True)



    if train:
        eegs_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_eegs")
        meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/train.csv"))
        meta_df = meta_df[meta_df["eeg_sub_id"] == 0].sort_values("eeg_id").reset_index(drop=True)[:100]
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
