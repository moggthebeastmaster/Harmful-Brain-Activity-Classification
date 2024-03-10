"""

https://www.kaggle.com/code/tmnwatanab/hms-resnet1d-gru-train-1-5-dataset/edit
"""


import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
import pyarrow.parquet as pq
from scipy.signal import butter, lfilter

root = Path(__file__).parents[4]
sys.path.append(str(root))
from src.framework.eeg_nn.config import EEGNeuralNetConfig

TARGETS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
TARGETS_INV = {x: y for y, x in TARGETS.items()}
TARGETS_COLUMNS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
FREQUENCY = 200
ONE_DATA_TIME = 50
COLUMN_NAMES_ALL = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4',
                'T6', 'O2', 'EKG']
COLUMN_NAMES_ALL_INV = {key: index for index, key in enumerate(COLUMN_NAMES_ALL)}

COLUMN_NAMES = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]

MAP_FEATURES = [
    ("Fp1", "T3"),
    ("T3", "O1"),
    ("Fp1", "C3"),
    ("C3", "O1"),
    ("Fp2", "C4"),
    ("C4", "O2"),
    ("Fp2", "T4"),
    ("T4", "O2"),
]
NAME2INDEX = {val: key for key, val in enumerate(COLUMN_NAMES)}

class EEGResnetGRUDataset(torch.utils.data.Dataset):
    def __init__(self, meta_df: pd.DataFrame, eegs_dir: Path, config: EEGNeuralNetConfig, with_label=False,
                 train_mode=False):
        self.meta_df = meta_df
        self.eegs_dir = eegs_dir
        self.config = config
        self.with_label = with_label

        self.meta_eeg_id = self.meta_df.eeg_id.values
        self.meta_eeg_sub_id = self.meta_df.eeg_sub_id.values if with_label else np.zeros_like(self.meta_eeg_id)
        self.meta_eeg_label_offset_seconds = self.meta_df.eeg_label_offset_seconds.values if "eeg_label_offset_seconds" in self.meta_df.columns else np.zeros_like(
            self.meta_eeg_id)

        self.meta_spectrogram_id = self.meta_df.spectrogram_id.values
        self.meta_spectrogram_sub_id = self.meta_df.spectrogram_sub_id.values if with_label else np.zeros_like(
            self.meta_spectrogram_id)
        self.meta_spectrogram_label_offset_seconds = self.meta_df.spectrogram_label_offset_seconds.values if with_label else np.zeros_like(
            self.meta_spectrogram_id)

        self.meta_patient_id = self.meta_df.patient_id.values

        self.bandpass_filter = ButterBandpassFilter(lowcut=0.5, highcut=20, fs=FREQUENCY, order=2)
        self.lowpass_filter = ButterLowpassFilter(order=2)

        self.mix_up_alpha = self.config.mix_up_alpha

        self.train_mode = train_mode

        if self.with_label:
            self.meta_label = self.meta_df.expert_consensus
            y = self.meta_df[TARGETS_COLUMNS].values
            self.meta_label_prob = y / y.sum(axis=1, keepdims=True)

    def __len__(self):
        return len(self.meta_eeg_id)

    def __getitem__(self, index):
        eeg_id = self.meta_eeg_id[index]
        # draw
        x, y = self.draw(index)
        # x = self.normalize(x)
        if self.train_mode:
            if (self.mix_up_alpha > 0.):

                random_index = torch.randint(0, len(self.meta_eeg_id), (1,)).detach().numpy()[0]
                l = np.random.beta(self.mix_up_alpha, self.mix_up_alpha, 1)[0]  # FIX: 出来ればtorch random で実装したい
                x_mix, y_mix = self.draw(random_index)
                # x_mix= self.normalize(x_mix)

                x = l * x + (1 - l) * x_mix
                y = l * y + (1 - l) * y_mix

            # random bandpass filter
            if torch.rand(size=(1, ))[0] <= 0.1:
                low_cut = torch.randint(10, 20, size=(1, )).numpy()[0]
                high_cut = low_cut + 5.
                filter = ButterBandpassFilter(lowcut=low_cut, highcut=high_cut, fs=FREQUENCY, order=2)
                x = filter(x)

        x = np.clip(x, -1024, 1024)
        x = np.nan_to_num(x, nan=0) / 32.0
        x = self.lowpass_filter(x)
        # x = self.normalize(x)
        if False:
            import matplotlib.pyplot as plt
            plt.plot(x[:500, -3:])
            plt.show()


        # cast
        x = x.astype(np.float32)

        return x, y, eeg_id, self.meta_eeg_label_offset_seconds[index]

    def draw(self, index):
        eeg_id = self.meta_eeg_id[index]
        eeg_path = self.eegs_dir.joinpath(f"{eeg_id}.parquet")
        # 対象データの中心時間。 基本的には offset + 25sec
        target_center_time = self.meta_eeg_label_offset_seconds[index] + ONE_DATA_TIME // 2
        # 対象データの切り出し範囲
        target_start_point = int(FREQUENCY * (target_center_time - self.config.data_use_second // 2))
        target_end_point = int(FREQUENCY * (target_center_time + self.config.data_use_second // 2))

        # 読み込み
        eeg_table = pq.read_table(eeg_path, memory_map=True, columns=COLUMN_NAMES)

        # kaggle の accellerate が GPU か　CPU で、なぜか読み込んだshape が異なる点に注意
        x = np.asarray(eeg_table)
        if x.shape[0] == 20:
            x = x.T

        # 対象データの切り出し
        x = x[target_start_point:target_end_point]

        mean_values = np.nanmean(x, axis=0)
        nan_indices = np.isnan(x)
        x[nan_indices] = np.take(mean_values, nan_indices.nonzero()[1])

        diff_x = np.zeros(shape=(x.shape[0], len(MAP_FEATURES)))
        for i, (a, b) in enumerate(MAP_FEATURES):
            diff_x[:, i] = x[:, NAME2INDEX[a]] - x[:, NAME2INDEX[b]]

        x = np.concatenate([x, diff_x], axis=1)

        #x = np.clip(x, -1024, 1024)
        #x = np.nan_to_num(x, nan=0) / 32.0
        #x = self.lowpass_filter(x)

        # ラベル情報
        y = self.meta_label_prob[index] if self.with_label else np.zeros(shape=(len(TARGETS, )), dtype=float)

        return x, y



    @staticmethod
    def normalize(eeg_array: np.array, eps=1e-5):
        """
        eeg_array.shape = [time, ch]
        """
        eeg_array = (eeg_array - np.nanmean(eeg_array, axis=0, keepdims=True)) / (
                    np.nanstd(eeg_array, axis=0, keepdims=True) + eps)
        return np.nan_to_num(eeg_array, nan=0)

def _butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype="band")

class ButterBandpassFilter:
    def __init__(self, lowcut, highcut, fs, order=5):
        b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
        self.b = b
        self.a = a

    def __call__(self, data):
        filtered_data = lfilter(self.b, self.a, data, axis=0)

        if False:
            import matplotlib.pyplot as plt
            plt.plot(data[:500, :1])
            plt.plot(filtered_data[:500, :1])
            plt.show()

        return filtered_data


class ButterLowpassFilter:
    def __init__(self, cutoff_freq=20, sampling_rate=FREQUENCY, order=4):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        self.b = b
        self.a = a

    def __call__(self, data):
        filtered_data = lfilter(self.b, self.a, data, axis=0)

        if False:
            import matplotlib.pyplot as plt
            plt.plot(data[:500, :1])
            plt.plot(filtered_data[:500, :1])
            plt.show()

        return filtered_data

def main():
    train = True

    config = EEGNeuralNetConfig(data_use_second=50,
                                mix_up_alpha=0.2)
    if train:
        eegs_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_eegs")
        meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/train.csv"))[:1000]
    else:
        meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/test.csv"))
        eegs_dir = root.joinpath("data/hms-harmful-brain-activity-classification/test_eegs")
        eegs_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_eegs")
        meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/train.csv"))[:1000]

    dataset = EEGResnetGRUDataset(meta_df, eegs_dir, config, with_label=train, train_mode=train)

    for data_one in tqdm.tqdm(dataset):
        pass
        # print(data_one)


if __name__ == '__main__':
    import line_profiler
    import atexit

    profile = line_profiler.LineProfiler()
    profile.add_module(EEGResnetGRUDataset)
    # profile.runcall(main)
    main()

    atexit.register(profile.print_stats)
