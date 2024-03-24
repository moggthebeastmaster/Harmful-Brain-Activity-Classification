import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
import torchaudio
import pyarrow.parquet as pq

root = Path(__file__).parents[4]
sys.path.append(str(root))
from src.framework.spectrograms_nn.config.efficient_net_config import EfficientNetConfig

TARGETS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
TARGETS_INV = {x: y for y, x in TARGETS.items()}
TARGETS_COLUMNS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
ONE_DATA_TIME = 600
FREQUENCY = 1 / 2
COLUMN_NAMES_ALL = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8',
                    'T4',
                    'T6', 'O2', 'EKG']
COLUMN_NAMES_ALL_INV = {key: index for index, key in enumerate(COLUMN_NAMES_ALL)}

COLUMN_NAMES = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4',
                'T6', 'O2']

COLUMN_NAMES = ['LL', 'LP', 'RP', 'RR']
CHAIN_FEATURES = [['Fp1', 'F7', 'T3', 'T5', 'O1'],
                  ['Fp1', 'F3', 'C3', 'P3', 'O1'],
                  ['Fp2', 'F8', 'T4', 'T6', 'O2'],
                  ['Fp2', 'F4', 'C4', 'P4', 'O2']]


class SpectrogramsDataset(torch.utils.data.Dataset):
    def __init__(self, meta_df: pd.DataFrame, spectrograms_dir: Path, config: EfficientNetConfig, with_label=False,
                 train_mode=False):
        self.meta_df = meta_df
        self.spectrograms_dir = spectrograms_dir
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
        self.train_mode = train_mode

        self.spectrogram = torchaudio.transforms.Spectrogram()
        self.masking1 = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.config.frequency_mask_range)
        self.masking2 = torchaudio.transforms.TimeMasking(time_mask_param=self.config.time_mask_range)

        if self.with_label:
            self.meta_label = self.meta_df.expert_consensus
            y = self.meta_df[TARGETS_COLUMNS].values
            self.meta_label_prob = y / y.sum(axis=1, keepdims=True)

    def __len__(self):
        return len(self.meta_eeg_id)

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray, int, float]:
        eeg_id = self.meta_eeg_id[index]
        # draw
        x, y = self.draw(index)

        if self.train_mode:
            x = self.masking1(torch.from_numpy(x))
            x = self.masking2(x).numpy()

        # cast and to 3ch
        x = x.astype(np.float32)
        x = np.stack([x] * 3, axis=0)

        if False:
            import matplotlib.pyplot as plt
            plt.imshow((((x[0].T - x.min()) / (x.max() - x.min())) * 255).astype(np.uint8), cmap='viridis', aspect='auto',
                       origin='lower')
            plt.show()

        return x, y, eeg_id, self.meta_eeg_label_offset_seconds[index]

    def draw(self, index):
        spectrogram_id = self.meta_spectrogram_id[index]
        spectrogram_path = self.spectrograms_dir.joinpath(f"{spectrogram_id}.parquet")

        # 対象データの中心時間。 基本的には offset + 25sec
        target_center_time = self.meta_eeg_label_offset_seconds[index] + ONE_DATA_TIME // 2
        # 対象データの切り出し範囲
        target_start_point = int((target_center_time - self.config.data_use_second // 2 + 1) * FREQUENCY)
        target_end_point = int((target_center_time + self.config.data_use_second // 2 + 1) * FREQUENCY)

        # 読み込み
        spectrogram_table = pq.read_table(spectrogram_path, memory_map=True)

        # kaggle の accellerate が GPU か　CPU で、なぜか読み込んだshape が異なる点に注意
        x = np.asarray(spectrogram_table)
        if x.shape[0] == 401:
            x = x.T

        # 対象データの切り出し
        x = x[target_start_point:target_end_point, 1:]

        mean_values = np.nanmean(x, axis=0)
        nan_indices = np.isnan(x)
        x[nan_indices] = np.take(mean_values, nan_indices.nonzero()[1])

        # log をとって標準化
        x = np.clip(x, np.exp(-6), np.exp(10))
        x = self.normalize(np.log(x))

        # ラベル情報
        y = self.meta_label_prob[index] if self.with_label else np.zeros(shape=(len(TARGETS, )), dtype=float)

        return x, y

    @staticmethod
    def normalize(eeg_array: np.array, eps=1e-5):
        """
        eeg_array.shape = [time, ch]
        """
        eeg_array = (eeg_array - np.nanmean(eeg_array, axis=(0, 1), keepdims=True)) / (
                np.nanstd(eeg_array, axis=(0, 1), keepdims=True) + eps)
        return np.nan_to_num(eeg_array, nan=0)


def main():
    train = True

    config = EfficientNetConfig(data_use_second=600)
    if train:
        spectrograms_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_spectrograms")
        meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/train.csv"))[82:1000]
    else:
        spectrograms_dir = root.joinpath("data/hms-harmful-brain-activity-classification/test_spectrograms")
        meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/test.csv"))

    dataset = SpectrogramsDataset(meta_df, spectrograms_dir, config, with_label=train, train_mode=train)

    for data_one in tqdm.tqdm(dataset):
        pass
        # print(data_one)


if __name__ == '__main__':
    import line_profiler
    import atexit

    profile = line_profiler.LineProfiler()
    profile.add_module(SpectrogramsDataset)
    # profile.runcall(main)
    main()

    atexit.register(profile.print_stats)
