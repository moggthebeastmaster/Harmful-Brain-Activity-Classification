import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
import torchaudio
import pyarrow.parquet as pq
import librosa
from PIL import Image
from scipy.signal import butter, lfilter

root = Path(__file__).parents[4]
sys.path.append(str(root))
from src.framework.spectrograms_nn.config.efficient_net_config import EfficientNetConfig

TARGETS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
TARGETS_INV = {x: y for y, x in TARGETS.items()}
TARGETS_COLUMNS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
ONE_DATA_TIME = 600
FREQUENCY = 1 / 2

EEG_ONE_DATA_TIME = 50
EEG_FREQUENCY = 200


COLUMN_NAMES_ALL = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8',
                    'T4',
                    'T6', 'O2', 'EKG']
COLUMN_NAMES_ALL_INV = {key: index for index, key in enumerate(COLUMN_NAMES_ALL)}

COLUMN_NAMES = ['LL', 'LP', 'RP', 'RR']
CHAIN_FEATURES = [['Fp1', 'F7', 'T3', 'T5', 'O1'],
                  ['Fp1', 'F3', 'C3', 'P3', 'O1'],
                  ['Fp2', 'F4', 'C4', 'P4', 'O2'],
                  ['Fp2', 'F8', 'T4', 'T6', 'O2']]


def stft(x):
    x = butter_lowpass_filter(x)
    stft_spec = librosa.stft(y=x,
                             hop_length=x.shape[-1] // 256,
                             n_fft=1024,
                             win_length=128,
                             )
    # rev = librosa.power_to_db(stft_spec[:128], ref=np.max).astype(np.float32)
    S, phase = librosa.magphase(stft_spec)
    S = S[:128]

    return S


def mel_spec_func(x):
    mel_spec = librosa.feature.melspectrogram(y=x,
                                              sr=200,
                                              hop_length=x.shape[-1] // 256,
                                              n_fft=1024,
                                              n_mels=128,
                                              fmin=0,
                                              fmax=20,
                                              win_length=128)
    rev = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
    return rev


class SpectrogramsEEGDataset(torch.utils.data.Dataset):
    def __init__(self, meta_df: pd.DataFrame, eegs_dir:Path, spectrograms_dir: Path, config: EfficientNetConfig, with_label=False,
                 train_mode=False):
        self.meta_df = meta_df
        self.eegs_dir = eegs_dir
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

        if self.config.mix_up_alpha > 0:
            self.beta = torch.distributions.Beta(self.config.mix_up_alpha, self.config.mix_up_alpha)

        if self.with_label:
            self.meta_label = self.meta_df.expert_consensus
            y = self.meta_df[TARGETS_COLUMNS].values
            self.meta_label_prob = y / y.sum(axis=1, keepdims=True)

    def __len__(self):
        return len(self.meta_eeg_id)

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray, int, float]:
        eeg_id = self.meta_eeg_id[index]
        # draw
        x1, x2, x3, y = self.draw(index)

        if self.train_mode:
            if self.config.mix_up_alpha > 0.:
                x1_, x2_, x3_, y_ = self.draw(torch.randint(0, len(self), size=(1, ))[0])
                l = self.beta.sample().numpy()
                x1 = l * x1 + (1-l) * x1_
                x2 = l * x2 + (1-l) * x2_
                x3 = l * x3 + (1-l) * x3_
                y = l * y + (1-l) * y_

            x1 = self.masking1(torch.from_numpy(x1))
            x1 = self.masking2(x1).numpy()
            x2 = self.masking1(torch.from_numpy(x2))
            x2 = self.masking2(x2).numpy()
            x3 = self.masking1(torch.from_numpy(x3))
            x3 = self.masking2(x3).numpy()

        # cast and to 3ch
        x = np.stack([x1, x2, x3], axis=0)

        if False:
            import matplotlib.pyplot as plt

            x_show = np.concatenate([x1, x2, x3], axis=0)
            plt.imshow((((x_show.T - x_show.min()) / (x_show.max() - x_show.min())) * 255).astype(np.uint8), cmap='viridis', aspect='auto',
                       origin='lower')
            plt.show()

        return x, y, eeg_id, self.meta_eeg_label_offset_seconds[index]

    def draw(self, index):

        ## kaggle spectrogram
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
        x_kaggle_spectrograms = self.normalize(np.log(x))

        if False:
            import matplotlib.pyplot as plt
            z = (x_kaggle_spectrograms - x_kaggle_spectrograms.min()) / (x_kaggle_spectrograms.max() - x_kaggle_spectrograms.min())
            plt.imshow(z.T)
            plt.show()

        ## eeg spectrogram
        eeg_id = self.meta_eeg_id[index]
        eeg_path = self.eegs_dir.joinpath(f"{eeg_id}.parquet")
        # 対象データの中心時間。 基本的には offset + 25sec
        target_center_time = self.meta_eeg_label_offset_seconds[index] + EEG_ONE_DATA_TIME // 2
        # 対象データの切り出し範囲
        target_start_point = int(EEG_FREQUENCY * (target_center_time - EEG_ONE_DATA_TIME // 2))
        target_end_point = int(EEG_FREQUENCY * (target_center_time + EEG_ONE_DATA_TIME // 2))

        # 読み込み
        eeg_table = pq.read_table(eeg_path, memory_map=True, columns=COLUMN_NAMES_ALL)

        # kaggle の accellerate が GPU か　CPU で、なぜか読み込んだshape が異なる点に注意
        x = np.asarray(eeg_table)
        if x.shape[0] == 20:
            x = x.T

        # 対象データの切り出し
        x = x[target_start_point:target_end_point]

        mean_values = np.nanmean(x, axis=0)
        nan_indices = np.isnan(x)
        x[nan_indices] = np.take(mean_values, nan_indices.nonzero()[1])

        # MEL_SPECTROGRAM をとる
        x_chain = np.zeros(shape=(128, 257, len(COLUMN_NAMES)))
        for k, name in enumerate(COLUMN_NAMES):
            for inner_k in range(len(CHAIN_FEATURES[k]) - 1):
                x_chain[..., k] += mel_spec_func(x[:, COLUMN_NAMES_ALL_INV[CHAIN_FEATURES[k][inner_k]]] \
                                 - x[:, COLUMN_NAMES_ALL_INV[CHAIN_FEATURES[k][inner_k+1]]])
        x_chain = x_chain/4
        """
        # MEL_SPECTROGRAM をとる
        x_chain = np.zeros(shape=(len(COLUMN_NAMES), 10000))
        for k, name in enumerate(COLUMN_NAMES):
            for inner_k in range(len(CHAIN_FEATURES[k]) - 1):
                x_chain[k] += x[:, COLUMN_NAMES_ALL_INV[CHAIN_FEATURES[k][inner_k]]] \
                                 - x[:, COLUMN_NAMES_ALL_INV[CHAIN_FEATURES[k][inner_k+1]]]
        x_chain = mel_spec_func(x_chain/4)
        x_chain = np.transpose(x_chain, (1,2,0))
        """

        mean_values = np.nanmean(x_chain, axis=0)
        nan_indices = np.isnan(x_chain)
        x_chain[nan_indices] = np.take(mean_values, nan_indices.nonzero()[1])

        x_eeg_spectrograms = x_chain.copy()
        x_eeg_spectrograms = self.normalize(x_eeg_spectrograms, axis=(0,1))
        x_eeg_spectrograms = np.stack([x_eeg_spectrograms[..., i] for i in range(4)], axis=0)
        x_eeg_spectrograms = Image.fromarray(np.concatenate([s for s in x_eeg_spectrograms], axis=0))
        x_eeg_spectrograms = np.asarray(x_eeg_spectrograms.resize((300, 400))).T


        # STFT をとる
        x_chain = np.zeros(shape=(128, 257, len(COLUMN_NAMES)))
        for k, name in enumerate(COLUMN_NAMES):
            for inner_k in range(len(CHAIN_FEATURES[k]) - 1):
                x_chain[..., k] += stft(x[:, COLUMN_NAMES_ALL_INV[CHAIN_FEATURES[k][inner_k]]] \
                                 - x[:, COLUMN_NAMES_ALL_INV[CHAIN_FEATURES[k][inner_k+1]]])
        x_chain = x_chain/4

        mean_values = np.nanmean(x_chain, axis=0)
        nan_indices = np.isnan(x_chain)
        x_chain[nan_indices] = np.take(mean_values, nan_indices.nonzero()[1])

        x_eeg_spectrograms_stft = x_chain.copy()
        x_eeg_spectrograms_stft = self.normalize(x_eeg_spectrograms_stft, axis=(0,1))
        x_eeg_spectrograms_stft = np.stack([x_eeg_spectrograms_stft[..., i] for i in range(4)], axis=0)
        x_eeg_spectrograms_stft = Image.fromarray(np.concatenate([s for s in x_eeg_spectrograms_stft], axis=0))
        x_eeg_spectrograms_stft = np.asarray(x_eeg_spectrograms_stft.resize((300, 400))).T

        if False:
            import matplotlib.pyplot as plt
            z = (x_eeg_spectrograms - x_eeg_spectrograms.min()) / (x_eeg_spectrograms.max() - x_eeg_spectrograms.min())
            plt.imshow(z.T)
            plt.show()


        # ラベル情報
        y = self.meta_label_prob[index] if self.with_label else np.zeros(shape=(len(TARGETS, )), dtype=float)

        return x_eeg_spectrograms,x_eeg_spectrograms_stft, x_kaggle_spectrograms, y

    @staticmethod
    def normalize(eeg_array: np.array, eps=1e-5, axis=(0, 1)):
        """
        eeg_array.shape = [time, ch]
        """
        eeg_array = (eeg_array - np.nanmean(eeg_array, axis=axis, keepdims=True)) / (
                np.nanstd(eeg_array, axis=axis, keepdims=True) + eps)
        return np.nan_to_num(eeg_array, nan=0)


def butter_lowpass_filter(
        data, cutoff_freq=20, sampling_rate=200, order=4
):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data

def main():
    train = True

    config = EfficientNetConfig(data_use_second=600, mix_up_alpha=0., time_mask_range=0, frequency_mask_range=0)
    if train:
        eegs_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_eegs")
        spectrograms_dir = root.joinpath("data/hms-harmful-brain-activity-classification/train_spectrograms")
        meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/train.csv"))[:1000]
    else:
        eegs_dir = root.joinpath("data/hms-harmful-brain-activity-classification/test_eegs")
        spectrograms_dir = root.joinpath("data/hms-harmful-brain-activity-classification/test_spectrograms")
        meta_df = pd.read_csv(root.joinpath("data/hms-harmful-brain-activity-classification/test.csv"))

    dataset = SpectrogramsEEGDataset(meta_df, eegs_dir, spectrograms_dir, config, with_label=train, train_mode=train)

    for data_one in tqdm.tqdm(dataset):
        pass
        # print(data_one)




if __name__ == '__main__':
    import line_profiler
    import atexit

    profile = line_profiler.LineProfiler()
    profile.add_module(SpectrogramsEEGDataset)
    # profile.runcall(main)
    main()

    atexit.register(profile.print_stats)
