import numpy as np
from torch.utils.data import Dataset
import sys
from pathlib import Path
import pandas as pd


root = Path(__file__).parents[4]
sys.path.append(str(root))

from src.framework.external_01.config import External01Config


USE_KAGGLE_SPECTROGRAMS = True
USE_EEG_SPECTROGRAMS = True
TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

TARS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
TARS2 = {x: y for y, x in TARS.items()}

class External01Dataset(Dataset):
    def __init__(self,
                 meta_df: pd.DataFrame,
                 eegs_dir:Path,
                 spectrograms_dir: Path,
                 config: External01Config,
                 with_label=False,
                 train_mode=False,
                 ):

        assert not with_label
        assert not train_mode

        # READ ALL EEG SPECTROGRAMS
        self.eeg_indexes = meta_df.eeg_id.unique()
        eeg_specs = {}
        for eeg_id in self.eeg_indexes:
            # CREATE SPECTROGRAM FROM EEG PARQUET
            img = spectrogram_from_eeg(eegs_dir / f"{eeg_id}.parquet")
            eeg_specs[eeg_id] = img

        # READ ALL SPECTROGRAMS
        spec_indexes = meta_df.spectrogram_id.unique()
        specs = {}
        for spec_id in spec_indexes:
            tmp = pd.read_parquet(spectrograms_dir / f"{spec_id}.parquet")
            specs[spec_id] = tmp.iloc[:, 1:].values


        meta_df_inner = meta_df.rename({'spectrogram_id': 'spec_id'}, axis=1)
        self.eeg_dataset = EEGDataset(meta_df_inner,
                                      augment=False,
                                      mode="test",
                                      eeg_specs=eeg_specs,
                                      specs=specs
                                      )


    def __len__(self):
        return len(self.eeg_dataset)

    def __getitem__(self, index):
        return self.eeg_dataset[index]

    def __getitems__(self, indexes):
        eeg_id = self.eeg_indexes[indexes]
        x = self.eeg_dataset.__getitems__(indexes)
        return list(zip(x, eeg_id))


class EEGDataset(Dataset):

    def __init__(self, data, augment=False, mode='train', specs=None, eeg_specs=None):
        self.data = data
        self.augment = augment
        self.mode = mode
        self.specs = specs
        self.eeg_specs = eeg_specs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.__getitems__([index])

    def __getitems__(self, indices):
        X, y = self._generate_data(indices)
        return X

    def _generate_data(self, indexes):
        X = np.zeros((len(indexes), 128, 256, 8), dtype='float32')
        y = np.zeros((len(indexes), 6), dtype='float32')
        img = np.ones((128, 256), dtype='float32')

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            if self.mode == 'test':
                r = 0
            else:
                r = int((row['min'] + row['max']) // 4)

            for k in range(4):
                # EXTRACT 300 ROWS OF SPECTROGRAM
                img = self.specs[row.spec_id][r:r + 300, k * 100:(k + 1) * 100].T

                # LOG TRANSFORM SPECTROGRAM
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)

                # STANDARDIZE PER IMAGE
                ep = 1e-6
                m = np.nanmean(img.flatten())
                s = np.nanstd(img.flatten())
                img = (img - m) / (s + ep)
                img = np.nan_to_num(img, nan=0.0)

                # CROP TO 256 TIME STEPS
                X[j, 14:-14, :, k] = img[:, 22:-22] / 2.0

            # EEG SPECTROGRAMS
            img = self.eeg_specs[row.eeg_id]
            X[j, :, :, 4:] = img

            if self.mode != 'test':
                y[j,] = row[TARGETS]

        return X, y


import pywt, librosa

USE_WAVELET = None

NAMES = ['LL', 'LP', 'RP', 'RR']

FEATS = [['Fp1', 'F7', 'T3', 'T5', 'O1'],
         ['Fp1', 'F3', 'C3', 'P3', 'O1'],
         ['Fp2', 'F8', 'T4', 'T6', 'O2'],
         ['Fp2', 'F4', 'C4', 'P4', 'O2']]


# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet='haar', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret = pywt.waverec(coeff, wavelet, mode='per')

    return ret

import matplotlib.pyplot as plt

def spectrogram_from_eeg(parquet_path, display=False):
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle:middle + 10_000]

    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128, 256, 4), dtype='float32')

    if display: plt.figure(figsize=(10, 7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):

            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            # DENOISE
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x) // 256,
                                                      n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

            # LOG TRANSFORM
            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:, :width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k] += mel_spec_db

        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:, :, k] /= 4.0

    return img