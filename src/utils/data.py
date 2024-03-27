"""
データ作成、前処理
"""

import pandas as pd
import cv2
import numpy as np
import pywt
import librosa

from tqdm import tqdm
from pathlib import Path
from scipy.signal import butter, lfilter



EEG_CHANNEL_LIST: list[list[str]] = [['Fp1','F7','T3','T5','O1'],
                                     ['Fp1','F3','C3','P3','O1'],
                                     ['Fp2','F8','T4','T6','O2'],
                                     ['Fp2','F4','C4','P4','O2']]
RAW_EEG_CHANNEL_LIST: list[str] = ['Fp1','T3','C3','O1','Fp2','C4','T4','O2']
TARGETS: list[str] = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
EEG_FREQUENCY: int = 200
EEG_STEP: int = int(50 * EEG_FREQUENCY)
EEG_SPEC_H: int = 128
EEG_SPEC_W: int = 256


def maddest(x: np.ndarray, axis: int=None)->np.ndarray:
    """平均絶対偏差

    Args:
        x (np.ndarray): _description_
        axis (int, optional): _description_. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    return np.mean(np.absolute(x - np.mean(x, axis)), axis)


def denoise(x: np.ndarray, wavelet: str="db1", level: int=1, axis: int=None)->np.ndarray:
    """waveletによるdenoise処理

    Args:
        x (np.ndarray): _description_
        wavelet (str, optional): _description_. Defaults to "db1".
        level (int, optional): _description_. Defaults to 1.
        axis (int, optional): _description_. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    coeff = pywt.wavedec(x, wavelet, mode="periodization")
    sigma = (1 / 0.6745) * maddest(coeff[-level], axis)

    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])

    return pywt.waverec(coeff, wavelet, mode="periodization")


def get_eeg_data(eeg_df: pd.DataFrame,
                 level: int, 
                 wavelet: str="haar"):
    
    raw_eeg = np.zeros((EEG_STEP, len(RAW_EEG_CHANNEL_LIST)))
    for i, ch in enumerate(RAW_EEG_CHANNEL_LIST):
        x = eeg_df[ch].values

        # 補完：すべてがnanの場合は0埋め
        m = np.nanmean(x)

        if not np.all(np.isnan(x)):
            x = np.nan_to_num(x, nan=m)
        else:
            x[:] = 0

        # denoise
        x = denoise(x, wavelet=wavelet, level=level)
        raw_eeg[..., i] = x

    return raw_eeg


def calc_mel_spectrogram(eeg_df: pd.DataFrame,
                         level: int, 
                         wavelet: str="haar", 
                         sr: int=200, 
                         n_fft: int=1024, 
                         fmin: int=0, 
                         fmax: int=20, 
                         win_length: int=128, 
                         **kwargs)->np.ndarray:
    """mel spectrogramを作成

    Args:
        eeg_df (pd.DataFrame): _description_

    Returns:
        np.ndarray: _description_
    """

    eeg_spec = np.zeros((EEG_SPEC_H, EEG_SPEC_W, len(EEG_CHANNEL_LIST)))
    for i, ch in enumerate(EEG_CHANNEL_LIST):
        for c in range(len(ch) - 1):
            x1 = eeg_df[ch[c]].values
            x2 = eeg_df[ch[c+1]].values

            # 補完：すべてがnanの場合は0埋め
            m1 = np.nanmean(x1)
            m2 = np.nanmean(x2)

            if not np.all(np.isnan(x1)):
                x1 = np.nan_to_num(x1, nan=m1)
            else:
                x1[:] = 0
            if not np.all(np.isnan(x2)):
                x2 = np.nan_to_num(x2, nan=m2)
            else:
                x2[:] = 0

            # denoise
            x = denoise(x1, wavelet=wavelet, level=level) - denoise(x2, wavelet=wavelet, level=level)
            mel_spec = librosa.feature.melspectrogram(y=x, 
                                                    sr=sr, 
                                                    hop_length=len(x)//EEG_SPEC_W,
                                                    n_fft=n_fft,
                                                    n_mels=EEG_SPEC_H,
                                                    fmin=fmin,
                                                    fmax=fmax,
                                                    win_length=win_length,
                                                    **kwargs
                                                    )
            mel_spec = np.clip(mel_spec, 1e-4, 1e+8)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:, :eeg_spec.shape[1]]
            eeg_spec[..., i] = mel_spec_db

        eeg_spec[..., i] /= 4.0

    return eeg_spec[::-1, ...]  # y軸反転


def separate_spec_data(spec_df: pd.DataFrame, spec_ch_dict: dict[str, list[str]])->np.ndarray:
    spec_data = np.zeros((EEG_SPEC_H, EEG_SPEC_W, len(spec_ch_dict)))

    for i, ch_col in enumerate(spec_ch_dict.values()):
        data = spec_df[ch_col].values  # (time, feature)
        data = data.transpose(1, 0) # (feature, time)
        # nan処理+clip
        m = np.nanmean(data)
        data[np.isnan(data)] = m
        data = np.clip(data, 1e-4, 1e+8)
        data = librosa.power_to_db(data, ref=np.max).astype(np.float32)
        data = data[::-1, :]    # y軸反転
        spec_data[..., i] = cv2.resize(data, (spec_data.shape[1], spec_data.shape[0])).astype(np.float32)

    return spec_data
    

def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


class DataCreator():
    def __init__(self):
        # raw eeg parameters
        self.raw_eeg_params = {
            "level":1,
            "wavelet": "haar",
        }

        # mel spectrogram parameters
        self.mel_params = {
            "level":1,
            "wavelet": "haar",
            "sr":200,
            "n_fft":1024,
            "fmin":0,
            "fmax":20,
            "win_length":128
        }


    def extract_unique_data(self, src_data_df: pd.DataFrame):
        # uniqueなeeg_idのdfを作成
        agg_dict = {
                "eeg_label_offset_seconds":["min", "max"],
                "spectrogram_id":"first",
                "spectrogram_label_offset_seconds":["min", "max"],
                "patient_id":"first",
                "seizure_vote":"sum",
                "lpd_vote":"sum",
                "gpd_vote":"sum",
                "lrda_vote":"sum",
                "grda_vote":"sum",
                "other_vote":"sum"
            }

        data_df = src_data_df.groupby("eeg_id").agg(agg_dict)
        data_df.columns = data_df.columns.map(lambda col: "_".join(col) if col[1] in ["min", "max"] else col[0])
        data_df = data_df.reset_index()

        return data_df
    

    def create_data(self, src_dir: Path, out_dir: Path, data_type_list: list[str] = ["raw_eeg", "eeg_spectrogram", "kaggle_spectrogram"]):
        
        src_data_df = pd.read_csv(src_dir.joinpath("train.csv"))
        data_df = self.extract_unique_data(src_data_df)
        src_eeg_dir = src_dir.joinpath("train_eegs")
        src_spec_dir = src_dir.joinpath("train_spectrograms")


        for data_name in data_type_list:
            if data_name not in ["raw_eeg", "eeg_spectrogram", "kaggle_spectrogram"]:
                raise ValueError()

        if "raw_eeg" in data_type_list:
            raw_eeg_dir = out_dir.joinpath("raw_eeg")
            raw_eeg_dir.mkdir(parents=True, exist_ok=True)

        if "eeg_spectrogram" in data_type_list:
            eeg_spec_dir = out_dir.joinpath("eeg_spectrogram")
            eeg_spec_dir.mkdir(parents=True, exist_ok=True)

        if "kaggle_spectrogram" in data_type_list:
            spec_dir = out_dir.joinpath("spectrogram")
            spec_dir.mkdir(parents=True, exist_ok=True)
            # spectrogram作成用にカラムを取得
            spec_ch_dict:dict[str, list[str]] = {"LL":[], "RL":[], "LP":[], "RP":[]}
            temp_path = list(src_spec_dir.glob("*.parquet"))[0]
            spec_ = pd.read_parquet(temp_path)
            for col in spec_.columns:
                for key in spec_ch_dict:
                    if key in col:
                        spec_ch_dict[key].append(col)
                        break
            del spec_


        data_list = []
        for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
            eeg_id = int(row["eeg_id"])
            spec_id = int(row["spectrogram_id"])
            eeg_mid_sec = int((row["eeg_label_offset_seconds_max"] - row["eeg_label_offset_seconds_min"])//2)
            spec_mid_sec = int((row["spectrogram_label_offset_seconds_max"] - row["spectrogram_label_offset_seconds_min"])//2)

            eeg_df = pd.read_parquet(src_eeg_dir.joinpath(f"{eeg_id}.parquet"))
            spec_df = pd.read_parquet(src_spec_dir.joinpath(f"{spec_id}.parquet"))

            # 必要な部分を切り取る
            eeg_start_idx = eeg_mid_sec * EEG_FREQUENCY
            eeg_df = eeg_df.iloc[eeg_start_idx:eeg_start_idx+EEG_STEP]

            if "raw_eeg" in data_type_list:
                raw_eeg = get_eeg_data(eeg_df, **self.raw_eeg_params)
                np.save(raw_eeg_dir.joinpath(f"{eeg_id}.npy"), raw_eeg)

            if "eeg_spectrogram" in data_type_list:
                eeg_spec = calc_mel_spectrogram(eeg_df, **self.mel_params)
                np.save(eeg_spec_dir.joinpath(f"{eeg_id}.npy"), eeg_spec)

            if "kaggle_spectrogram" in data_type_list:
                spec_df = spec_df.loc[(spec_df["time"]>=spec_mid_sec)&(spec_df["time"]<=spec_mid_sec+EEG_STEP)]
                spec = separate_spec_data(spec_df, spec_ch_dict)
                np.save(spec_dir.joinpath(f"{eeg_id}.npy"), spec) # specもeeg_idのファイル名とする
            
            data_list.append(row)

        use_data_df = pd.DataFrame(data_list)
        use_data_df.to_csv(out_dir.joinpath("train.csv"), index=False)


if __name__ == "__main__":
    import os
    data_root = Path(os.environ["kaggle_data_root"]).joinpath("hms-harmful-brain-activity-classification")

    data_creator = DataCreator()
    data_creator.create_data(src_dir=data_root, out_dir=Path("./outputs/test"))