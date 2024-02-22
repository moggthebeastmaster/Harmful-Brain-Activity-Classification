from pathlib import Path
import pandas as pd
import tqdm

root_dir = Path(__file__).parents[1]
data_dir = root_dir.joinpath("data/hms-harmful-brain-activity-classification")


train_meta_df = pd.read_csv(data_dir.joinpath("train.csv"))
print(train_meta_df)


train_eeg_paths = list(data_dir.joinpath("train_eegs").glob("*"))
for p in tqdm.tqdm(train_eeg_paths):
    df = pd.read_parquet(p)
    eeg_id = p.stem

    meta_df = train_meta_df[train_meta_df["eeg_id"]==int(eeg_id)]
    print(eeg_id, df.shape, meta_df.shape)
    break

train_spectra_paths = list(data_dir.joinpath("train_spectrograms").glob("*"))
for p in tqdm.tqdm(train_spectra_paths):
    df = pd.read_parquet(p)
    print(df)
    break

