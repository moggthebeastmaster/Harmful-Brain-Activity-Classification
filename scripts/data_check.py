from pathlib import Path
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

root_dir = Path(__file__).parents[1]
data_dir = root_dir.joinpath("data/hms-harmful-brain-activity-classification")


train_meta_df = pd.read_csv(data_dir.joinpath("train.csv"))
print(train_meta_df)

from src.framework.eeg_nn.data.eeg_dataset import TARGETS_COLUMNS

label_df = train_meta_df[TARGETS_COLUMNS]

z = label_df.to_numpy()==0
zz = pd.Series(z.sum(axis=1)).value_counts()

sum_array = label_df.to_numpy().sum(axis=1, keepdims=True)
pd.DataFrame(sum_array).hist(bins=100, log=True)
plt.show()

z:pd.DataFrame = label_df / sum_array


zz = z.sort_values(TARGETS_COLUMNS[0])

z.hist(bins=100, log=True)

import matplotlib.pyplot as plt
plt.show()

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

