from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

TARGETS_COLUMNS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
root_dir = Path(__file__).parents[1]

def get_scored_df(result_dir):

    pred_list = []
    label_list = []
    for k in range(5):
        inner_predict_df = pd.read_csv(result_dir / f"fold_{k}" / "predicts.csv", index_col=0)
        inner_label_df = pd.read_csv(result_dir / f"fold_{k}" / "label.csv", index_col=0)
        inner_label_df["fold"] = k
        pred_list.append(inner_predict_df)
        label_list.append(inner_label_df)

    pred_df = pd.concat(pred_list).reset_index(drop=True)
    label_df = pd.concat(label_list).reset_index(drop=True)

    import torch
    kl_div = torch.nn.KLDivLoss(reduction="none")
    target = torch.Tensor(label_df.iloc[:, 1:-1].to_numpy())
    input_ = torch.Tensor(pred_df.iloc[:, 1:].to_numpy())
    score_torch = kl_div((input_ + 1e-15).log(), target)

    score_torch_sum = score_torch.sum(dim=1).numpy()

    temp_df = label_df.copy()
    temp_df["kl_div"] = score_torch_sum
    # temp_df = pd.concat([temp_df, pred_df], axis=1)
    temp_df = temp_df.merge(pred_df, on="eeg_id", suffixes=(None, "_pred"))

    print(score_torch_sum)

    return temp_df


result_dir = root_dir.joinpath("outputs/runner/spectrograms_nn/efficientnet_b0/20240223")
score_df_1 = get_scored_df(result_dir)

result_dir = root_dir.joinpath("outputs/runner/eeg_nn/resnet_gru/20240314_no_early")
score_df_2 = get_scored_df(result_dir)

result_dir = root_dir.joinpath("outputs/runner/nn_model/wave_net/20240215")
score_df_3 = get_scored_df(result_dir)


th = 1
bad_df_1_num = (score_df_1["kl_div"]>th).sum()
bad_df_2_num = (score_df_2["kl_div"]>th).sum()
bad_df_3_num = (score_df_3["kl_div"]>th).sum()
bad_df_mean = score_df_1[((score_df_1["kl_div"] + score_df_2["kl_div"] + score_df_3["kl_div"])/3 > th)]

bad_df_mean.to_csv("bad_train_20240317.csv")
bad_eed_id = bad_df_mean["eeg_id"].astype(int).to_numpy()
train_df = pd.read_csv(root_dir.joinpath("data/hms-harmful-brain-activity-classification/train.csv"))
t_df = train_df[train_df["eeg_id"].map(lambda s: s in bad_eed_id)]
t_df.to_csv("bad_train_20240317.csv", index=False)



th = 0.3
good_df_mean = score_df_1[((score_df_1["kl_div"] + score_df_2["kl_div"] + score_df_3["kl_div"])/3 < th)]

all_sum = score_df_1[TARGETS_COLUMNS].sum(axis=0)
good_sum = good_df_mean[TARGETS_COLUMNS].sum(axis=0)
bad_sum = bad_df_mean[TARGETS_COLUMNS].sum(axis=0)

sum_df = pd.concat([all_sum, good_sum, bad_sum], axis=1)
sum_df.columns = ["all", "good", "bad"]



sum_df = sum_df/ sum_df.sum(axis=0)

weight_sample = 1/sum_df["all"]
weight_sample = weight_sample / weight_sample.sum()
print(weight_sample.to_numpy())

sum_df.plot.bar()
plt.show()








bad_df = score_df_1[((score_df_1["kl_div"]>th) & (score_df_2["kl_div"]>th)) & (score_df_3["kl_div"]>th)]

zzzz = score_df_2["kl_div"][score_df_2["kl_div"]<th].mean()
zzzzz = score_df_2["kl_div"].mean()
print(zzzz,zzzzz)

score_df_1["kl_div"].hist(log=True, bins=100)
score_df_2["kl_div"].hist(log=True, bins=100)
plt.show()
# .sort_values("kl_div")

