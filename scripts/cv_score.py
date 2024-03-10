from pathlib import Path
import pandas as pd
import numpy as np


root_dir = Path(__file__).parents[1]
result_dir = root_dir.joinpath("outputs/runner/spectrograms_nn/eeg_efficientnet_b0/20240303")


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
a = torch.Tensor([[0.,0.,1.]]) + 1e-15
b = torch.Tensor([[0.,0.,1.]])
z = kl_div(a.log(), b)

target = torch.Tensor(label_df.iloc[:, 1:-1].to_numpy())
input_ = torch.Tensor(pred_df.iloc[:, 1:].to_numpy())
s = kl_div((target + 1e-15).log(), input_)

ss = s.mean(dim=1).numpy()

temp_df = label_df.copy()
temp_df["kl_div"] = ss
temp_df = pd.concat([temp_df, pred_df], axis=1)
temp_df = temp_df.sort_values("kl_div")

# import matplotlib.pyplot as plt
# temp_df.hist("kl_div")
# plt.show()



from src.kaggle_score import kaggle_score
score = kaggle_score(label_df.copy().drop("fold", axis=1), pred_df.copy(), row_id_column_name="eeg_id")

print(score)
