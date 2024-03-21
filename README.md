# Harmful-Brain-Activity-Classification

## Install
~~~
pip install -r requirements.txt
~~~

torch は以下を使用している。

~~~
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
~~~

## 使用方法

* 学習は train_cv.py を使う
* 推論は submission.py を使う
* 最適化にはtrain_optuna.py を使う。

## TODO

* Runner および SubmissionRunner の使用方法を作成する。
* 単一モデルで CV:0.7以下 , LB0.5 以下 程度のモデルを複数用意したい
  * 5 つくらいが目標
* アンサンブルで LB 0.3 を切ることが目標

## Results

### Single Models

|   model_framework   |              src              | 5-CV Score | LB Score |  update   |
|:-------------------:|:-----------------------------:|:----------:|:--------:|:---------:|
|       WaveNet       |     src.framework.eeg_nn      |   0.710    |   0.58   | 2024/2/15 |
|       XGBoost       |     src.framework.xgboost     |    0.93    |   0.8    | 2024/3/2  |
|   efficientnet_b0   | src.framework.spectrograms_nn |    0.69    |   0.5    | 2024/2/24 |
| eeg_efficientnet_b0 | src.framework.spectrograms_nn |    0.75    |   1.42   | 2024/3/3  |
|      ResnetGRU      |     src.framework.eeg_nn      |   0.672    |   0.44   | 2024/3/14 |

### External Model

| model_framework |                                                src                                                 | 5-CV Score | LB Score |  update   |
|:---------------:|:--------------------------------------------------------------------------------------------------:|:----------:|:--------:|:---------:|
|   External01    |                https://www.kaggle.com/datasets/crackle/hms-efficientnetb0-pt-ckpts                 |            |   0.4    | 2024/3/15 |
|   External02    |      https://www.kaggle.com/code/yunsuxiaozi/hms-baseline-resnet34d-512-512-training-5-folds       |            |   0.46   | 2024/3/15 |
|   External03    |                    https://www.kaggle.com/code/andreasbis/hms-inference-lb-0-41                    |            |   0.41   | 2024/3/16 |
|   External04    | https://www.kaggle.com/code/konstantinboyko/hms-resnet1d-gru-1-stage-inference-1-5-signal/notebook |            |   0.37   | 2024/3/16 |

### Blend

|                    models                    | 5-CV Score | LB Score |  update   |
|:--------------------------------------------:|:----------:|:--------:|:---------:|
|      WaveNet, XGBoost, efficientnet_b0       |            |   0.49   | 2024/3/10 |
|           WaveNet, efficientnet_b0           |   0.7125   |   0.45   | 2024/2/23 |
| efficientnetb0_wavenet_resnetgru_external01  |            |   0.38   | 2024/3/15 |
|                  external14                  |            |          | 2024/3/16 |
|                 externalall                  |            |          | 2024/3/16 |
| efficientnetb0_wavenet_resnetgru_externalall |            | time out | 2024/3/16 |
| efficientnetb0_wavenet_resnetgru_external134 |            | time out | 2024/3/17 |
| efficientnetb0_wavenet_resnetgru_external14  |            |   0.36   | 2024/3/17 |


### Ensemble

|                   models                    | 5-CV Score | LB Score |  update   |
|:-------------------------------------------:|:----------:|:--------:|:---------:|
| efficientnetb0_wavenet_resnetgru_external14 |            |   0.33   | 2024/3/20 |
|      efficientnetb0_wavenet_resnetgru       |            |   0.56   | 2024/3/21 |
|  efficientnetb0_wavenet_resnetgru_xgboost   |            |   0.66   | 2024/3/21 |
