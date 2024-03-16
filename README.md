# Harmful-Brain-Activity-Classification

## Install

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

## TODO

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

| model_framework |                                           src                                           | 5-CV Score | LB Score |  update   |
|:---------------:|:---------------------------------------------------------------------------------------:|:----------:|:--------:|:---------:|
|   External01    |           https://www.kaggle.com/datasets/crackle/hms-efficientnetb0-pt-ckpts           |            |   0.4    | 2024/3/15 |
|   External02    | https://www.kaggle.com/code/yunsuxiaozi/hms-baseline-resnet34d-512-512-training-5-folds |            |          | 2024/3/15 |
|   External03    |              https://www.kaggle.com/code/andreasbis/hms-inference-lb-0-41               |            |          | 2024/3/16 |

### Ensemble

|                   models                    | 5-CV Score | LB Score |  update   |
|:-------------------------------------------:|:----------:|:--------:|:---------:|
|      WaveNet, XGBoost, efficientnet_b0      |            |   0.49   | 2024/3/10 |
|          WaveNet, efficientnet_b0           |   0.7125   |   0.45   | 2024/2/23 |
| efficientnetb0_wavenet_resnetgru_external01 |            |   0.38   | 2024/2/23 |
