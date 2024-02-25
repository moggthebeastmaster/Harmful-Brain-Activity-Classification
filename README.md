# Harmful-Brain-Activity-Classification

## Install

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

## TODO



## Results

### Single Models

| model_framework |              src              |       5-CV Score        | LB Score |  update   |
|:---------------:|:-----------------------------:|:-----------------------:|:--------:|:---------:|
|     WaveNet     |     src.framework.eeg_nn      |          0.710          |   0.58   | 2024/2/15 |
|     XGBoost     |     src.framework.xgboost     |          0.840          |   0.9    | 2024/2/18 |
| efficientnet_b0 | src.framework.spectrograms_nn |          0.715          |          | 2024/2/24 |


### Ensemble



|      models      | 5-CV Score | LB Score |  update   |
|:----------------:|:----------:|:--------:|:---------:|
| WaveNet, EGBoost |   0.775    |   1.02   | 2024/2/23 |
