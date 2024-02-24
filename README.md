# Harmful-Brain-Activity-Classification

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
