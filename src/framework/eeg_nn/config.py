from dataclasses import dataclass


@dataclass
class EEGNeuralNetConfig:
    # 20240214_2 のベストパラメータを使用

    model_framework: str = "WaveNet"
    data_use_second: int = 50
    batch_size: int = 2 ** 7
    accumulate_grad_batches: int = 2 ** 0  # この回数だけロスを蓄積してパラメータを更新する。
    learning_rate: float = 0.00042675084055210485
    weight_decay: float = 0.6523374845104841
    warmup_steps_ratio: float = 0.1
    max_epoch: int = 10
    num_worker: int = 0
    mix_up_alpha: float = 0.14878275160655596
    early_stop: bool = True

    drop_out: float = 0.11787224522185014
    num_base_channels: int = 53
