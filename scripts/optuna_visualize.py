from pathlib import Path
import pandas as pd
import optuna
import sqlite3

root = Path(__file__).parents[1]

if __name__ == '__main__':
    frame_work = "spectrograms_nn"
    model_name = "eeg_efficientnet_b0"
    date = "20240316"
    output_dir = root.joinpath("outputs", "optuna", frame_work, model_name, date)


    db_path = r"C:\work\HMS\optuna_results.db"
    # db_path = r"C:\Users\kenap\OneDrive\デスクトップ\optuna_results.db"


    study_name = '-'.join([frame_work, model_name, date])
    storage = f'sqlite:///{db_path}'
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True)

    print(study.best_value, study.best_params)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()