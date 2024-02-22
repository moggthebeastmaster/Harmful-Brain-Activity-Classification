from pathlib import Path
import pandas as pd
import optuna

root = Path(__file__).parents[1]

if __name__ == '__main__':

    frame_work = "xgboost"
    model_name = "xgboost"
    date = "20240217"
    output_dir = root.joinpath("outputs", "optuna", frame_work, model_name, date)

    study_name = '-'.join([frame_work, model_name, date])
    storage = 'sqlite:///../optuna_results.db'
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True)

    print(study.best_value, study.best_params)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()