import pandas as pd
import pycaret.classification as pc
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature, ModelSignature
from mlflow.types.schema import Schema, ColSpec
from sklearn.metrics import log_loss, f1_score, classification_report, confusion_matrix

mlflow.set_tracking_uri("sqlite:///mlruns.db")
#mlflow.set_registry_uri("sqlite:///mlruns.db")

experiment_name = 'ProjetoEngML'
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id = experiment.experiment_id


with mlflow.start_run(experiment_id=experiment_id, run_name = 'PipelineAplicacao'):

    model_uri = f"models:/modelo_kobe_shots@staging"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    kobe_shots_prod = pd.read_parquet('../data/raw/dataset_kobe_prod.parquet')
    kobe_shots_prod = kobe_shots_prod.dropna()
    vars_interesse = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
    var_alvo = 'shot_made_flag'
    #kobe_shots_prod = kobe_shots_prod[vars_interesse]
    
    pred_prod = loaded_model.predict(kobe_shots_prod[['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']])
    kobe_shots_prod['predict_score'] = pred_prod

    kobe_shots_prod.to_parquet('../data/processed/prediction_prod.parquet')
    mlflow.log_artifact('../data/processed/prediction_prod.parquet')

    mlflow.log_metrics({
        'log_loss_prod': log_loss(kobe_shots_prod[[var_alvo]], pred_prod),
        'f1_score_prod': f1_score(kobe_shots_prod[[var_alvo]], pred_prod)   
    })

mlflow.end_run()