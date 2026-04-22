import yaml
from src.data.ingestion import load_data
from src.features.build_features import create_preprocessor
from src.pipelines.pipeline import create_pipeline
from src.models.train import train_model

# carregar config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

df = load_data(config["data_path"])

df = df.dropna()

X = df.drop(columns=[config["target"], "customerID"])
y = df[config["target"]]

preprocessor = create_preprocessor(
    config["numerical_features"],
    config["categorical_features"]
)

pipeline = create_pipeline(preprocessor)

model = train_model(pipeline, X, y)

print("Treinamento finalizado 🚀")

import joblib
joblib.dump(model, "model.pkl")

from src.models.experiment import run_experiments

run_experiments(
    X,
    y,
    config["numerical_features"],
    config["categorical_features"]
)

from src.models.dim_experiment import run_dimensionality_experiments

run_dimensionality_experiments(X, y, preprocessor)
