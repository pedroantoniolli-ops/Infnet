import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.pipelines.pipeline import create_pipeline
from src.pipelines.dim_pipeline import (
    create_pipeline_pca,
    create_pipeline_lda
)


def run_dimensionality_experiments(X, y, preprocessor):

    mlflow.set_experiment("dimensionality-reduction")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipelines = {
        "baseline": create_pipeline(preprocessor),
        "pca": create_pipeline_pca(preprocessor),
        "lda": create_pipeline_lda(preprocessor)
    }

    for name, pipeline in pipelines.items():

        with mlflow.start_run(run_name=name):

            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_test)
            report = classification_report(y_test, preds, output_dict=True)

            mlflow.log_metric("accuracy", report["accuracy"])
            mlflow.log_metric("f1_churn", report["Yes"]["f1-score"])
            mlflow.log_metric("recall_churn", report["Yes"]["recall"])

            mlflow.sklearn.log_model(pipeline, name)

            print(f"{name} finalizado 🚀")