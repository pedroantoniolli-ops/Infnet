import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model(pipeline, X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("telco-churn")

    with mlflow.start_run():

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        report = classification_report(y_test, preds, output_dict=True)

        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("recall_churn", report["Yes"]["recall"])

        mlflow.sklearn.log_model(pipeline, "model")

    return pipeline