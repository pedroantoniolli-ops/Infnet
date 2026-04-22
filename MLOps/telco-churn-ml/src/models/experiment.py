import mlflow
import mlflow.sklearn

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.pipelines.pipeline import create_pipeline
from src.features.build_features import create_preprocessor


def run_experiments(X, y, num_cols, cat_cols):

    mlflow.set_experiment("telco-churn-experiments")

    models = {
        "logistic": (
            LogisticRegression(max_iter=1000),
            {"model__C": [0.1, 1, 10]}
        ),
        "tree": (
            DecisionTreeClassifier(),
            {"model__max_depth": [3, 5, 10]}
        ),
        "rf": (
            RandomForestClassifier(),
            {
                "model__n_estimators": [50, 100],
                "model__max_depth": [5, 10]
            }
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for name, (model, params) in models.items():

        preprocessor = create_preprocessor(num_cols, cat_cols)
        pipeline = create_pipeline(preprocessor, model)

        grid = GridSearchCV(
            pipeline,
            params,
            cv=5,
            scoring="f1",
            n_jobs=-1
        )

        with mlflow.start_run(run_name=name):

            grid.fit(X_train, y_train)

            preds = grid.predict(X_test)
            report = classification_report(y_test, preds, output_dict=True)

            # métricas principais
            mlflow.log_metric("accuracy", report["accuracy"])
            mlflow.log_metric("f1_churn", report["Yes"]["f1-score"])
            mlflow.log_metric("recall_churn", report["Yes"]["recall"])

            # parâmetros
            mlflow.log_params(grid.best_params_)

            # modelo
            mlflow.sklearn.log_model(grid.best_estimator_, name)

            print(f"{name} concluído 🚀")