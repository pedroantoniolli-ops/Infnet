from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def create_pipeline(preprocessor, model):
    return Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])