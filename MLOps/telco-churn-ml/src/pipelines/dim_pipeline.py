# Pipeline com PCA

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def create_pipeline_pca(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("pca", PCA(n_components=10)),
        ("model", RandomForestClassifier())
    ])

# Pipeline com LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def create_pipeline_lda(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("lda", LinearDiscriminantAnalysis()),
        ("model", RandomForestClassifier())
    ])

