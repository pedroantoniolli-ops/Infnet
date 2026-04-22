import pandas as pd
from scipy.stats import ks_2samp

def detect_drift(df_train, df_new, column):

    stat, p_value = ks_2samp(df_train[column], df_new[column])

    if p_value < 0.05:
        return f"Drift detectado em {column}"
    else:
        return f"Sem drift em {column}"
    
    detect_drift(df_train, df_new, "MonthlyCharges")

    mlflow.log_metric("new_data_accuracy", value)

    