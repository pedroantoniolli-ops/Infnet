import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    # Corrigir tipo
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df