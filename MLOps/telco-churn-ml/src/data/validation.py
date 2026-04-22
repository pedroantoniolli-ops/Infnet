def check_missing(df):
    return df.isnull().sum()

def check_balance(df, target):
    return df[target].value_counts()