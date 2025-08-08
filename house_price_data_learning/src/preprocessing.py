import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df = df[["GrLivArea", "YearBuilt", "OverallQual", "GarageCars", "SalePrice"]]
    df = df.dropna()
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    return X, y