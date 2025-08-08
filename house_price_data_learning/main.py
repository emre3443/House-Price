import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import load_data
from src.model import train_and_evaluate

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

X, y = load_data("data/house_prices.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "XGBoost": XGBRegressor(verbosity=0),
    "CatBoost": CatBoostRegressor(verbose=0)
}

print("Model Performansları (MSE):\n")
for name, model in models.items():
    mse = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    print(f"{name}: {mse}")