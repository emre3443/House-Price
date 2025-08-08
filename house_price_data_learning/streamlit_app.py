import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ev Fiyat Tahmini", layout="centered")

st.title("🏠 Ev Fiyat Tahmini")
st.markdown("### Linear Regression, CatBoost ve XGBoost ile karşılaştırmalı tahmin")

grlivarea = st.slider("Yaşam Alanı (m²)", min_value=20, max_value=500, value=150)
yearbuilt = st.slider("İnşa Yılı", min_value=1900, max_value=2025, value=2000)
overallqual = st.slider("Genel Kalite (1-10)", min_value=1, max_value=10, value=5)
garagecars = st.slider("Garaj Kapasitesi (Araç)", min_value=0, max_value=4, value=1)

input_df = pd.DataFrame({
    "GrLivArea": [grlivarea],
    "YearBuilt": [yearbuilt],
    "OverallQual": [overallqual],
    "GarageCars": [garagecars]
})

df = pd.read_csv("data/data_house_price.csv")
df = df[["GrLivArea", "YearBuilt", "OverallQual", "GarageCars", "SalePrice"]].dropna()

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

models = {
    "Linear Regression": LinearRegression(),
    "XGBoost": XGBRegressor(verbosity=0),
    "CatBoost": CatBoostRegressor(verbose=0)
}

results = {}
for name, model in models.items():
    model.fit(X, y)
    prediction = model.predict(input_df)[0]
    results[name] = int(prediction)

st.markdown("## 🔍 Tahmin Edilen Fiyatlar")
for name, pred in results.items():
    st.write(f"**{name}:** ${pred:,.0f}")


st.markdown("### 📊 Model Tahmin Karşılaştırması")
fig, ax = plt.subplots()
ax.bar(results.keys(), results.values(), color=["skyblue", "orange", "green"])
ax.set_ylabel("Tahmini Fiyat ($)")
st.pyplot(fig)

st.markdown("---")
st.caption("✨ Geliştirici: Mustafa Emre Gök(https://www.linkedin.com/in/mustafaemregok/")

results = {}
for name, model in models.items():
    model.fit(X, y)
    prediction = model.predict(input_df)[0]
    results[name] = int(prediction)

st.markdown("## 🔍 Tahmin Edilen Fiyatlar")
for name, pred in results.items():
    st.write(f"**{name}:** ${pred:,.0f}")

tahminler = np.array(list(results.values()))
ortalama = np.mean(tahminler)
std_sapma = np.std(tahminler)

st.markdown(f"### 📊 Tahminlerin Ortalaması: ${ortalama:,.0f}")
st.markdown(f"### 📊 Tahminlerin Standart Sapması: ${std_sapma:,.0f}")


st.markdown("### 📊 Model Tahmin Karşılaştırması")
fig, ax = plt.subplots()
ax.bar(results.keys(), results.values(), color=["skyblue", "orange", "green"])
ax.set_ylabel("Tahmini Fiyat ($)")
st.pyplot(fig)