# 🏠 Ev Fiyat Tahmini Projesi

Bu projede, evin bazı temel özelliklerine göre satış fiyatı tahmin edilmektedir. Üç farklı model karşılaştırılmıştır:

- Linear Regression
- XGBoost Regressor
- CatBoost Regressor

## Kullanılan Özellikler
- GrLivArea (m2 cinsinden yaşam alanı)
- YearBuilt (İnşa yılı)
- OverallQual (Genel kalite puanı)
- GarageCars (Garaj kapasitesi)

## Sonuçlar (örnek)
- Linear Regression: 2445761380.12 (MSE)
- XGBoost: 1354829140.57
- CatBoost: 1290841579.99

## Kullanılan Kütüphaneler
- pandas, scikit-learn, xgboost, catboost, matplotlib

## Nasıl Çalıştırılır
```bash
pip install -r requirements.txt
python main.py