import joblib
import pandas as pd

important_features = joblib.load(r'D:\Python Projects\Data_Analytics_Project\Project_code\important_features.pkl')
best_model = joblib.load(r'D:\Python Projects\Data_Analytics_Project\Project_code\best_model.pkl')
data = pd.read_csv(r'D:\Python Projects\Data_Analytics_Project\Project_code\waterborne_disease_test_data.csv')

X_full = pd.get_dummies(data, drop_first=True)
X_aligned = X_full.reindex(columns=important_features, fill_value=0)
X_aligned = X_aligned.drop(columns=["Water Source Type_River", "Nitrate Level (mg/L)", "Year"])
X = X_aligned


predictions = best_model.predict(X)

chances = []
for i in predictions:
    chances.append((i/100000)*100)

for i in chances:
    print(f'{i:.2f}', end = '\t')