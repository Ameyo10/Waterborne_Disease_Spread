import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,classification_report
from sklearn.decomposition import PCA

# Load data
important_features = joblib.load(r'D:\Python Projects\Data_Analytics_Project\Project_code\important_features.pkl')
data = pd.read_csv(r'D:\Python Projects\Data_Analytics_Project\Project_code\water_pollution_disease.csv')

# Prepare features
X_full = pd.get_dummies(data, drop_first=True)
X_aligned = X_full.reindex(columns=important_features, fill_value=0)
X_aligned = X_aligned.drop(columns=["Water Source Type_River", "Nitrate Level (mg/L)", "Year"])
x = X_aligned
y = data['disease_spread']

# Check feature correlations
print("Feature correlations with target:")
print(X_aligned.corrwith(y).sort_values())

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Applying PCA
n_features = x_train.shape[1]
max_pca_components = min(n_features, x_train.shape[0]) - 1  
pca_options = [0.85, 0.95] + [i for i in range(1, max_pca_components+1) if i < max_pca_components]

# Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('decomposition', PCA()),
    ('regressor', Ridge())
])

# Parameter grid (simplified to avoid failures)
param_grid = [
    {
        'decomposition__n_components': pca_options,
        'regressor': [RandomForestRegressor(random_state=42)],
        'regressor__n_estimators': [50, 100],
        'regressor__max_depth': [None, 5]
    }
]

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,  # Reduced from 15 for speed
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    error_score='raise'  # To see actual errors
)

# Fit the model
grid.fit(x_train, y_train)

# Evaluation
pred = grid.predict(x_test)
print("\n--- Regression Metrics ---")
print(f'R-squared: {r2_score(y_test, pred):.4f}')
print(f'MAE: {mean_absolute_error(y_test, pred):.4f}')
print(f'MSE: {mean_squared_error(y_test, pred):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.4f}')

print("\nBest parameters:", grid.best_params_)

joblib.dump(grid.best_estimator_, 'best_model.pkl')
# Feature importance (for tree-based models)
# if hasattr(grid.best_estimator_.named_steps['regressor'], 'feature_importances_'):
#     print("\nFeature importances:")
#     importances = grid.best_estimator_.named_steps['regressor'].feature_importances_
#     features = x.columns if grid.best_params_['decomposition'] is None else \
#                [f"PC{i+1}" for i in range(grid.best_params_['decomposition__n_components'])]
    # print(pd.Series(importances, index=features).sort_values(ascending=False))