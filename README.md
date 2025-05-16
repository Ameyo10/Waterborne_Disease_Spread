# 💧 Waterborne Disease Risk Estimation using Machine Learning

This project estimates the likelihood of a person being affected by waterborne diseases based on environmental, sanitation, water quality, and socioeconomic features. It uses real-world inspired data to predict disease burden and categorize regions into risk levels.

---

## 📂 Project Structure

```
├── data
│   └── water_pollution_disease.csv           # Cleaned dataset
│
├── preprocessing.py                          # Data cleaning, feature engineering
├── model_training.py                         # Model selection and hyperparameter tuning
├── visualize_results.py                      # Visualizations and insights
├── predict_new_data.py                       # Predicting on new/unseen data
├── important_features.pkl                    # Saved features selected from training
├── best_model.pkl                            # Trained model saved with joblib
├── README.md                                 # Project documentation
```

---

## 🔍 Problem Statement

Waterborne diseases like cholera, typhoid, and diarrhea remain major threats in regions with inadequate sanitation and clean water access. This project aims to:

* Predict disease burden per region (regression)
* Classify risk levels: **Low**, **Medium**, **High**

---

## 🧠 Key Features Used

* Water quality metrics: pH, turbidity, nitrate, lead
* Access to clean water, sanitation, rainfall, temperature
* GDP per capita, healthcare access, population density

---

## ✅ Methods Used

* **Data Cleaning & Interpolation** for missing values
* **Outlier Detection** using Z-score filtering
* **Feature Selection** using correlation and mutual information
* **Model Training** with:

  * Random Forest Regressor
  * Ridge Regression
  * Gradient Boosting
* **PCA** for dimensionality reduction
* **GridSearchCV** for hyperparameter tuning
* **Risk Classification** based on predicted case thresholds

---

## 📈 Evaluation Metrics

* R² Score
* MAE / MSE / RMSE
* Classification Accuracy (when binned)

---

## 📊 Visualizations

* Correlation heatmap
* Histograms of key water and health indicators
* Disease spread over years
* Regional risk pie charts

---

## 🛠 How to Use

```bash
# Step 1: Preprocess the dataset
python preprocessing.py

# Step 2: Train the model
python model_training.py

# Step 3: Visualize insights
python visualize_results.py

# Step 4: Make predictions on new data
python predict_new_data.py
```

---
