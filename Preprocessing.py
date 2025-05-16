import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import joblib

# Getting the data
data = pd.read_csv(r'D:\Python Projects\Data_Analytics_Project\Project_code\water_pollution_disease.csv')

# Filling the blank values with interpolating datas
data['Water Treatment Method'] = data['Water Treatment Method'].interpolate("polynomial", order =2 , inplace= True)

# Creating a new column titled disease_spread - includes the total number of cases of disease spread in that region
data['disease_spread'] = data["Diarrheal Cases per 100,000 people"] + data["Cholera Cases per 100,000 people"] + data["Typhoid Cases per 100,000 people"]


bins = [0, 200, 500, np.inf]
labels = ['Low', 'Medium', 'High']
data['disease_burden_class'] = pd.cut(data['disease_spread'], bins=bins, labels=labels)


# Diving into categorical and numerical columns
excluding_col = [
    "disease_spread",
    "Diarrheal Cases per 100,000 people",
    "Cholera Cases per 100,000 people",
    "Typhoid Cases per 100,000 people"
    ]

catergorical_col= data.select_dtypes(include= ['object']).columns[0:]
numeric_col = data.select_dtypes(include = ['integer']).columns[0:]

# Eliminating the outliers
z_scores = np.abs(stats.zscore(data[numeric_col]))
data = data[(z_scores < 3).all(axis=1)]

x = pd.get_dummies(data.drop(columns=excluding_col), drop_first=True)
y = data['disease_spread']
dummy_variable = x.join(y)


# Finding the relationship 
# Linear relationship
corr = dummy_variable.corr()
top_corr_features = []
for feature, corr in corr.items():
    if feature != 'disease_spread' and len(top_corr_features) <= 5:
        top_corr_features.append(feature)

print(top_corr_features)


# Non Linear Relationship
mi = mutual_info_regression(x,y)
mi_series = pd.Series(mi, index = x.columns)
top_mi_features = mi_series.sort_values(ascending = False).head(5).index.tolist()
print(top_mi_features)

important_features = list(dict.fromkeys(top_mi_features + top_corr_features))
X = x[important_features]



# Saving the important features
joblib.dump(important_features,'important_features.pkl')

# Saving the changes in the original csv file
data.to_csv(r'D:\Python Projects\Data_Analytics_Project\Project_code\water_pollution_disease.csv',index = False)