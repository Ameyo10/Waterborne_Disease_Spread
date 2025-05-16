import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'D:\Python Projects\Data_Analytics_Project\Project_code\water_pollution_disease.csv')

def visualizing_correlation():

    excluding_col = [
        "disease_spread",
        "Diarrheal Cases per 100,000 people",
        "Cholera Cases per 100,000 people",
        "Typhoid Cases per 100,000 people"
        ]

    x = pd.get_dummies(data.drop(columns = excluding_col ), drop_first=True)
    y = data['disease_spread']
    dummy_variable = x.join(y)

    corr_matrix = dummy_variable.corr()
    selected_col = corr_matrix['disease_spread'].drop('disease_spread').sort_values(ascending=False)

    sns.heatmap(selected_col.to_frame(), annot = True, fmt=".2g")
    plt.show()

def more_disease():
    data['Place'] = data['Country'] + "'s " + data['Region']
    avg = data['disease_spread'].mean()
    regions = {}

    for i,row in data.iterrows():
        if row['disease_spread'] > avg:
            regions[row['Place']] = row['disease_spread']
            print(row['Place'], row['disease_spread'],end = '\t')
    
    return regions

def hist_plots():
    for col in ['pH Level','Nitrate Level (mg/L)', 'Lead Concentration (µg/L)', 'Rainfall (mm per year)',
                 'Access to Clean Water (% of Population)', 'GDP per Capita (USD)']:
        plt.figure()
        sns.histplot(data[col], kde=True)
        plt.title(f"Distribution of {col.title()}")
        plt.show()

def scatter_plots():
    for col in ['Access to Clean Water (% of Population)']:
            
        plt.figure()
        sns.scatterplot(x= col, y='disease_spread',hue = data['disease_burden_class'] ,data=data)
        plt.title(f"{col} vs Disease Spread")
        plt.show()

def line_visual():
    plt.figure()
    sns.lineplot(x='Year', y='disease_spread', data=data)
    plt.title("Disease Spread Over Years")
    plt.show()

def avg_disease_spread_vs_country():
    # Group by country and compute mean
    vs = data.groupby('Country')['disease_spread'].mean()

    # Print each country's average
    for country, avg in vs.items():
        print(f"The average disease spread in {country} is {avg:.2f}.", end = '\n')

import matplotlib.pyplot as plt

def plot_disease_burden_pie(data):
    columns = ['disease_burden_class', 'Water Source']
    for col in columns:
        if col in data.columns:
            class_counts = data[col].value_counts()

            # Pick color palette dynamically
            color_palette = sns.color_palette("pastel", len(class_counts))

            # Plotting the pie chart
            plt.figure(figsize=(6, 6))
            plt.pie(
                class_counts,
                labels=class_counts.index,
                autopct='%1.1f%%',
                startangle=140,
                colors=color_palette
            )
            plt.title(f"Distribution of {col}")
            plt.axis('equal')
            plt.show()
        else:
            print(f"Column '{col}' not found in the dataset.")

# visualizing_correlation()
# more_disease()
# hist_plots()
# scatter_plots()
plot_disease_burden_pie(data)
# line_visual()
# avg_disease_spread_vs_country()