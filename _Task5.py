#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic accident data (replace with your actual dataset)
np.random.seed(42)
n_samples = 1000
hours = np.random.randint(0, 24, n_samples)
road_conditions = np.random.choice(['Clear', 'Rainy', 'Snowy'], n_samples)
accident_counts = np.random.poisson(10, n_samples)

# Create a DataFrame
df = pd.DataFrame({
    'Hour': hours,
    'Road_Conditions': road_conditions,
    'Accident_Counts': accident_counts
})

# EDA: Hourly accident counts
hourly_counts = df.groupby('Hour')['Accident_Counts'].mean()
plt.figure(figsize=(8, 4))
plt.plot(hourly_counts.index, hourly_counts.values, marker='o')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Accident Counts')
plt.title('Hourly Accident Counts')
plt.grid(True)
plt.show()

# EDA: Bar plot for road conditions
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Road_Conditions')
plt.xlabel('Road Conditions')
plt.ylabel('Accident Counts')
plt.title('Accident Counts by Road Conditions')
plt.show()

# Geospatial visualization (requires actual geographical data)
# You can use libraries like Folium to plot accident locations on a map.

# Regression analysis (linear regression example)
from sklearn.linear_model import LinearRegression

X = pd.get_dummies(df['Road_Conditions'], drop_first=True)
y = df['Accident_Counts']

model = LinearRegression()
model.fit(X, y)

print("Regression coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")


# In[ ]:




