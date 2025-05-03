import pdfplumber, re, os
import pandas as pd
from datetime import datetime

import json, os
import pandas as pd
import seaborn as sns
dfs = []
rawFiles = os.listdir('/workspaces/data/commute/output/raw/')
for cDate in rawFiles:
    print(cDate)
    with open(f'/workspaces/wikipedia/Data/raw/{cDate}', 'r') as file: 
        data2 = json.load(file)

    d = pd.DataFrame(data2['routes'])
    dfs.append(d)


with open(f'/workspaces/data/commute/output/raw/m_uw_2024_11_22_08_00.json', 'r') as file: 
        data2 = json.load(file)


d = pd.DataFrame(data2['routes'])
d['distance'] = d['localizedValues'].apply(lambda x: x['distance']['text'])
d['duration'] = d['localizedValues'].apply(lambda x: x['duration']['text'])
d['staticDuration'] = d['localizedValues'].apply(lambda x: x['staticDuration']['text'])


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrame with a datetime column and a continuous variable
data = {
    'datetime': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'number_of_cars': [i + (i % 5) * 2 for i in range(100)]
}
df = pd.DataFrame(data)

# Ensure the datetime column is in datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract features
df['day_of_week'] = df['datetime'].dt.dayofweek  # Monday=0, Sunday=6
df['day'] = df['datetime'].dt.day

# Summary statistics
print("Summary Statistics:")
print(df.describe())

# Plot number of cars by day of the week
plt.figure(figsize=(10, 6))
sns.boxplot(x='day_of_week', y='number_of_cars', data=df)
plt.xlabel('Day of the Week')
plt.ylabel('Number of Cars')
plt.title('Number of Cars by Day of the Week')
plt.show()

# Plot number of cars by day of the month
plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='number_of_cars', data=df)
plt.xlabel('Day of the Month')
plt.ylabel('Number of Cars')
plt.title('Number of Cars by Day of the Month')
plt.show()

# Correlation matrix
correlation_matrix = df.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrame with a datetime column and a continuous variable
data = {
    'datetime': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'number_of_cars': [i + (i % 5) * 2 for i in range(100)]
}
df = pd.DataFrame(data)

# Extract features
df['day_of_week'] = df['datetime'].dt.dayofweek  # Monday=0, Sunday=6
df['day'] = df['datetime'].dt.day

# Box plots for day of the week and day of the month
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(x='day_of_week', y='number_of_cars', data=df, ax=axes[0])
sns.boxplot(x='day', y='number_of_cars', data=df, ax=axes[1])
axes[0].set_title('Number of Cars by Day of the Week')
axes[0].set_xlabel('Day of the Week')
axes[0].set_ylabel('Number of Cars')
axes[1].set_title('Number of Cars by Day of the Month')
axes[1].set_xlabel('Day of the Month')
axes[1].set_ylabel('Number of Cars')

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()
