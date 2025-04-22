import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("D:/python programs with data sets/Air Quality DataSet.csv")
# view the data
print("Data set: ")
print(df.head())
print()

# Basic Information
print("Basic info: ")
print(df.info())
print()

# Describe the Data
print("Dataset Description: ")
print(df.describe())
print()

# Find null Values
print("Total null Values: ")
print(df.isnull().sum())
print()

# Replace null Values
df.replace(np.nan,'0',inplace=True)

# Convert Date and Time to datetime
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)


# Set seaborn style
sns.set(style="whitegrid")

'''# 1. Line Chart: CO(GT) over time
plt.figure(figsize=(5, 5))
plt.plot(df['Datetime'], df['CO(GT)'], color='red')
plt.title('1. CO(GT) Concentration Over Time')
plt.xlabel('Datetime')
plt.ylabel('CO(GT)')
plt.tight_layout()
plt.show()'''

# 2. Pie Chart: Distribution by Air Quality Category
bins = [0, 100, 200, 300, 400, np.inf]
labels = ['Good', 'Moderate', 'Unhealthy for Sensitive', 'Unhealthy', 'Very Unhealthy']
df['AQ_Category'] = pd.cut(df['AirQualityIndex'], bins=bins, labels=labels)
pie_data = df['AQ_Category'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
plt.title('2. Air Quality Index Distribution')
plt.tight_layout()
plt.show()

# 3. Bar Chart: Average PM2.5 by Day of Week
pm25_day = df.groupby('DayOfWeek')['PM2.5'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=pm25_day.index, y=pm25_day.values, hue=pm25_day.index, palette="Blues_d", legend=False)

plt.title('3. Average PM2.5 by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average PM2.5')
plt.tight_layout()
plt.show()

# 4. Histogram: Distribution of NO2(GT)
plt.figure(figsize=(10, 6))
sns.histplot(df['NO2(GT)'], bins=30, kde=True, color="green")
plt.title('4. Distribution of NO2(GT)')
plt.xlabel('NO2(GT)')
plt.tight_layout()
plt.show()

# 5. Horizontal Bar Chart: Average O3(GT) by Hour
o3_hour = df.groupby('Hour')['O3(GT)'].mean()
plt.figure(figsize=(10, 8))
o3_hour.sort_values().plot(kind='barh', color="orange")
plt.title('5. Average O3(GT) by Hour')
plt.xlabel('Average O3(GT)')
plt.tight_layout()
plt.show()

# 6. Heatmap: Correlation between Pollutants
pollutants = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'O3(GT)', 'SO2(GT)', 'PM2.5', 'PM10']
corr_matrix = df[pollutants].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('6. Correlation between Pollutants')
plt.tight_layout()
plt.show()

# 7. Scatter Plot: NOx(GT) vs NO2(GT)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='NOx(GT)', y='NO2(GT)', data=df, color='purple')
plt.title('7. NOx(GT) vs NO2(GT)')
plt.xlabel('NOx(GT)')
plt.ylabel('NO2(GT)')
plt.tight_layout()
plt.show()

# 8. Box Plot: PM10 by Day of Week
plt.figure(figsize=(10, 6))
sns.boxplot(x='DayOfWeek', y='PM10', hue='DayOfWeek', data=df, palette='Set2', legend=False)

plt.title('8. PM10 Distribution by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('PM10')
plt.tight_layout()
plt.show()

# 9. Violin Plot: CO(GT) distribution by Hour
plt.figure(figsize=(12, 6))
sns.violinplot(x='Hour', y='CO(GT)', hue='Hour', data=df, palette='muted', legend=False)

plt.title('9. CO(GT) Distribution by Hour')
plt.xlabel('Hour')
plt.ylabel('CO(GT)')
plt.tight_layout()
plt.show()

# 10. Stacked Bar Chart: PM2.5 and PM10 by Day of Week
pm_by_day = df.groupby('DayOfWeek')[['PM2.5', 'PM10']].mean()
plt.figure(figsize=(10, 6))
plt.bar(pm_by_day.index, pm_by_day['PM2.5'], label='PM2.5')
plt.bar(pm_by_day.index, pm_by_day['PM10'], bottom=pm_by_day['PM2.5'], label='PM10')
plt.title('10. PM2.5 and PM10 by Day of Week (Stacked)')
plt.xlabel('Day of Week')
plt.ylabel('Pollution Levels')
plt.legend()
plt.tight_layout()
plt.show()
