import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Initializing sample data
data = {
    'Name': ['Igor', 'Sasha', 'Shero', 'Julien', 'Alina', 'Marie', 'John', 'Peter', 'Anna', 'James'],
    'Age': [25, 30, 35, 40, 22, 28, 33, 38, 27, 31],
    'City': ['Kyiv', 'Paris', 'Lviv', 'Warsaw', 'Prague', 'Paris', 'New York', 'Berlin', 'London', 'Rome'],
    'Salary': [70000, 80000, 90000, 100000, 75000, 85000, 72000, 95000, 78000, 88000],
    'Experience': [2, 5, 8, 12, 1, 4, 7, 10, 3, 6]
}
df = pd.DataFrame(data)

# Display basic information and data types
print("DataFrame Info:")
print(df.info())
print("\nData Types:")
print(df.dtypes)

# Calculate advanced statistics
print("\nAdvanced Statistics:")
print(df.describe().T)

# Correlation analysis
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Z-score normalization
scaler = StandardScaler()
numerical_columns = ['Age', 'Salary', 'Experience']
df[['Age_normalized', 'Salary_normalized', 'Experience_normalized']] = scaler.fit_transform(df[numerical_columns])

# Outlier detection with IQR method
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Salary'] < lower_bound) | (df['Salary'] > upper_bound)]
print("\nOutliers in Salary:")
print(outliers)

# Advanced filtering
high_earners_by_city = df[df['Salary'] > df['Salary'].mean()].groupby('City').size().reset_index(name='Count')
print("\nHigh Earners by City:")
print(high_earners_by_city)

# Time series simulation
years = pd.date_range(start='2023', periods=5, freq='Y')
for name in df['Name']:
    initial_salary = df.loc[df['Name'] == name, 'Salary'].values[0]
    growth_rate = np.random.uniform(0.02, 0.08)  # Random growth rate between 2% and 8%
    salary_projection = [initial_salary * (1 + growth_rate) ** i for i in range(5)]
    df.loc[df['Name'] == name, [str(year.year) for year in years]] = salary_projection

print("\nSalary Projections:")
print(df)

# K-means clustering
X = df[numerical_columns]
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Visualizations
import matplotlib
matplotlib.use('Agg')
fig, axs = plt.subplots(3, 2, figsize=(15, 12))

# Age Distribution
age_counts = df['Age'].value_counts().sort_index()
axs[0, 0].bar(age_counts.index, age_counts.values, edgecolor='black', color='skyblue')
axs[0, 0].set_title('Age Distribution')
axs[0, 0].set_xlabel('Age')
axs[0, 0].set_ylabel('Count')
axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

# Set x-axis limits
axs[0, 0].set_xlim(df['Age'].min() - 1, df['Age'].max() + 1)

# Add value labels
for i, v in enumerate(age_counts.values):
    axs[0, 0].text(age_counts.index[i], v, str(v), ha='center', va='bottom')

# Rotate x-axis labels
axs[0, 0].set_xticks(age_counts.index)
axs[0, 0].set_xticklabels(age_counts.index, rotation=45)

# Salaries by City
df.boxplot(column='Salary', by='City', ax=axs[0, 1])
axs[0, 1].set_title('Salary Distribution by City')
axs[0, 1].set_ylabel('Salary')
plt.sca(axs[0, 1])
plt.xticks(rotation=45)

# Age vs Salary
axs[1, 0].scatter(df['Age'], df['Salary'])
axs[1, 0].set_title('Age vs Salary')
axs[1, 0].set_xlabel('Age')
axs[1, 0].set_ylabel('Salary')

# Heatmap of correlation matrix
im = axs[1, 1].imshow(correlation_matrix, cmap='coolwarm')
axs[1, 1].set_title('Correlation Heatmap')
plt.colorbar(im, ax=axs[1, 1])
axs[1, 1].set_xticks(range(len(correlation_matrix.columns)))
axs[1, 1].set_yticks(range(len(correlation_matrix.columns)))
axs[1, 1].set_xticklabels(correlation_matrix.columns, rotation=45)
axs[1, 1].set_yticklabels(correlation_matrix.columns)

# Parallel coordinates for clustering results
pd.plotting.parallel_coordinates(df[['Age', 'Salary', 'Experience', 'Cluster']], 'Cluster', ax=axs[2, 0])
axs[2, 0].set_title('Parallel Coordinates Plot of Clusters')

# Salary projection
for name in df['Name']:
    axs[2, 1].plot(years, df.loc[df['Name'] == name, [str(year.year) for year in years]].values[0], label=name)
axs[2, 1].set_title('Salary Projections Over Time')
axs[2, 1].set_xlabel('Year')
axs[2, 1].set_ylabel('Projected Salary')
axs[2, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')

plt.tight_layout()
plt.savefig('data_analysis_plot.png')

# Statistical questions
print("\nStatistical Tests:")
# Is there a significant difference in salaries between employees above and below 30?
young = df[df['Age'] <= 30]['Salary']
old = df[df['Age'] > 30]['Salary']
t_stat, p_value = stats.ttest_ind(young, old)
print(f"T-test for salary difference (age <= 30 vs > 30): t-statistic = {t_stat}, p-value = {p_value}")

# Is there a significant difference in salaries among cities?
f_stat, p_value = stats.f_oneway(*[group['Salary'].values for name, group in df.groupby('City')])
print(f"ANOVA test for salary difference among cities: F-statistic = {f_stat}, p-value = {p_value}")

# Export data to CSV
df.to_csv('processed_employee_data.csv', index=False)
print("\nProcessed data saved to 'processed_employee_data.csv'")
