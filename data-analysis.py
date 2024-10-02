import pandas as pd
import matplotlib.pyplot as plt

# Sample DataFrame creation
data = {
    'Name': ['Igor', 'Sasha', 'Shero', 'Julien', 'Alina', 'Marie', 'John', 'Peter', 'Anna', 'James'],
    'Age': [25, 30, 35, 40, 22, 28, 33, 38, 27, 31],
    'City': ['Kyiv', 'Paris', 'Lviv', 'Warsaw', 'Prague', 'Paris', 'New York', 'Berlin', 'London', 'Rome'],
    'Salary': [70000, 80000, 90000, 100000, 75000, 85000, 72000, 95000, 78000, 88000]
}
df = pd.DataFrame(data)

# Display basic information
print("DataFrame Info:")
print(df.info())

# Calculate basic statistics
average_salary = df['Salary'].mean()
median_salary = df['Salary'].median()
min_salary = df['Salary'].min()
max_salary = df['Salary'].max()
salary_std = df['Salary'].std()

print(f"\nAverage Salary: {average_salary}")
print(f"Median Salary: {median_salary}")
print(f"Min Salary: {min_salary}")
print(f"Max Salary: {max_salary}")
print(f"Salary Standard Deviation: {salary_std}")

# Filter DataFrame by Age > 30
filtered_df_age = df[df['Age'] > 30]
print("\nFiltered DataFrame (Age > 30):")
print(filtered_df_age)

# Filter DataFrame by Salary > 80000
filtered_df_salary = df[df['Salary'] > 80000]
print("\nFiltered DataFrame (Salary > 80000):")
print(filtered_df_salary)

# Group by City and calculate average salary
grouped_city = df.groupby('City')['Salary'].mean().reset_index()
print("\nAverage Salary by City:")
print(grouped_city)

# Sorting by Salary
sorted_df = df.sort_values(by='Salary', ascending=False)
print("\nDataFrame Sorted by Salary:")
print(sorted_df)

# Visualizations
plt.figure(figsize=(10, 6))

# Histogram of Ages
plt.subplot(2, 2, 1)
plt.hist(df['Age'], bins=5, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Bar plot of average salary by city
plt.subplot(2, 2, 2)
plt.bar(grouped_city['City'], grouped_city['Salary'], color='skyblue')
plt.title('Average Salary by City')
plt.xlabel('City')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)

# Box plot of Salaries
plt.subplot(2, 2, 3)
plt.boxplot(df['Salary'])
plt.title('Salary Distribution')
plt.ylabel('Salary')

# Scatter plot of Age vs Salary
plt.subplot(2, 2, 4)
plt.scatter(df['Age'], df['Salary'], color='red')
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')

plt.tight_layout()
plt.show()
