import pandas as pd
import numpy as np

# Load the data
location_df = pd.read_csv('dataset/location_2021.csv')
cases_train_df = pd.read_csv('dataset/cases_2021_train.csv')

# Handle missing values
cases_train_df['province'] = cases_train_df['province'].fillna('Unknown')
location_df['Province_State'] = location_df['Province_State'].fillna('Unknown')

location_df['Lat'] = location_df['Lat'].fillna(location_df['Lat'].mean())
location_df['Long_'] = location_df['Long_'].fillna(location_df['Long_'].mean())

# Convert 'age' to a numeric type, forcing non-numeric values to NaN
cases_train_df['age'] = pd.to_numeric(cases_train_df['age'], errors='coerce')

# Calculate the median age, excluding NaN values
median_age = cases_train_df['age'].median(skipna=True)

# Fill missing values for 'age' with the median age
cases_train_df['age'] = cases_train_df['age'].fillna(median_age)

# Get the most common value (the mode) for 'sex', which returns a Series
most_common_sex = cases_train_df['sex'].mode()

# If there is at least one mode, use the first one. Otherwise, fill with 'Unknown'
most_common_sex = most_common_sex[0] if not most_common_sex.empty else 'Unknown'

# Fill missing values for 'sex' with the most common sex
cases_train_df['sex'] = cases_train_df['sex'].fillna(most_common_sex)

# Combine datasets
combined_df = pd.merge(
    cases_train_df,
    location_df,
    left_on=['country', 'province'],
    right_on=['Country_Region', 'Province_State'],
    how='inner'
)

# Calculate Expected_Mortality_Rate
combined_df['Deaths'] = combined_df['Deaths'].fillna(0)
combined_df['Confirmed'] = combined_df['Confirmed'].fillna(0)
combined_df['Expected_Mortality_Rate'] = combined_df.apply(
    lambda row: (row['Deaths'] / row['Confirmed']) if row['Confirmed'] > 0 else 0,
    axis=1
)

print("\nMissing values in combined dataset:\n", combined_df.isnull().sum())

# Select a subset
subset_df = combined_df.copy()

# Check the data after handling missing values
print(subset_df.head())
