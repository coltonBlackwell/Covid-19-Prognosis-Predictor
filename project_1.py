import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, classification_report, \
    confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

locations_df = pd.read_csv('dataset/location_2021.csv')
cases_train_df = pd.read_csv('dataset/cases_2021_train.csv')
cases_test_df = pd.read_csv('dataset/cases_2021_test.csv')
countries_continents_df = pd.read_csv('dataset/Countries by continents.csv')
population_df = pd.read_csv('dataset/world_population.csv')

location_countries = set(locations_df['Country_Region'].unique())
population_countries = set(population_df['Country/Territory'].unique())

in_location_not_population = location_countries - population_countries
print("In 'location_df' but not in 'population_df':", in_location_not_population)

in_population_not_location = population_countries - location_countries
print("In 'population_df' but not in 'location_df':", in_population_not_location)

# outcome_counts_age = cases_train_df['age'].value_counts().sort_values(ascending=True)
outcome_counts_age = cases_train_df['age'].sort_values(ascending=True)
print("outcome_counts_age")
print(outcome_counts_age.head(50))

non_nan_count = cases_train_df['age'].notnull().sum()
hyphen_count = cases_train_df['age'].astype(str).str.contains('-').sum()

print("hyphen_count")
print(non_nan_count, hyphen_count)

# Q1
country_name_mapping = {
    "Ivory Coast": "Cote d'Ivoire",
    "DR Congo": "Congo (Kinshasa)",
    "Republic of the Congo": "Congo (Brazzaville)",
    "South Korea": "Korea, South",
    "United States": "US",
    "Cape Verde": "Cabo Verde",
    "Myanmar": "Burma",
    "Czech Republic": "Czechia",
    "Taiwan": "Taiwan*",
    "Vatican City": "Holy See",
    "Palestine": "West Bank and Gaza"
}

# Apply mapping to population_df
population_df['Country/Territory'] = population_df['Country/Territory'].replace(country_name_mapping)

# Merge population data with COVID-19 data from location_df
# location_with_pop_df = locations_df.merge(population_df, left_on='Country_Region', right_on='Country/Territory', how='left')

country_confirmed_sum = locations_df.groupby('Country_Region')['Confirmed'].sum().reset_index()
location_with_pop_df = country_confirmed_sum.merge(population_df, left_on='Country_Region',
                                                   right_on='Country/Territory', how='left')
location_with_pop_df['Confirmed_per_country'] = location_with_pop_df['Confirmed'] / location_with_pop_df[
    '2022 Population']

print("location_with_pop_df.groupby('Country_Region')['Confirmed'].sum().sort_values(ascending=False)")
# print(location_with_pop_df.groupby('Country_Region')['Confirmed'].sum().sort_values(ascending=False))

# confirmed, deaths cases by country
country_stats = locations_df.groupby('Country_Region')[['Confirmed', 'Deaths']].sum().reset_index()
top_20_countries_by_confirmed = country_stats.sort_values(by='Confirmed', ascending=False).head(20)

# print(top_20_countries_by_confirmed)

# confirmed cases by continents
continent_confirmed_sum = location_with_pop_df.groupby('Continent')['Confirmed'].sum().reset_index()

# print(continent_confirmed_sum)

# Calculate data availability by continent using total confirmed cases.
continent_confirmed = location_with_pop_df.groupby('Continent')['Confirmed'].sum().sort_values(ascending=False)

# Visualize data availability by continent with a bar plot
plt.figure(figsize=(12, 8))
sns.barplot(x=continent_confirmed.index, y=continent_confirmed.values, palette="coolwarm",
            hue=continent_confirmed.index, dodge=False)
plt.title('Data Availability by Continent - Total Confirmed Cases')
plt.xlabel('Continent')
plt.ylabel('Total Confirmed Cases')
plt.xticks(rotation=45)
plt.show()

# Calculate confirmed cases per Country for each country
# location_with_pop_df['Confirmed_per_country'] = location_with_pop_df['Confirmed'].groupby('Country_Region') / location_with_pop_df['2022 Population'].groupby('Country/Territory')

print("location_with_pop_df.head(10)")
# print(location_with_pop_df.head(10))

# Check top 10 countries by confirmed cases per Country
top_countries_confirmed_per_country = location_with_pop_df.sort_values(by='Confirmed_per_country',
                                                                       ascending=False).head(10)

top_countries_confirmed_per_country[['Country_Region', 'Confirmed', '2022 Population', 'Confirmed_per_country']]

print("top_countries_confirmed_per_country")
# print(top_countries_confirmed_per_country)

# Load map data (e.g., country geographic data provided by 'naturalearth_lowres')
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge 'world' DataFrame with COVID-19 data based on country name.
# Use 'location_with_pop_df' DataFrame but ensure all text is in English.
merged_df = world.merge(location_with_pop_df, left_on='name', right_on='Country_Region', how='left')
merged_df['Confirmed_per_country'] = merged_df['Confirmed_per_country'].fillna(0)

# Visualize map based on confirmed cases per Country
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged_df.plot(column='Confirmed_per_country', ax=ax, legend=True,
               legend_kwds={'label': "Confirmed Cases per Country"},
               cmap='OrRd')
plt.title('World COVID-19 Confirmed Cases per Country')
plt.show()

# Q2
print("\nMissing values in train dataset:\n", cases_train_df.isnull().sum())
# Handle missing values
cases_train_df['province'] = cases_train_df['province'].fillna('Unknown')
locations_df['Province_State'] = locations_df['Province_State'].fillna('Unknown')


#
# locations_df['Lat'] = locations_df['Lat'].fillna(locations_df['Lat'].mean())
# locations_df['Long_'] = locations_df['Long_'].fillna(locations_df['Long_'].mean())
#
# # Convert 'age' to a numeric type, forcing non-numeric values to NaN
# cases_train_df['age'] = pd.to_numeric(cases_train_df['age'], errors='coerce')
#
def convert_age(age):
    if pd.isna(age):
        return np.nan
    if '-' in age:
        start, end = age.split('-')
        if start.isdigit() and end.isdigit():
            start, end = float(start), float(end)
            return (start + end) / 2 if (end - start) < 10 else np.nan
        else:
            # '80-' -> 80
            return float(start) if start.isdigit() else np.nan
    return float(age)  # '0.75' -> 0.75


# cases_train_df['age'] = cases_train_df['age'].astype(str).apply(convert_age)
# cases_test_df['age'] = cases_test_df['age'].astype(str).apply(convert_age)

# Convert 'age' to numeric and coerce errors into NaN
cases_train_df['age'] = pd.to_numeric(cases_train_df['age'], errors='coerce')
cases_test_df['age'] = pd.to_numeric(cases_test_df['age'], errors='coerce')

cases_train_df = cases_train_df.dropna(subset=['age'])
cases_test_df = cases_test_df.dropna(subset=['age'])

# cases_train_df['age'] = cases_train_df['age'].fillna(-1)  # Using -1 for missing values
# cases_test_df['age'] = cases_test_df['age'].fillna(-1)
# median_age = cases_train_df['age'].mean(skipna=True)
#
# # Fill missing values for 'age' with the median age
# cases_train_df['age'] = cases_train_df['age'].fillna(median_age)
# cases_test_df['age'] = cases_test_df['age'].fillna(median_age)

# Fill missing values for 'sex'
# cases_train_df['sex'] = pd.to_numeric(cases_train_df['sex'], errors='coerce')
# cases_test_df['sex'] = pd.to_numeric(cases_test_df['sex'], errors='coerce')

cases_train_df['sex'] = cases_train_df['sex'].fillna(-1)  # Using -1 for missing values
cases_test_df['sex'] = cases_test_df['sex'].fillna(-1)

print("\nMissing values in train dataset after handling:\n", cases_train_df.isnull().sum())

# Combine datasets
combined_df = pd.merge(
    cases_train_df,
    locations_df,
    left_on=['country', 'province'],
    right_on=['Country_Region', 'Province_State'],
    how='inner'
)

# Combine datasets
combined_test_df = pd.merge(
    cases_test_df,
    locations_df,
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

# Calculate Expected_Mortality_Rate
combined_test_df['Deaths'] = combined_test_df['Deaths'].fillna(0)
combined_test_df['Confirmed'] = combined_test_df['Confirmed'].fillna(0)
combined_test_df['Expected_Mortality_Rate'] = combined_test_df.apply(
    lambda row: (row['Deaths'] / row['Confirmed']) if row['Confirmed'] > 0 else 0,
    axis=1
)

print("\nMissing values in combined train dataset:\n", combined_df.isnull().sum())
# print("\nMissing values in combined test dataset:\n", combined_test_df.isnull().sum())

cases_train_df = combined_df.copy()
cases_test_df = combined_test_df.copy()

# Check the data after handling missing values
print(cases_train_df.head())
print(cases_test_df.head())

# Q3 Feature selection

train_df = pd.read_csv('dataset/cases_2021_train_processed_2.csv')
test_df = pd.read_csv('dataset/cases_2021_test_processed_unlabelled_2.csv')

# selected_features = ['age', 'sex', 'chronic_disease_binary', 'latitude', 'longitude', 'Deaths',
#                      'Recovered', 'Confirmed', 'Active', 'Incident_Rate', 'Case_Fatality_Ratio']

selected_features = ['age', 'sex', 'chronic_disease_binary', 'latitude', 'longitude', 'Deaths',
                     'Recovered', 'Confirmed']
# 'outcome_group' as the target variable
outcome_feature = 'outcome_group_mapped'

numerical_features = ['age', 'latitude', 'longitude', 'Deaths',
                      'Recovered', 'Confirmed', 'Active', 'Incident_Rate', 'Case_Fatality_Ratio']
scaler = StandardScaler()
scaler.fit(cases_train_df[numerical_features])
train_df[numerical_features] = scaler.transform(train_df[numerical_features])
test_df[numerical_features] = scaler.transform(test_df[numerical_features])

# Q4: Mapping the features
# Define mappings for categorical features
sex_mapping = {'male': 0, 'female': 1}
chronic_disease_mapping = {False: 0, True: 1}
outcome_group_mapping = {'deceased': 0, 'hospitalized': 1, 'nonhospitalized': 2}
# outcome_group_mapping = {
#     'Deceased': 0,
#     'dies': 0,
#     'death': 0,
#     'Hospitalized': 1,
#     # 'Non-hospitalized': 2
#     'Recovered': 2,
#     'recovered': 2,
#     'discharge': 2,
#     'discharged': 2
# }

# Map 'sex' for both training and testing datasets
train_df['sex'] = train_df['sex'].map(sex_mapping).fillna(-1)
test_df['sex'] = test_df['sex'].map(sex_mapping).fillna(-1)

# Map 'chronic_disease_binary' for both datasets
train_df['chronic_disease_binary'] = train_df['chronic_disease_binary'].map(chronic_disease_mapping)
test_df['chronic_disease_binary'] = test_df['chronic_disease_binary'].map(chronic_disease_mapping)

# The instruction suggests mapping an 'outcome_group', which seems to be based on an 'outcome' column
# Apply the mapping to 'outcome_group' based on 'outcome' values for the training dataset
# It's assumed that 'outcome' column exists in cases_train_df and needs to be mapped to 'outcome_group'
train_df['outcome_group_mapped'] = train_df['outcome_group'].map(outcome_group_mapping)

outcome_counts = train_df['outcome_group'].value_counts()
print("outcome_counts")
print(outcome_counts)

# keeping the selected features and 'outcome_group' for the training dataset
cases_train_df_selected = train_df[selected_features + [outcome_feature]]
cases_test_df_selected = test_df[selected_features]

nan_counts_by_outcome_group = train_df.groupby('outcome_group').apply(lambda x: x.isnull().sum())
print("NaN count-----------------------")
print(nan_counts_by_outcome_group)

# Q5
# Display the initial class distribution based on 'outcome_group'
print("Original training dataset 'outcome_group' class distribution:")
original_distribution = train_df['outcome_group_mapped'].value_counts()
print(original_distribution)

print("\nMissing values in balanced_cases_train_df:\n", train_df.isnull().sum())

# Preparation for oversampling
# Find the number of samples in the largest class
max_size = train_df['outcome_group_mapped'].value_counts().max()

# Perform oversampling for each class
oversampled_list = []
for outcome, group in cases_train_df_selected.groupby('outcome_group_mapped'):
    oversampled_group = resample(group,
                                 replace=True,  # Allow sample replication
                                 n_samples=max_size,  # Match each class to the max_size
                                 random_state=123)  # Random state for reproducibility
    oversampled_list.append(oversampled_group)

# Concatenate the oversampled dataframes
balanced_cases_train_df = pd.concat(oversampled_list)

# Display the final, balanced class distribution
print("\nBalanced training dataset 'outcome_group' class distribution:")
balanced_distribution = balanced_cases_train_df['outcome_group_mapped'].value_counts()
print(balanced_distribution)
print("\nMissing values in balanced_cases_train_df:\n", balanced_cases_train_df.isnull().sum())

print(test_df[selected_features].head(20))

# Q6
# Drop rows with missing values from both X and y
# cases_train_df_clean = cases_train_df[selected_features + [outcome_feature]].dropna()
# cases_train_df_clean = balanced_cases_train_df[selected_features + [outcome_feature]].dropna()

cases_train_df_clean = balanced_cases_train_df[selected_features + [outcome_feature]]
balanced_df = cases_train_df_clean[outcome_feature].value_counts()
print("Final balanced")
print(balanced_df)

# Extract features (X) and target variable (y) after dropping missing values
X = cases_train_df_clean[selected_features]
print("\nMissing values in X:\n", X.isnull().sum())
y = cases_train_df_clean[outcome_feature]

# Splitting the cleaned data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Set up the hyperparameter grid for Random Forest including tree count, max depth, and bootstrapping option
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
    # 'min_samples_split': [2, 5, 10]
}
print("1")
f1_scorer = make_scorer(f1_score, average='macro', labels=[0])  # '0' corresponds to 'deceased'
print("2")
# Initialize GridSearchCV with additional scoring metrics for Random Forest
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring={
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'f1_deceased': f1_scorer
    },
    refit='f1_macro'
)
print("3")
# Perform hyperparameter tuning and model training with grid search
grid_search.fit(X_train, y_train)
print("4")
# Extract the results into a DataFrame
results = pd.DataFrame(grid_search.cv_results_)
print("5")
# Save the results to a text file
results.to_csv('hyperparameter_tuning_results_rf.txt', index=False)
print("6")
# Report the best parameters and their corresponding scores
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")
print(f"Mean macro F1-score across the validation sets: {results.loc[grid_search.best_index_, 'mean_test_f1_macro']}")
print(f"Mean F1-score on 'deceased' across the validation sets: {results.loc[grid_search.best_index_, 'mean_test_f1_deceased']}")
print(f"Mean overall accuracy across the validation sets: {results.loc[grid_search.best_index_, 'mean_test_accuracy']}")

# Evaluate the best model's performance on the validation dataset
y_val_pred = grid_search.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation set accuracy: {val_accuracy}")

print(f"Best parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)

report = classification_report(y_val, y_pred, target_names=['deceased', 'hospitalized', 'non_hospitalized'])
print(report)

# Predict on the training set
y_train_pred = best_model.predict(X_train)

# Predict on the validation set
y_val_pred = best_model.predict(X_val)

cm_train = confusion_matrix(y_train, y_train_pred)
cm_val = confusion_matrix(y_val, y_val_pred)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt='g')
plt.title('Confusion Matrix for Training Set')

plt.subplot(1, 2, 2)
sns.heatmap(cm_val, annot=True, fmt='g')
plt.title('Confusion Matrix for Validation Set')

plt.show()

# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
# Logistic regression

# cases_train_df_clean = balanced_cases_train_df[selected_features + [outcome_feature]]
# balanced_df = cases_train_df_clean['outcome_group'].value_counts()
# print("final balanced")
# print(balanced_df)
#
# # Extract features (X) and target variable (y) after dropping missing values
# X = cases_train_df_clean[selected_features]
# print("\nMissing values in X:\n", X.isnull().sum())
# y = cases_train_df_clean[outcome_feature]
#
# # Splitting the cleaned data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # # Initialize the Logistic Regression model
# # model = LogisticRegression(max_iter=1000, random_state=42)
# # Initialize the Multinomial Logistic Regression model
# model = LogisticRegression(max_iter=10000, random_state=42, multi_class='multinomial')
#
# # Train the model
# model.fit(X_train, y_train)
#
# # Predictions on the validation set
# y_pred = model.predict(X_val)
#
# # Evaluate the model
# accuracy = accuracy_score(y_val, y_pred)
# precision = precision_score(y_val, y_pred, average='weighted')
# recall = recall_score(y_val, y_pred, average='weighted')
#
# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
#
# # Set up the hyperparameter grid
# param_grid = {
#     'logisticregression__C': [0.01, 0.1, 1, 10, 100],
#     'logisticregression__solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
#     # 'logisticregression__solver': ['auto'],
#     'logisticregression__penalty': ['l2', None]
# }
#
# f1_scorer = make_scorer(f1_score, average='macro', labels=[0])  # '0' corresponds to 'deceased'
# # Create a pipeline including the scaler and the logistic regression model
# pipeline = make_pipeline(StandardScaler(),
#                          LogisticRegression(max_iter=10000, random_state=42, multi_class='multinomial'))
#
# # Initialize GridSearchCV with additional scoring metrics
# grid_search = GridSearchCV(
#     pipeline,
#     param_grid,
#     cv=5,
#     scoring={
#         'accuracy': 'accuracy',
#         'f1_macro': 'f1_macro',
#         'f1_deceased': f1_scorer
#     },
#     refit='f1_macro'
# )
#
# # Perform hyperparameter tuning and model training with grid search
# grid_search.fit(X_train, y_train)
#
# # Extract the results into a DataFrame
# results = pd.DataFrame(grid_search.cv_results_)
#
# # Save the results to a text file
# results.to_csv('hyperparameter_tuning_results.txt', index=False)
#
# # Report the best parameters and their corresponding scores
# print(f"Best parameters: {grid_search.best_params_}")
# print(f"Best cross-validation score: {grid_search.best_score_}")
# print(f"Mean macro F1-score across the validation sets: {results.loc[grid_search.best_index_, 'mean_test_f1_macro']}")
# print(
#     f"Mean F1-score on 'deceased' across the validation sets: {results.loc[grid_search.best_index_, 'mean_test_f1_deceased']}")
# print(f"Mean overall accuracy across the validation sets: {results.loc[grid_search.best_index_, 'mean_test_accuracy']}")
#
# # Evaluate the best model's performance on the validation dataset
# y_val_pred = grid_search.predict(X_val)
# val_accuracy = accuracy_score(y_val, y_val_pred)
# print(f"Validation set accuracy: {val_accuracy}")
#
# print(f"Best parameters: {grid_search.best_params_}")
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_val)
#
# report = classification_report(y_val, y_val_pred, target_names=['deceased', 'hospitalized', 'non_hospitalized'])
# print(report)

# Q7
# Hyperparameters to explore
n_estimators_values = [10, 50, 100, 200]
max_depth_values = [None, 5, 10, 20]

# Storing results
results = []

for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Predict on both training and validation sets
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Calculate and store the accuracy and F1 score for both sets
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        val_f1 = f1_score(y_val, y_val_pred, average='macro')

        # Append results
        results.append({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_f1': train_f1,
            'val_f1': val_f1
        })

# Converting results to a DataFrame for easier plotting
results_df = pd.DataFrame(results)

# Example plot: Validation Accuracy vs. Number of Estimators for different Max Depth
for max_depth in max_depth_values:
    subset = results_df[results_df['max_depth'] == max_depth]
    plt.plot(subset['n_estimators'], subset['val_accuracy'], label=f'Max Depth: {max_depth}')

plt.xlabel('Number of Estimators')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy by Number of Estimators and Max Depth')
plt.legend()
plt.show()

# # Values of C to explore
# C_values = [0.01, 0.1, 1, 10, 100]
# # C_values = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
#
# train_accuracies = []
# val_accuracies = []
# train_f1_scores = []
# val_f1_scores = []
#
# for C in C_values:
#     model = LogisticRegression(C=C, max_iter=10000, random_state=42, multi_class='multinomial')
#     model.fit(X_train, y_train)
#
#     # Predict on both training and validation sets
#     y_train_pred = model.predict(X_train)
#     y_val_pred = model.predict(X_val)
#
#     # Calculate and store the accuracy and F1 score for both sets
#     train_accuracies.append(accuracy_score(y_train, y_train_pred))
#     val_accuracies.append(accuracy_score(y_val, y_val_pred))
#     train_f1_scores.append(f1_score(y_train, y_train_pred, average='macro'))
#     val_f1_scores.append(f1_score(y_val, y_val_pred, average='macro'))
#
# # Plotting the results
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.plot(C_values, train_accuracies, label='Training Accuracy')
# plt.plot(C_values, val_accuracies, label='Validation Accuracy')
# plt.xlabel('C (Regularization strength)')
# plt.ylabel('Accuracy')
# plt.xscale('log')
# plt.title('Accuracy vs Regularization strength')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(C_values, train_f1_scores, label='Training F1 Score')
# plt.plot(C_values, val_f1_scores, label='Validation F1 Score')
# plt.xlabel('C (Regularization strength)')
# plt.ylabel('F1 Score')
# plt.xscale('log')
# plt.title('F1 Score vs Regularization strength')
# plt.legend()
#
# plt.tight_layout()
# plt.show()

# Q9


best_model = grid_search.best_estimator_
y_preds = best_model.predict(test_df[selected_features])



# def create_submission_file(y_preds, model_name):
#     file_name = f"submission_{model_name}.csv"
#
#     with open(file_name, "w", newline='') as csvfile:
#         wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
#         wr.writerow(["Id", "Prediction"])
#         for i, pred in enumerate(y_preds):
#             # wr.writerow([i, pred])
#             wr.writerow([str(i), str(pred)])
#
#
# create_submission_file(y_preds, "logreg")

def create_submission_file(y_preds, file_name):
    with open(file_name, "w") as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["Id", "Prediction"])
        for i, pred in enumerate(y_preds):
            wr.writerow([str(i), str(pred)])


create_submission_file(y_preds, "submission_rf.csv")
