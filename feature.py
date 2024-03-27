import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the datasets
train_df = pd.read_csv('dataset/cases_2021_train.csv')
test_df = pd.read_csv('dataset/cases_2021_test.csv')

# Q3 Feature selection
selected_features = ['age', 'sex', 'chronic_disease_binary', 'latitude', 'longitude']
# 'outcome_group' as the target variable
outcome_feature = 'outcome_group'

# Convert 'age' to numeric and coerce errors into NaN
train_df['age'] = pd.to_numeric(train_df['age'], errors='coerce')
test_df['age'] = pd.to_numeric(test_df['age'], errors='coerce')

# Q4: Mapping the features
# Define mappings for categorical features
sex_mapping = {'male': 0, 'female': 1}
chronic_disease_mapping = {False: 0, True: 1}
# outcome_group_mapping = {'deceased': 0, 'hospitalized': 1, 'non_hospitalized': 2}
outcome_group_mapping = {
    'Deceased': 0,
    'Hospitalized': 1,
    # 'Non-hospitalized': 2
    'Recovered': 2
}

# Map 'sex' for both training and testing datasets
train_df['sex'] = train_df['sex'].map(sex_mapping).fillna(-1)  # Using -1 for missing values
test_df['sex'] = test_df['sex'].map(sex_mapping).fillna(-1)

# Map 'chronic_disease_binary' for both datasets
train_df['chronic_disease_binary'] = train_df['chronic_disease_binary'].map(chronic_disease_mapping)
test_df['chronic_disease_binary'] = test_df['chronic_disease_binary'].map(chronic_disease_mapping)

# The instruction suggests mapping an 'outcome_group', which seems to be based on an 'outcome' column
# Apply the mapping to 'outcome_group' based on 'outcome' values for the training dataset
# It's assumed that 'outcome' column exists in train_df and needs to be mapped to 'outcome_group'
train_df['outcome_group'] = train_df['outcome'].map(outcome_group_mapping)

# For the test dataset, apply similar mapping if 'outcome' column exists and needs to be predicted
# Assuming test_df also has 'outcome' column that needs to be mapped for consistency in preprocessing
# Note: If test_df doesn't have 'outcome' column or if it's the target for prediction, this step may be adjusted accordingly
test_df['outcome_group'] = test_df.get('outcome').map(outcome_group_mapping) if 'outcome' in test_df else None

# keeping the selected features and 'outcome_group' for the training dataset
train_df_selected = train_df[selected_features + [outcome_feature]]
test_df_selected = test_df[selected_features]

# Display the initial class distribution based on 'outcome_group'
print("Original training dataset 'outcome_group' class distribution:")
original_distribution = train_df['outcome_group'].value_counts()
print(original_distribution)

# Preparation for oversampling
# Find the number of samples in the largest class
max_size = train_df['outcome_group'].value_counts().max()

# Perform oversampling for each class
oversampled_list = []
for outcome, group in train_df.groupby('outcome_group'):
    oversampled_group = resample(group,
                                 replace=True,  # Allow sample replication
                                 n_samples=max_size,  # Match each class to the max_size
                                 random_state=123)  # Random state for reproducibility
    oversampled_list.append(oversampled_group)

# Concatenate the oversampled dataframes
balanced_train_df = pd.concat(oversampled_list)

# Display the final, balanced class distribution
print("\nBalanced training dataset 'outcome_group' class distribution:")
balanced_distribution = balanced_train_df['outcome_group'].value_counts()
print(balanced_distribution)


# Q6
# Drop rows with missing values from both X and y
train_df_clean = train_df[selected_features + [outcome_feature]].dropna()

# Extract features (X) and target variable (y) after dropping missing values
X = train_df_clean[selected_features]
y = train_df_clean[outcome_feature]

# Splitting the cleaned data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# # Initialize the Logistic Regression model
# model = LogisticRegression(max_iter=1000, random_state=42)
# Initialize the Multinomial Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial', solver='lbfgs')

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
