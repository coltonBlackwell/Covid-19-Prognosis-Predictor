import pandas as pd
import numpy as np

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

# Q4
# Mapping for 'sex' and 'chronic_disease_binary'
sex_mapping = {'male': 0, 'female': 1}
chronic_disease_mapping = {False: 0, True: 1}

train_df['sex'] = train_df['sex'].map(sex_mapping).fillna(-1)
test_df['sex'] = test_df['sex'].map(sex_mapping).fillna(-1)

train_df['chronic_disease_binary'] = train_df['chronic_disease_binary'].map(chronic_disease_mapping)
test_df['chronic_disease_binary'] = test_df['chronic_disease_binary'].map(chronic_disease_mapping)

# Apply the corrected mapping to 'outcome_group'
outcome_group_mapping = {'deceased': 0, 'hospitalized': 1, 'non_hospitalized': 2}
train_df[outcome_feature] = train_df['outcome'].map(outcome_group_mapping)

# keeping the selected features and 'outcome_group' for the training dataset
train_df_selected = train_df[selected_features + [outcome_feature]]
test_df_selected = test_df[selected_features]
