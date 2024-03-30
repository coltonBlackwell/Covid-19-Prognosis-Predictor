import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

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
# train_df_clean = cases_train_df[selected_features + [outcome_feature]].dropna()
train_df_clean = balanced_train_df[selected_features + [outcome_feature]].dropna()

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

# Set up the hyperparameter grid
param_grid = {
    'logisticregression__C': [0.01, 0.1, 1, 10, 100],
    'logisticregression__solver': ['lbfgs', 'newton-cg', 'sag', 'saga']
}

f1_scorer = make_scorer(f1_score, average='macro', labels=[0])  # assuming '0' corresponds to 'deceased'
# Create a pipeline including the scaler and the logistic regression model
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'))

# Initialize GridSearchCV with additional scoring metrics
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring={
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'f1_deceased': f1_scorer
    },
    refit='accuracy'  # Choose the metric to use for refitting the best model
)

# Perform hyperparameter tuning and model training with grid search
grid_search.fit(X_train, y_train)

# Extract the results into a DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# Save the results to a text file
results.to_csv('hyperparameter_tuning_results.txt', index=False)

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

# Q7
# Values of C to explore
C_values = [0.01, 0.1, 1, 10, 100]

train_accuracies = []
val_accuracies = []
train_f1_scores = []
val_f1_scores = []

for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000, random_state=42, multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, y_train)

    # Predict on both training and validation sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Calculate and store the accuracy and F1 score for both sets
    train_accuracies.append(accuracy_score(y_train, y_train_pred))
    val_accuracies.append(accuracy_score(y_val, y_val_pred))
    train_f1_scores.append(f1_score(y_train, y_train_pred, average='macro'))
    val_f1_scores.append(f1_score(y_val, y_val_pred, average='macro'))

# Plotting the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(C_values, train_accuracies, label='Training Accuracy')
plt.plot(C_values, val_accuracies, label='Validation Accuracy')
plt.xlabel('C (Regularization strength)')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.title('Accuracy vs Regularization strength')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(C_values, train_f1_scores, label='Training F1 Score')
plt.plot(C_values, val_f1_scores, label='Validation F1 Score')
plt.xlabel('C (Regularization strength)')
plt.ylabel('F1 Score')
plt.xscale('log')
plt.title('F1 Score vs Regularization strength')
plt.legend()

plt.tight_layout()
plt.show()