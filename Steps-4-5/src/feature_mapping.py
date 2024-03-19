import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler # Make sure to run pip install -U imbalanced-learn to run

cases_data = pd.read_csv('../data/cases_2021_train_processed_2 - cases_2021_train_processed_2.csv')
cases_test = pd.read_csv('../data/cases_2021_test_processed_unlabelled_2 - cases_2021_test_processed_unlabelled_2.csv')


#-----(CONVERTING SEX CATEGORICAL)------

cases_data['sex'] = pd.Categorical(cases_data['sex'])
cases_test['sex'] = pd.Categorical(cases_test['sex'])

cases_data['sex_code'] = cases_data['sex'].cat.codes
cases_test['sex_code'] = cases_test['sex'].cat.codes

#-----(CONVERTING CHRONIC_DISEASE_BINARY CATEGORICAL)------

cases_data['chronic_disease_binary'] = pd.Categorical(cases_data['chronic_disease_binary'])
cases_test['chronic_disease_binary'] = pd.Categorical(cases_test['chronic_disease_binary'])

cases_data['chronic_disease_binary_code'] = cases_data['chronic_disease_binary'].cat.codes
cases_test['chronic_disease_binary_code'] = cases_test['chronic_disease_binary'].cat.codes

#-----(CONVERTING OUTCOME_GROUP TO CATEGORICAL)------

cases_data['outcome_group'] = pd.Categorical(cases_data['outcome_group'])
cases_data['outcome_group_code'] = cases_data['outcome_group'].cat.codes

#-----(CONVERTING PROVINCE CATEGORICAL)------  // If province code is -1, that means it is NULL

cases_data['province'] = pd.Categorical(cases_data['province'])
cases_test['province'] = pd.Categorical(cases_test['province'])

cases_data['province_code'] = cases_data['province'].cat.codes
cases_test['province_code'] = cases_test['province'].cat.codes

#-----(CONVERTING COUNTRY CATEGORICAL)------  // If country code is -1, that means it is NULL

cases_data['country'] = pd.Categorical(cases_data['country'])
cases_test['country'] = pd.Categorical(cases_test['country'])

cases_data['country_code'] = cases_data['country'].cat.codes
cases_test['country_code'] = cases_test['country'].cat.codes

#-----(BALANCE DATASET CLASSES USING UNDERSAMPLING)------  // If country code is -1, that means it is NULL


X = cases_data.drop(['outcome_group'], axis=1)
y = cases_data['outcome_group']

# print("Imbalanced class before: ", y.value_counts())

min_samples = min(cases_data['outcome_group'].value_counts())
rus = RandomUnderSampler(sampling_strategy={'hospitalized': min_samples, 'nonhospitalized': min_samples, 'deceased': min_samples})
X_res, y_res = rus.fit_resample(X, y)

# print("Balanced class after: ", y_res.value_counts()) IMPORTANT for report


resampled_data = pd.concat([X_res, pd.DataFrame(y_res, columns=['outcome_group'])], axis=1)
resampled_data.to_csv('../result/undersampled_processed_data.csv', index=False)  # Save resulting file to resampled_data.csv 

#Use undersampled_processed_data.csv for your models in step 6 !! (DONT NEED TO UNDERSAMPLE TEST DATASET)