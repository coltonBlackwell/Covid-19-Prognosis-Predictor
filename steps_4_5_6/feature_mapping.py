import pandas as pd
import numpy as np
import os

cases_data = pd.read_csv('../dataset/subset/cases_2021_train_processed_2 - cases_2021_train_processed_2.csv')
cases_test = pd.read_csv('../dataset/subset/cases_2021_test_processed_unlabelled_2 - cases_2021_test_processed_unlabelled_2.csv')


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

# Display the resulting DataFrame
print(cases_data)