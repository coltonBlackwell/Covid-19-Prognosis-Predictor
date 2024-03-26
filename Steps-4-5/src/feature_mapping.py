import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler # Make sure to run pip install -U imbalanced-learn to run

cases_data = pd.read_csv('../data/cases_2021_train_processed_2 - cases_2021_train_processed_2.csv')
# cases_test = pd.read_csv('../data/cases_2021_test_processed_unlabelled_2 - cases_2021_test_processed_unlabelled_2.csv')


#-----(CONVERTING SEX CATEGORICAL)------

cases_data['sex'] = pd.Categorical(cases_data['sex'])
# cases_test['sex'] = pd.Categorical(cases_test['sex'])

cases_data['sex_code'] = cases_data['sex'].cat.codes
# cases_test['sex_code'] = cases_test['sex'].cat.codes

#-----(CONVERTING CHRONIC_DISEASE_BINARY CATEGORICAL)------

cases_data['chronic_disease_binary'] = pd.Categorical(cases_data['chronic_disease_binary'])
# cases_test['chronic_disease_binary'] = pd.Categorical(cases_test['chronic_disease_binary'])

cases_data['chronic_disease_binary_code'] = cases_data['chronic_disease_binary'].cat.codes
# cases_test['chronic_disease_binary_code'] = cases_test['chronic_disease_binary'].cat.codes


#-----(CONVERTING PROVINCE CATEGORICAL)------  // If province code is -1, that means it is NULL

cases_data['province'] = pd.Categorical(cases_data['province'])
# cases_test['province'] = pd.Categorical(cases_test['province'])

cases_data['province_code'] = cases_data['province'].cat.codes
# cases_test['province_code'] = cases_test['province'].cat.codes

#-----(CONVERTING COUNTRY CATEGORICAL)------  // If country code is -1, that means it is NULL

cases_data['country'] = pd.Categorical(cases_data['country'])
# cases_test['country'] = pd.Categorical(cases_test['country'])

cases_data['country_code'] = cases_data['country'].cat.codes
# cases_test['country_code'] = cases_test['country'].cat.codes

#-----(CONVERTING OUTCOME_GROUP TO CATEGORICAL)------

cases_data['outcome_group'] = pd.Categorical(cases_data['outcome_group'])
cases_data['outcome_group_code'] = cases_data['outcome_group'].cat.codes

#-----(Ensure all data types are numeric)---------------

cases_data['age'] = pd.to_numeric(cases_data['age'], errors='coerce')
cases_data['latitude'] = pd.to_numeric(cases_data['latitude'], errors='coerce')
cases_data['longitude'] = pd.to_numeric(cases_data['longitude'], errors='coerce')
cases_data['Confirmed'] = pd.to_numeric(cases_data['Confirmed'], errors='coerce')
cases_data['Deaths'] = pd.to_numeric(cases_data['Deaths'], errors='coerce')
cases_data['Recovered'] = pd.to_numeric(cases_data['Recovered'], errors='coerce')
cases_data['Active'] = pd.to_numeric(cases_data['Active'], errors='coerce')
cases_data['Incident_Rate'] = pd.to_numeric(cases_data['Incident_Rate'], errors='coerce')
cases_data['Case_Fatality_Ratio'] = pd.to_numeric(cases_data['Case_Fatality_Ratio'], errors='coerce')


#-----(REMOVING DATA CONFIRMATION)----------

cases_data = cases_data.drop(columns=['date_confirmation'])


#-----(BALANCE DATASET CLASSES USING UNDERSAMPLING)------  // If country code is -1, that means it is NULL


X = cases_data.drop(['outcome_group'], axis=1)
y = cases_data['outcome_group']

# print("Imbalanced class before: ", y.value_counts())

max_samples = max(cases_data['outcome_group'].value_counts())
ros = RandomOverSampler(sampling_strategy={'hospitalized': max_samples, 'nonhospitalized': max_samples, 'deceased': max_samples})
X_res, y_res = ros.fit_resample(X, y)

# print("Balanced class after: ", y_res.value_counts())


resampled_data = pd.concat([X_res, pd.DataFrame(y_res, columns=['outcome_group'])], axis=1)
resampled_data.to_csv('../result/oversampled_processed_data.csv', index=False)  # Save resulting file to resampled_data.csv 

#Use undersampled_processed_data.csv for your models in step 6 !! (DONT NEED TO UNDERSAMPLE TEST DATASET)