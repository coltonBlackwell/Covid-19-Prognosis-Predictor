# take in ../case_train_procesed.csv
# remove source
# remove date confirmation
# Add mapping for sex_code,chronic_disease_binary_code,province_code,country_code,outcome_group_code,outcome_group
# save file into Steps-6/result/
# then after this preprocessing file should be good for hyperparameter tuning


import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler # Make sure to run pip install -U imbalanced-learn to run


hyperparameter_tuning_data = pd.read_csv('/home/colton/Documents/university/3rd Year/2nd Semester/CMPT 459/Assignments/CMPT459-Final-Group-Project/Steps-1-2-3/results/case_train_processed.csv')

# #-----(DROPPING UNNECESSARY COLUMNS)------

hyperparameter_tuning_data = hyperparameter_tuning_data.drop(['source'], axis=1)
hyperparameter_tuning_data = hyperparameter_tuning_data.drop(['outcome_group'], axis=1)
hyperparameter_tuning_data = hyperparameter_tuning_data.drop(['additional_information'], axis=1)

# #-----(MERGING COLUMNS)------

# merging classes Recovered and recovered

hyperparameter_tuning_data['outcome'] = hyperparameter_tuning_data['outcome'].replace({'recovered': 'Recovered'})
hyperparameter_tuning_data['outcome'] = hyperparameter_tuning_data['outcome'].replace({'discharge': 'discharged'})
hyperparameter_tuning_data['outcome'] = hyperparameter_tuning_data['outcome'].replace({'died': 'Deceased'})

# #-----(CONVERTING SEX CATEGORICAL)------


hyperparameter_tuning_data['sex'] = pd.Categorical(hyperparameter_tuning_data['sex'])
hyperparameter_tuning_data['sex_code'] = hyperparameter_tuning_data['sex'].cat.codes

# #-----(CONVERTING CHRONIC_DISEASE_BINARY CATEGORICAL)------

hyperparameter_tuning_data['chronic_disease_binary'] = pd.Categorical(hyperparameter_tuning_data['chronic_disease_binary'])
hyperparameter_tuning_data['chronic_disease_binary_code'] = hyperparameter_tuning_data['chronic_disease_binary'].cat.codes


# #-----(CONVERTING PROVINCE CATEGORICAL)------  // If province code is -1, that means it is NULL

hyperparameter_tuning_data['province'] = pd.Categorical(hyperparameter_tuning_data['province'])
hyperparameter_tuning_data['province_code'] = hyperparameter_tuning_data['province'].cat.codes

# #-----(CONVERTING COUNTRY CATEGORICAL)------  // If country code is -1, that means it is NULL

hyperparameter_tuning_data['country'] = pd.Categorical(hyperparameter_tuning_data['country'])
hyperparameter_tuning_data['country_code'] = hyperparameter_tuning_data['country'].cat.codes

# #-----(CONVERTING OUTCOME_GROUP TO CATEGORICAL)------

hyperparameter_tuning_data['outcome'] = pd.Categorical(hyperparameter_tuning_data['outcome'])
hyperparameter_tuning_data['outcome_group_code'] = hyperparameter_tuning_data['outcome'].cat.codes

# #-----(Ensure all data types are numeric)---------------

hyperparameter_tuning_data['age'] = pd.to_numeric(hyperparameter_tuning_data['age'], errors='coerce')
hyperparameter_tuning_data['latitude'] = pd.to_numeric(hyperparameter_tuning_data['latitude'], errors='coerce')
hyperparameter_tuning_data['longitude'] = pd.to_numeric(hyperparameter_tuning_data['longitude'], errors='coerce')
hyperparameter_tuning_data['Confirmed'] = pd.to_numeric(hyperparameter_tuning_data['Confirmed'], errors='coerce')
hyperparameter_tuning_data['Deaths'] = pd.to_numeric(hyperparameter_tuning_data['Deaths'], errors='coerce')
hyperparameter_tuning_data['Recovered'] = pd.to_numeric(hyperparameter_tuning_data['Recovered'], errors='coerce')
hyperparameter_tuning_data['Active'] = pd.to_numeric(hyperparameter_tuning_data['Active'], errors='coerce')
hyperparameter_tuning_data['Incident_Rate'] = pd.to_numeric(hyperparameter_tuning_data['Incident_Rate'], errors='coerce')
hyperparameter_tuning_data['Case_Fatality_Ratio'] = pd.to_numeric(hyperparameter_tuning_data['Case_Fatality_Ratio'], errors='coerce')


# #-----(REMOVING DATA CONFIRMATION)----------

hyperparameter_tuning_data = hyperparameter_tuning_data.drop(columns=['date_confirmation'])


# #-----(BALANCE DATASET CLASSES USING UNDERSAMPLING)------  // If country code is -1, that means it is NULL


X = hyperparameter_tuning_data.drop(['outcome'], axis=1)
y = hyperparameter_tuning_data['outcome']

print("Imbalanced class before: ", y.value_counts())

max_samples = max(hyperparameter_tuning_data['outcome'].value_counts())
ros = RandomOverSampler(sampling_strategy={'Hospitalized': max_samples, 'Deceased': max_samples, 'Recovered': max_samples}) # Ignoring other outcomes as they are infinitly small comapred to these three
X_res, y_res = ros.fit_resample(X, y)

print("Balanced class after: ", y_res.value_counts())


# resampled_data = pd.concat([X_res, pd.DataFrame(y_res, columns=['outcome'])], axis=1)
# X_res.to_csv('../hyperparameter_tuning_data/oversampled_processed_data.csv', index=False)  # Save resulting file to resampled_data.csv 

X_res.to_csv('../hyperparameter_tuning_data/hyperparameter_tuning_data.csv', index=False)

# #Use undersampled_processed_data.csv for your models in step 6 !! (DONT NEED TO UNDERSAMPLE TEST DATASET)