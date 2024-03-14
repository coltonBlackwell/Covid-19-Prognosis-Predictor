import pandas as pd
import numpy as np
import os

#----------- (SCANNING FILES)-----------


cases_data = pd.read_csv('../dataset/cases_2021_train.csv')
cases_test = pd.read_csv('../dataset/cases_2021_test.csv')
location_data = pd.read_csv('../dataset/location_2021.csv')


#----------- (DATA CLEANING)-----------

# - Remove the LAST_UPDATE column, it doesnt addmuch 
location_data = location_data.drop(['Last_Update'], axis=1)


# - REplace age/sex with -1
cases_data = cases_data.fillna({'age': -1, 'sex': -1})
cases_test = cases_test.fillna({'age': -1, 'sex': -1})


#----------- (REMOVING OUTLIERS)-----------

# Convert 'age' column to numeric type, coerce errors to NaN
cases_data['age'] = pd.to_numeric(cases_data['age'], errors='coerce')
cases_test['age'] = pd.to_numeric(cases_test['age'], errors='coerce')

# - Remove unlikely ages from dataset

cases_data = cases_data[(cases_data['age'] >= 0) & (cases_data['age'] < 105)]
cases_test = cases_test[(cases_test['age'] >= 0) & (cases_test['age'] < 105)]

# - Remove unlikely Long/Lat from dataset

cases_data = cases_data[(cases_data['latitude'] >= -90) & (cases_data['latitude'] < 90)]
cases_data = cases_data[(cases_data['longitude'] >= -180) & (cases_data['longitude'] < 180)]
cases_test = cases_test[(cases_test['longitude'] >= -180) & (cases_test['longitude'] < 180)]
cases_test = cases_test[(cases_test['latitude'] >= -90) & (cases_test['latitude'] < 90)] 


# - Remove if province is unknown - The long and lat is missing
location_data = location_data.dropna(subset=['Lat', 'Long_'], how='any')


#----------- (JOINING DATASETS)-----------

join_train = cases_data.merge(location_data, left_on=['province', 'country'], right_on=['Province_State', 'Country_Region'], how='inner')
join_test = cases_test.merge(location_data, left_on=['province', 'country'], right_on=['Province_State', 'Country_Region'], how='inner')

join_train['Expected_Mortality_Rate'] = join_train['Deaths'] / join_train['Confirmed'] * 100
join_test['Expected_Mortality_Rate'] = join_test['Deaths'] / join_test['Confirmed'] * 100

columns_to_drop = ['Lat', 'Long_', 'Province_State', 'Country_Region']
join_train.drop(columns_to_drop, axis=1, inplace=True)
join_test.drop(columns_to_drop, axis=1, inplace=True)

join_train.to_csv('../dataset/case_train_processed.csv', index=False)
join_test.to_csv('../dataset/case_test_processed.csv', index=False)


#NOTES ABOUT FEATURE SELECTION
