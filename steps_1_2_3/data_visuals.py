import pandas as pd
import geopandas as gpd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import country_converter as coco
from datetime import datetime, timedelta
import os.path
import matplotlib.colors as mcolors


def plot_bar_chart(x_values, y_values, x_label, y_label, title):
    plt.figure(figsize=(10, 6))
    plt.bar(x_values, y_values, color='skyblue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

df = pd.read_csv('../dataset/location_2021.csv')

cases_df = pd.read_csv('../dataset/cases_2021_test.csv')
location_df = pd.read_csv('../dataset/location_2021.csv')
merged_df = pd.merge(cases_df, location_df, how='inner', left_on=['country', 'province'], right_on=['Country_Region', 'Province_State'])


#-----(COUNTRIES WITH THE MOST CONFIRMED CASES)----------

grouped_df = df.groupby('Country_Region')['Confirmed'].sum().reset_index()
sorted_df = grouped_df.sort_values(by='Confirmed', ascending=False)
top_15_countries = sorted_df.head(15)
plot_bar_chart(top_15_countries['Country_Region'], top_15_countries['Confirmed'], 'Country', 'Confirmed Cases', 'Top 15 Countries with Most Confirmed Cases')

#-----(PROVINCES WITH THE MOST CONFIRMED CASES)----------

grouped_df = df.groupby('Province_State')['Confirmed'].sum().reset_index()
sorted_df = grouped_df.sort_values(by='Confirmed', ascending=False)
top_15_provinces = sorted_df.head(15)
plot_bar_chart(top_15_provinces['Province_State'], top_15_provinces['Confirmed'], 'Province', 'Confirmed Cases', 'Top 15 Provinces with Most Confirmed Cases')

# #-----(COUNTRIES WITH THE MOST DEATHS)----------

grouped_df = df.groupby('Country_Region')['Deaths'].sum().reset_index()
sorted_df = grouped_df.sort_values(by='Deaths', ascending=False)
top_15_countries = sorted_df.head(15)
plot_bar_chart(top_15_countries['Country_Region'], top_15_countries['Deaths'], 'Country', 'Deaths', 'Top 15 Countries with Most Deaths')

# #-----(COUNTRIES WITH HIGHEST FATALITY RATE)----------

df['Fatality_Ratio'] = (df['Deaths'] / df['Confirmed']) * 100
country_fatality = df.groupby('Country_Region')['Fatality_Ratio'].mean().reset_index()
country_fatality = country_fatality.sort_values(by='Fatality_Ratio', ascending=False).head(15)
plot_bar_chart(country_fatality['Country_Region'], country_fatality['Fatality_Ratio'], 'Country', 'Fatality Ratio (%)', 'Top 15 Countries with Highest Fatality Ratios to Confirmed Cases')

#-----(PROVINCES WITH HIGHEST FATALITY RATE)---------- (FIX ISSUE WITH UNKNOWN CASE!!)

df['Fatality_Ratio'] = (df['Deaths'] / df['Confirmed']) * 100
country_fatality = df.groupby('Province_State')['Fatality_Ratio'].mean().reset_index()
country_fatality = country_fatality.sort_values(by='Fatality_Ratio', ascending=False).head(15)
plot_bar_chart(country_fatality['Province_State'], country_fatality['Fatality_Ratio'], 'Country', 'Fatality Ratio (%)', 'Top 15 Provinces with Highest Fatality Ratios to Confirmed Cases')

# #-----(COUNTRIES WITH HIGHEST INCIDENT RATE)----------

df_sorted = df.sort_values(by='Incident_Rate', ascending=False)
top_15_countries = df_sorted.head(15)
plot_bar_chart(top_15_countries['Country_Region'], top_15_countries['Incident_Rate'], 'Country', 'Incidence Rate', 'Top 15 Countries with Highest Incidence Rates')

#-----(PROVINCES WITH HIGHEST INCIDENT RATE)----------


df_sorted = df.sort_values(by='Incident_Rate', ascending=False)
top_15_provinces = df_sorted.head(22)
plot_bar_chart(top_15_provinces['Province_State'], top_15_provinces['Incident_Rate'], 'Province', 'Incidence Rate', 'Top 15 Provinces with Highest Incidence Rates')

#-----(FEMALE TO MALE CASES)----------


female_cases = merged_df[merged_df['sex'] == 'female']['Confirmed'].sum()
male_cases = merged_df[merged_df['sex'] == 'male']['Confirmed'].sum()

labels = ['female', 'male']
sizes = [female_cases, male_cases]
colors = ['pink', 'lightblue']

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of COVID-19 Cases by Gender')
plt.axis('equal')
plt.show()

#-----(HEATMAP OF CONFIRMED CASES)----------

plt.figure(figsize=(20, 10))
plt.title('Heat Map of Confirmed Cases')

scatter_sizes = location_df['Confirmed']**0.5
norm = mcolors.Normalize(0, 100000)
normalized_sizes = norm(scatter_sizes)
scatter_generator = plt.scatter(x=location_df['Long_'], y=location_df['Lat'], s=scatter_sizes, c=normalized_sizes, cmap='viridis', alpha=0.5)

plt.xlabel('Longitude')
plt.ylabel('Latitude')

sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Number of Active Cases')

plt.show()

#-----(DATA PREPROCESSING)----------

