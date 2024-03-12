import pandas as pd
import geopandas as gpd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import country_converter as coco
from datetime import datetime, timedelta

def plot_bar_chart(x_values, y_values, x_label, y_label, title):
    plt.figure(figsize=(10, 6))
    plt.bar(x_values, y_values, color='skyblue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

df = pd.read_csv('dataset/location_2021.csv')

cases_df = pd.read_csv('dataset/cases_2021_test.csv')
location_df = pd.read_csv('dataset/location_2021.csv')

merged_df = pd.merge(cases_df, location_df, how='inner', left_on=['country', 'province'], right_on=['Country_Region', 'Province_State'])


#-----(COUNTRIES WITH THE MOST CONFIRMED CASES)----------

grouped_df = df.groupby('Country_Region')['Confirmed'].sum().reset_index()

sorted_df = grouped_df.sort_values(by='Confirmed', ascending=False)

top_15_countries = sorted_df.head(15)

plt.figure(figsize=(10, 6))
plt.bar(top_15_countries['Country_Region'], top_15_countries['Confirmed'], color='skyblue')
plt.xlabel('Country')
plt.ylabel('Confirmed Cases')
plt.title('Top 15 Countries with Most Confirmed Cases')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#-----(PROVINCES WITH THE MOST CONFIRMED CASES)----------

grouped_df = df.groupby('Province_State')['Confirmed'].sum().reset_index()

sorted_df = grouped_df.sort_values(by='Confirmed', ascending=False)

top_15_provinces = sorted_df.head(15)

plt.figure(figsize=(10, 6))
plt.bar(top_15_provinces['Province_State'], top_15_provinces['Confirmed'], color='skyblue')
plt.xlabel('Country')
plt.ylabel('Confirmed Cases')
plt.title('Top 15 Provinces with Most Confirmed Cases')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# #-----(COUNTRIES WITH THE MOST DEATHS)----------

grouped_df = df.groupby('Country_Region')['Deaths'].sum().reset_index()
sorted_df = grouped_df.sort_values(by='Deaths', ascending=False)
top_15_countries = sorted_df.head(15)

plt.figure(figsize=(10, 6))
plt.bar(top_15_countries['Country_Region'], top_15_countries['Deaths'], color='skyblue')
plt.xlabel('Country')
plt.ylabel('Deaths')
plt.title('Top 15 Countries with Most Deaths')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# #-----(COUNTRIES WITH HIGHEST FATALITY RATE)----------


df['Fatality_Ratio'] = (df['Deaths'] / df['Confirmed']) * 100

country_fatality = df.groupby('Country_Region')['Fatality_Ratio'].mean().reset_index()
country_fatality = country_fatality.sort_values(by='Fatality_Ratio', ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.bar(country_fatality['Country_Region'], country_fatality['Fatality_Ratio'], color='skyblue')
plt.xlabel('Country')
plt.ylabel('Fatality Ratio (%)')
plt.title('Top 15 Countries with Highest Fatality Ratios to Confirmed Cases')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#-----(PROVINCES WITH HIGHEST FATALITY RATE)---------- (FIX ISSUE WITH UNKNOWN CASE!!)


# df['Fatality_Ratio'] = (df['Deaths'] / df['Confirmed']) * 100

# country_fatality = df.groupby('Province_State')['Fatality_Ratio'].mean().reset_index()
# country_fatality = country_fatality.sort_values(by='Fatality_Ratio', ascending=False).head(15)

# plt.figure(figsize=(10, 6))
# plt.bar(country_fatality['Province_State'], country_fatality['Fatality_Ratio'], color='skyblue')
# plt.xlabel('Country')
# plt.ylabel('Fatality Ratio (%)')
# plt.title('Top 15 Provinces with Highest Fatality Ratios to Confirmed Cases')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()

df['Fatality_Ratio'] = (df['Deaths'] / df['Confirmed']) * 100

# Group by Province_State and calculate mean Fatality Ratio
country_fatality = df.groupby('Province_State')['Fatality_Ratio'].mean().reset_index()
country_fatality = country_fatality.sort_values(by='Fatality_Ratio', ascending=False).head(15)

# Call the function
plot_bar_chart(country_fatality['Province_State'], country_fatality['Fatality_Ratio'], 'Country', 'Fatality Ratio (%)', 'Top 15 Provinces with Highest Fatality Ratios to Confirmed Cases')


# #-----(COUNTRIES WITH HIGHEST INCIDENT RATE)----------


df_sorted = df.sort_values(by='Incident_Rate', ascending=False)

top_15_countries = df_sorted.head(15)

# plt.figure(figsize=(10, 6))
# plt.bar(top_15_countries['Country_Region'], top_15_countries['Incident_Rate'], color='skyblue')
# plt.xlabel('Country')
# plt.ylabel('Incidence Rate')
# plt.title('Top 15 Countries with Highest Incidence Rates')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()


plot_bar_chart(top_15_countries['Country_Region'], top_15_countries['Incident_Rate'], 'Country', 'Incidence Rate', 'Top 15 Countries with Highest Incidence Rates')



#-----(PROVINCES WITH HIGHEST INCIDENT RATE)----------


df_sorted = df.sort_values(by='Incident_Rate', ascending=False)

top_15_provinces = df_sorted.head(22)

plt.figure(figsize=(10, 6))
plt.bar(top_15_provinces['Province_State'], top_15_provinces['Incident_Rate'], color='skyblue')
plt.xlabel('Country')
plt.ylabel('Incidence Rate')
plt.title('Top 15 Provinces with Highest Incidence Rates')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

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

#------------------------------------

# # Read location and cases data
# location_df = pd.read_csv("dataset/location_2021.csv")
# cases_df = pd.read_csv("dataset/cases_2021_test.csv")

# cases_df.fillna(0, inplace=True)  # Replace missing values in cases dataset with 0
# location_df.fillna(0, inplace=True)  # Replace missing values in location dataset with 0


# # Merge location and cases data
# merged_df = pd.merge(cases_df, location_df, how="inner", left_on=["country", "province"], right_on=["Country_Region", "Province_State"])

# # Calculate Expected Mortality Rate
# # merged_df["Expected_Mortality_Rate"] = merged_df["Deaths"] / merged_df["Confirmed"]

# # Group by country and calculate total deaths and total confirmed cases
# country_stats = merged_df.groupby('country').agg({'Deaths': 'sum', 'Confirmed': 'sum'})

# # Calculate Expected Mortality Rate per country
# country_stats['Expected_Mortality_Rate'] = country_stats['Deaths'] / country_stats['Confirmed']

# # Display the results
# print(country_stats)

# merged_df = pd.merge(merged_df, country_stats[['Expected_Mortality_Rate']], how='left', left_on='country', right_index=True)


# # Read shapefile
# SHAPEFILE = 'shapefiles/worldmap/ne_10m_admin_0_countries.shp'
# geo_df = gpd.read_file(SHAPEFILE)[['ADMIN', 'ADM0_A3', 'geometry']]
# geo_df.columns = ['country', 'country_code', 'geometry']


# # Merge GeoDataFrame with COVID-19 data
# merged_geo_df = geo_df.merge(merged_df, left_on='country', right_on='Country_Region')

# print(merged_geo_df)

# print(merged_geo_df.columns)



# desired_min = 0
# desired_max = 180  # Change this to your desired maximum value

# # Plotting
# fig, ax = plt.subplots(figsize=(20, 8))
# merged_df.plot(column='Long_', ax=ax, linewidth=1, cmap='viridis', legend=True, legend_kwds={'label': "Longitude"}, vmin=desired_min, vmax=desired_max)
# ax.set_title('Longitude by Country Representation', fontdict={'fontsize': '25', 'fontweight': '3'})
# plt.show()