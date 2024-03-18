import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns

locations_df = pd.read_csv('dataset/location_2021.csv')
cases_train_df = pd.read_csv('dataset/cases_2021_train.csv')
countries_continents_df = pd.read_csv('dataset/Countries by continents.csv')
population_df = pd.read_csv('dataset/world_population.csv')

locations_df.head(), cases_train_df.head(), countries_continents_df.head()

# Add continent information based on country name
locations_df = locations_df.merge(countries_continents_df, left_on='Country_Region', right_on='Country', how='left')
cases_train_df = cases_train_df.merge(countries_continents_df, left_on='country', right_on='Country', how='left')

# Calculate data availability by continent using total confirmed cases.
continent_confirmed = locations_df.groupby('Continent')['Confirmed'].sum().sort_values(ascending=False)

# Visualize data availability by continent with a bar plot
plt.figure(figsize=(12, 8))
sns.barplot(x=continent_confirmed.index, y=continent_confirmed.values, palette="coolwarm", hue=continent_confirmed.index, dodge=False)
plt.title('Data Availability by Continent - Total Confirmed Cases')
plt.xlabel('Continent')
plt.ylabel('Total Confirmed Cases')
plt.xticks(rotation=45)
plt.show()

population_df.head()

# Extract relevant population data: country name and population in 2022
population_relevant_df = population_df[['Country/Territory', '2022 Population']]

# Merge population data with locations_df
location_with_pop_df = locations_df.merge(population_relevant_df, left_on='Country_Region', right_on='Country/Territory', how='left')

# Calculate confirmed cases per capita for each country
location_with_pop_df['Confirmed_per_Capita'] = location_with_pop_df['Confirmed'] / location_with_pop_df['2022 Population']

# Check top 10 countries by confirmed cases per capita
top_countries_confirmed_per_capita = location_with_pop_df.sort_values(by='Confirmed_per_Capita', ascending=False).head(10)

top_countries_confirmed_per_capita[['Country_Region', 'Confirmed', '2022 Population', 'Confirmed_per_Capita']]

# Load map data (e.g., country geographic data provided by 'naturalearth_lowres')
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge 'world' DataFrame with COVID-19 data based on country name.
# Use 'location_with_pop_df' DataFrame but ensure all text is in English.
merged_df = world.merge(location_with_pop_df, left_on='name', right_on='Country_Region', how='left')
merged_df['Confirmed_per_Capita'] = merged_df['Confirmed_per_Capita'].fillna(0)

# Visualize map based on confirmed cases per capita
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged_df.plot(column='Confirmed_per_Capita', ax=ax, legend=True,
               legend_kwds={'label': "Confirmed Cases per Capita"},
               cmap='OrRd')  # Adjust color palette as needed.
plt.title('World COVID-19 Confirmed Cases per Capita')
plt.show()
