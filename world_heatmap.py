import pandas as pd
import geopandas as gpd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import country_converter as coco
from datetime import datetime, timedelta


# Read location and cases data
location_df = pd.read_csv("dataset/location_2021.csv")
cases_df = pd.read_csv("dataset/cases_2021_test.csv")

# Merge location and cases data
merged_df = location_df.merge(cases_df, left_on='Country_Region', right_on='country')

# Read shapefile
SHAPEFILE = 'shapefiles/worldmap/ne_10m_admin_0_countries.shp'
geo_df = gpd.read_file(SHAPEFILE)[['ADMIN', 'ADM0_A3', 'geometry']]
geo_df.columns = ['country', 'country_code', 'geometry']


# Merge GeoDataFrame with COVID-19 data
merged_geo_df = geo_df.merge(location_df, left_on='country', right_on='Country_Region')

desired_min = -180
desired_max = 180  # Change this to your desired maximum value


# Plotting
fig, ax = plt.subplots(figsize=(20, 8))
merged_geo_df.plot(column='Long_', ax=ax, linewidth=1, cmap='viridis', legend=True, legend_kwds={'label': "Longitude"}, vmin=desired_min, vmax=desired_max)
ax.set_title('Longitude by Country Representation', fontdict={'fontsize': '25', 'fontweight': '3'})
plt.show()
