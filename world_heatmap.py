import pandas as pd
import geopandas as gpd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import country_converter as coco
from datetime import datetime, timedelta


# Read Location dataset
location_df = pd.read_csv("dataset/location_2021.csv")

# Read shapefile
SHAPEFILE = 'shapefiles/worldmap/ne_10m_admin_0_countries.shp'
geo_df = gpd.read_file(SHAPEFILE)[['ADMIN', 'ADM0_A3', 'geometry']]
geo_df.columns = ['country', 'country_code', 'geometry']

# Drop Antarctica (takes up too much space)
geo_df = geo_df[geo_df['country'] != 'Antarctica']

# Merge COVID-19 data with GeoDataFrame
merged_df = geo_df.merge(location_df, left_on='country', right_on='Country_Region')

# Plotting
fig, ax = plt.subplots(figsize=(20, 8))
merged_df.plot(column='Deaths', ax=ax, linewidth=1, cmap='viridis', legend=True, legend_kwds={'label': "Deaths"})
ax.set_title('Daily COVID-19 Deaths', fontdict={'fontsize': '25', 'fontweight': '3'})
plt.show()
