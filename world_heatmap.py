import pandas as pd
import geopandas as gpd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import country_converter as coco
from datetime import datetime, timedelta



covid_df = pd.read_csv("dataset/location_2021.csv")
print(covid_df.head(3))