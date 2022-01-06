import matplotlib.pyplot as plt
import pandas as pd
# import pylab as pl
import numpy as np
import wget

# %matplotlib inline

url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv"

file = wget.download(url)

df = pd.read_csv(file)

# take a look at the dataset
print(df.head())

