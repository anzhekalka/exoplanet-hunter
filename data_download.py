import pandas as pd
url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"
df = pd.read_csv(url) #goes to the url, downloads the table, stores in df (DataFrame)
df.to_csv('data/kepler_exoplanets.csv', index=False) #saves df to the file

print(df.shape)           # ~10,000 rows
print(df['koi_disposition'].value_counts())
# CONFIRMED / CANDIDATE / FALSE POSITIVE