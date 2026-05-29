"""
train.py - trains and evaluates the Random Forest classifier on NASA Kepler data.
Cleans the dataset, prepares features, trains the model, evaluates accuracy,
and saves the model and feature list to the model/ folder.
"""

import pandas as pd
from sklearn.model_selection import train_test_split # divide a dataset into training and testing subsets
from sklearn.ensemble import RandomForestClassifier # supervised ML alg
from sklearn.metrics import classification_report
import joblib 



df = pd.read_csv('data/kepler_exoplanets.csv')

#cleaning data 
cols = ['koi_period','koi_prad','koi_teq','koi_insol',
        'koi_model_snr','koi_steff','koi_slogg','koi_srad'] #features of the KOI we keep 
print("Nb of rows in the initial table")
rows_before = df.shape[0]
print(rows_before)

df = df.dropna(subset=cols) #drop all NULL values 
print("Nb of the rows after dropping the NULL values (according ot the 8 columns we prioritised)")
print(df.shape[0])
print("Rows removed:", rows_before - df.shape[0])


#initialising input X and output y
df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])] #sorting df to remove all 'CANDIDATE' from koi_disposition
df['koi_disposition'] = df['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0}) #mapping 1 to 'CONFIRMED', 0 to 'FALSE POSITIVE'
X = df[cols] #input data
y = df['koi_disposition'] #output data
print(X.shape)
print(y.shape)
print(y.value_counts())

#training the model 
X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.2, random_state=42
) #splitting the df into training and testing data 

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\n!!!Model trained successfully!!!\n")

#evaluating (prediction on the testing data)
predictions = model.predict(X_test)
print("PREDICTED VALUES")
print(pd.Series(predictions).value_counts())

print("\n")
print("ACTUAL VALUES")
print(y_test.value_counts())

print(classification_report(y_test, predictions, 
      target_names=['False Positive', 'Confirmed Exoplanet']) )

#completion
joblib.dump(model, 'model/model.pkl')
joblib.dump(cols, 'model/features.pkl')
print("\n!!!model saved!!\n")
