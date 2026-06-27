import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow 
import dagshub
import pickle 
import os
import yaml
import json


# fetching training and testing data

train_data = pd.read_csv("./data/external/train_data.csv")
test_data = pd.read_csv("./data/external/test_data.csv")

X_test = test_data.drop(columns =['delivery_time'])
y_test = test_data['delivery_time']

# loading model
with open("models/model.pkl",'rb') as file:
    model = pickle.load(file)


y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print('mean absolute error: ', mae)
print('r2_score',r2)

metrics = {
    'mean absolute error': mae,
    'r2_score': r2
}

# saving results
data_path = os.path.join('reports')
os.makedirs(data_path, exist_ok=True)

with open ('reports/metrics.json','w') as file:
    json.dump(metrics, file, indent=4 )



