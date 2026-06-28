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
from logger import logger

def load_data(training_data_path, testing_data_path):
    try:
        train_data = pd.read_csv(training_data_path)
        test_data = pd.read_csv(testing_data_path)
        logger.info('training and testing data load sucessfully')
        return train_data, test_data
    except Exception as e:
        logger.error(f"check the data path of both training and testing")
        raise



# load model
def load_model(model_path):
    try:
        with open(model_path,'rb') as file:
            model = pickle.load(file)
        logger.info('the model load sucessfully')
        return model
    except Exception as e:
        logger.error(f"check the model path {e}")
        raise



def save_metric(file_path,metric):
    try:
        with open (file_path,'w') as file:
            json.dump(metric, file, indent=4 )
        logger.info('json file saved sucessfully')
    except Exception as e:
        logger.error(f"check the json file path")
        raise



def main():

    # loading train and test data
    train_data, test_data = load_data("./data/external/train_data.csv","./data/external/test_data.csv")
    
    # fetching testing data for model evaluation
    X_test = test_data.drop(columns =['delivery_time'])
    y_test = test_data['delivery_time']

    # load model
    model = load_model("models/model.pkl")

    # predictions
    y_pred = model.predict(X_test)

    # model evaluation
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    print('mean absolute error: ', mae)
    print('r2_score',r2)

    metrics = {
        'mean absolute error': mae,
        'r2_score': r2
    }

    # making data_path for save metrics 
    data_path = os.path.join('reports')
    os.makedirs(data_path, exist_ok=True)

    # save json file
    save_metric('reports/metrics.json', metrics)


if __name__ == '__main__':
    main()

