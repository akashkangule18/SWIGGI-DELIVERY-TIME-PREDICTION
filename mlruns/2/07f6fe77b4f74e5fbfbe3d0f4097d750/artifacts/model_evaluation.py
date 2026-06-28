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

# load yaml file
def load_yaml(file_path):
    try:
        with open(file_path,'r') as file:
            yaml_params = yaml.safe_load(file)
        logger.info('yaml.param file load sucessfully')
        return yaml_params
    except Exception as e:
        logger.error(f"please check the params.yaml file path")
        raise


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

    mlflow.set_experiment('Final model register')
    with mlflow.start_run(run_name='model_registr'):
            
            # loading params.yaml file
            yaml_params = load_yaml('params.yaml')

            # loading train and test data
            train_data, test_data = load_data("./data/external/train_data.csv","./data/external/test_data.csv")
            
            # fetching testing data for model evaluation
            X_test = test_data.drop(columns =['delivery_time'])
            y_test = test_data['delivery_time']

            # fetching x_train and y_train for model signature
            X_train = train_data.drop(columns=['delivery_time'])
            y_train = train_data['delivery_time']

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


            # logging data
            training_data = train_data
            training_data = mlflow.data.from_pandas(training_data)

            testing_data = test_data
            testing_data = mlflow.data.from_pandas(testing_data)

            mlflow.log_input(training_data,context='training data')
            mlflow.log_input(testing_data, context='testing_data')


            # logging params
            params = model.named_steps['regressor'].get_params()
            mlflow.log_params(params)

            # logging params.yaml parameters
            mlflow.log_params(yaml_params['data_spliting'])
            mlflow.log_params(yaml_params['optuna'])
            mlflow.log_params(yaml_params['final_model'])

            # logging metrics
            mlflow.log_metric('mean absolute error', mae)
            mlflow.log_metric('r2_score',r2)

            # log file
            mlflow.log_artifact(__file__)
            mlflow.log_artifact('reports/metrics.json')

            # signature
            signature = mlflow.models.infer_signature(X_train,model.predict(X_train))

            # model logging and registering
            mlflow.sklearn.log_model(
                sk_model= model,
                name = 'LGBMRegressor',
                signature= signature,
                registered_model_name='LightGBMRegressor',
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            )


if __name__ == '__main__':
    main()

