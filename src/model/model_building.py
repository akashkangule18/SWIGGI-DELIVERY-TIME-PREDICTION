import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow 
import dagshub
import pickle 
import os
import yaml

# loading params.yaml file
with open('params.yaml','r') as file:
    params = yaml.safe_load(file)

# getting only data spliting params
data_split_params = params['data_spliting']

# optuna params
optuna_params = params['optuna']

# model params 
model_params = params['final_model']



# loading processsed data set
new_df = pd.read_csv("./data/interim/new_df.csv")

# dropping the null values beacuse we ran two experiments and experiment with droppping null values ig giving the better results as compare to 
# null values imputation
new_df.dropna(inplace = True)


# pipeline building
# seperating data 
X = new_df.drop(columns="delivery_time")
y = new_df['delivery_time']

# spliting data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = data_split_params['test_size'], random_state = data_split_params['random_state'])

# seperating numerical and categorical columns
num_cols = X.select_dtypes(include=['int64','float64','int32']).columns
ohe_cols = ['weather_cond', 'vehicle_type', 'festival', 'city_type','city_name']
ord_cols = ['traffic']

# getting necessary libraries
from sklearn.preprocessing import RobustScaler, PowerTransformer,OrdinalEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ignoring the warnings
import warnings
warnings.filterwarnings('ignore')


# ordinal encoder
ord_encoder = OrdinalEncoder(
    categories=[['low','jam','medium','high']],
    handle_unknown= "use_encoded_value",
    unknown_value= -1
)

# buiilding pipeline
num_pipe = Pipeline([
    ('transformer',PowerTransformer(method="yeo-johnson")),
    ('scaler',RobustScaler())
])

ord_pipe = Pipeline([
    ('ord_enc',ord_encoder)
])

ohe_pipe = Pipeline([
    ('ohe_encoder',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False))
])

transformer = ColumnTransformer([
    ('num',num_pipe,num_cols),
    ('ord',ord_pipe,ord_cols),
    ('ohe',ohe_pipe,ohe_cols)
])

pipeline = Pipeline([
    ('preprocess',transformer),
    ('regressor',XGBRegressor())
])


pipeline.fit(X_train,y_train)


# mkaing clone of this pipeline as base pipeline
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
base_pipeline = clone(pipeline)


# using optuna

import optuna

def objective(trial):
    model_name = trial.suggest_categorical('model',['RFR','DTR','XGBR','LGBMR'])

    # runnig if else conditions
    if model_name == 'RFR':
        model = RandomForestRegressor(
            n_estimators= trial.suggest_int('n_estimators',100,300),
            max_depth= trial.suggest_int('max_depth',3,7),
            min_samples_split= trial.suggest_int('min_samples_split',2,5),
            min_samples_leaf= trial.suggest_int('min_sample_leaf',10,50),
            min_impurity_decrease= trial.suggest_float('min_impurity_decrease',0.01,0.1),
            n_jobs=-1
        )


    elif model_name == 'DTR':
        model = DecisionTreeRegressor(
            max_depth= trial.suggest_int('max_depth',2,7),
            min_samples_split= trial.suggest_int('min_samples_split',2,5),
            min_samples_leaf= trial.suggest_int('min_samples_float',10,50),
            min_impurity_decrease= trial.suggest_float('min_impurity_decrease',0.01,0.1)
        )

    elif model_name == 'XGBR':
        model = XGBRegressor(
            n_estimators = trial.suggest_int('n_estimators',100,300),
            max_depth = trial.suggest_int('max_depth',2,7),
            learning_rate = trial.suggest_float('learning_rate',0.05,0.1),
            min_child_weight = trial.suggest_int('min_child_weight',3,10),
            subsample = trial.suggest_float('sub_sample',0.5,1),
            colsample_bytree = trial.suggest_float('colsample_bytree',0.5,1),
            gamma = trial.suggest_float('gamma',0.1,0.5)
        )

    else:
        model = LGBMRegressor(
            num_leaves=trial.suggest_int('num_leaves', 20, 100),
            min_child_samples = trial.suggest_int('min_child_samples',10, 50),
            max_depth=trial.suggest_int('max_depth', 1, 15),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1),
            n_estimators=500,
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 0.0, 1.0),
            reg_lambda=trial.suggest_float('reg_lambda', 0.0, 1.0),
            random_state=42,
            n_jobs=-1,
            verbose = -1
        )


    # again clome the base pipelinne
    new_pipe = clone(base_pipeline)
    new_pipe.set_params(
        regressor = model
    )

    
    scores = cross_val_score(
        new_pipe,
        X_train,
        y_train,
        cv = optuna_params['cv'],
        scoring= 'neg_mean_absolute_error'
    )

    return -scores.mean()


# creating optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=optuna_params['n_trials'])       

# fetching best parametrs and its value
print('best_params: ', study.best_params)
print('best value: ', study.best_value)

best_params = study.best_params.copy()
best_params.pop("model")

# building the final model
final_model = LGBMRegressor(
    **best_params,
    n_estimators= model_params['n_estimators'],
    random_state=model_params['random_state'],
    n_jobs=-1,
    verbose=-1
)

# cloning the base pipeline
LGRregressor = clone(base_pipeline)
LGRregressor.set_params(
        regressor = final_model
)

LGRregressor.fit(X_train,y_train)

# model saving
import os

os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as file:
    pickle.dump(LGRregressor, file)


# saving trainig and testing data
train_data = X_train
train_data['delivery_time'] = y_train

test_data = X_test
test_data['delivery_time'] = y_test

# again saving data
data_path = os.path.join('data','external')
os.makedirs(data_path,exist_ok=True)

train_data.to_csv(os.path.join(data_path,'train_data.csv'), index = False)
test_data.to_csv(os.path.join(data_path,'test_data.csv'),index = False)





