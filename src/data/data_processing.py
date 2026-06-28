import pandas as pd
import numpy as np
import seaborn as sns
import mlflow
import dagshub
import os
import yaml
from logger import logger


# making function for data loading
def load_data(data_path):
    try:
        df = pd.read_csv(data_path)
        logger.info('data load sucessfully')
        return df
    except Exception as e:
        logger.error(f'please check the data files {e}')
        raise



# lowering all the values  and removing al  the unwnated spaces like trailing and leading spaces in all columns
def lower(value):
    try:
        return value.lower().strip()
    except Exception as e:
        logger.error(f'chek the coding part{e}')
        raise



def replace_nan(df):
    try:
        df.replace(
            {
                r'^\s*NaN\s*$': np.nan,
                r'^\s*nan\s*$': np.nan
            },
            regex=True,
            inplace=True
        )

        logger.info("Missing values replaced successfully.")
        return df

    except Exception as e:
        logger.error(f"Error replacing NaN values: {e}")
        raise




# data saving
def save_data(data_path,df):
    try:
        df.to_csv(os.path.join(data_path,'data.csv'),index=False)
        logger.info('data saved sucessfully')
    except Exception as e:
        logger.error(f'there is no datapath is specified {e}')
        raise
    


def main():
    df = load_data("./data/raw/uncleaned_swiggy.xls")
    
    # saving data 
    # # renames the cols
    df.rename(columns = {'Delivery_person_ID':'rider_id',
                        'Delivery_person_Age':'age',
                        'Delivery_person_Ratings':'rating',
                        'Restaurant_latitude':'rest_lat',
                        'Restaurant_longitude':'rest_long',
                        'Delivery_location_latitude':'destination_lat',
                        'Delivery_location_longitude':'destination_long',
                        'Order_Date':'order_date',
                        'Time_Orderd':'order_time',
                        'Time_Order_picked':'order_picked_time',
                        'Weatherconditions':'weather_cond',
                        'Road_traffic_density':'traffic',
                        'Vehicle_condition':'vehicle_cond',
                        'Type_of_order':'order_type',
                        'Type_of_vehicle':'vehicle_type',
                        'multiple_deliveries':'multiple_orders',
                        'Festival':'festival',
                        'City':'city_type',
                        'Time_taken(min)':'delivery_time'
                        }, inplace = True
            )
    
    # mkaing all the capital characters in lowercase
    cols = [
    "rider_id",
    "weather_cond",
    "traffic",
    "order_type",
    "vehicle_type",
    "festival",
    "city_type"]

    for col in cols:
         df[col] = df[col].apply(lower)


    # spliting
    df['weather_cond']= df['weather_cond'].str.split().str.get(1)

    # for delivery time
    df['delivery_time'] = df['delivery_time'].str.split().str.get(1)

    # fetching city name
    df['city_name'] = df['rider_id'].str.split('res').str.get(0)

    # replacing the nan values
    df = replace_nan(df)

    # changing data type
    df['age'] = df['age'].astype('float')
    df['rating'] = df['rating'].astype('float')
    df['multiple_orders'] = df['multiple_orders'].astype(float)
    df['delivery_time'] = df['delivery_time'].astype('int')

    # # for datetime columns
    df['order_date'] = pd.to_datetime(df['order_date'],dayfirst=True)
    df['order_time'] = pd.to_datetime(
    df['order_time'],
    format='%H:%M:%S',
    errors='coerce'
)
    df['order_picked_time'] = pd.to_datetime(df['order_picked_time'], format='%H:%M:%S', errors='coerce')

    # data_path
    data_path = os.path.join('data','processed')
    os.makedirs(data_path,exist_ok=True)

    # saving data
    save_data(data_path,df)
    
    
if __name__ == "__main__":
    main()