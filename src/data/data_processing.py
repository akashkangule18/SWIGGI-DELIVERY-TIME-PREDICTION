import pandas as pd
import numpy as np
import seaborn as sns
import mlflow
import dagshub
import os
import yaml

# loading the data
df = pd.read_csv("./data/raw/uncleaned_swiggy.xls")


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


# lowering all the values  and removing al  the unwnated spaces like trailing and leading spaces in all columns
def lower(column):
    return column.lower().strip()


df['rider_id'] = df['rider_id'].apply(lower)
df['weather_cond'] = df['weather_cond'].apply(lower)
df['traffic'] = df['traffic'].apply(lower)
df['order_type'] = df['order_type'].apply(lower)
df['vehicle_type'] = df['vehicle_type'].apply(lower)
df['festival'] = df['festival'].apply(lower)
df['city_type'] = df['city_type'].apply(lower)


# fixing the nan values from the data from all the columns

# for age
df['age']= df['age'].replace('NaN ', np.nan)

# for rating
df['rating']= df['rating'].replace('NaN ', np.nan)

# for order time
df['order_time']= df['order_time'].replace('NaN ', np.nan)

# for order date
df['order_date']= df['order_date'].replace('nan', np.nan)

# for order picked time
df['order_picked_time']= df['order_picked_time'].replace('NaN ', np.nan)

# for traffic
df['traffic']= df['traffic'].replace('nan', np.nan)

# for vaehical condotion
df['vehicle_cond']= df['vehicle_cond'].replace('NaN ', np.nan)

# for order typpe
df['order_type']= df['order_type'].replace('nan', np.nan)

# for vehicle type
df['vehicle_type']= df['vehicle_type'].replace('nan', np.nan)

# for multiple orders
df['multiple_orders']= df['multiple_orders'].replace('NaN ', np.nan)

# for festivel
df['festival']= df['festival'].replace('nan', np.nan)

# for city type
df['city_type']= df['city_type'].replace('nan', np.nan)

# for weather column
df['weather_cond']= df['weather_cond'].str.split().str.get(1)
df['weather_cond']= df['weather_cond'].replace('nan',np.nan)


# for delivery time
df['delivery_time'] = df['delivery_time'].str.split().str.get(1)

# Changing datatypes
df['age'] = df['age'].astype('float')
df['rating'] = df['rating'].astype('float')
df['multiple_orders'] = df['multiple_orders'].astype(float)


# for datetime columns
df['order_date'] = pd.to_datetime(df['order_date'],dayfirst=True)
df['order_time'] = pd.to_datetime(df['order_time'], format='%H:%M:%S')
df['order_picked_time'] = pd.to_datetime(df['order_picked_time'], format='%H:%M:%S')

# with output column
df['delivery_time'] = df['delivery_time'].astype('int')

# fetching city name
df['city_name'] = df['rider_id'].str.split('res').str.get(0)



# data saving
data_path = os.path.join('data','processed')
os.makedirs(data_path,exist_ok=True)
df.to_csv(os.path.join(data_path,'data.csv'),index=False)
