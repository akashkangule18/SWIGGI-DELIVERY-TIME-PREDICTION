import pandas as pd
import numpy as np
import os
import seaborn as sns
import yaml
import mlflow
import dagshub



# getting proceessed data
df = pd.read_csv("./data/processed/data.csv")

# converting back to datetime features
df["order_date"] = pd.to_datetime(df["order_date"])
df["order_time"] = pd.to_datetime(df["order_time"])
df["order_picked_time"] = pd.to_datetime(df["order_picked_time"])

# building new feartures
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month
df['month_name'] = df['order_date'].dt.month_name()
df['day'] = df['order_date'].dt.day_name()


# creating the weekend column
def  weekend(column):
    if column == 'Saturday' or column == 'Sunday':
        return 1
    else:
        return 0

df['weekend']= df['day'].apply(weekend)


# now extracting hours and miniuts form the order time and order picked time
# creating new column such order_picked_dur. it will describe that after how many miniuts order was picked
# and again creating time period such as morning, afternon, evening and night

df['ordered_hour'] = df['order_time'].dt.hour
df['ordered_miniut'] = df['order_time'].dt.minute
df['order_picked_time_minute'] = df['order_picked_time'].dt.minute

# new column order_picked_dura
df['order_picked_in_minute'] = df['order_picked_time_minute'] - df['ordered_miniut']


# new column time duration
def time(hour):

    if hour  < 12:
        return 'morning'
    elif hour <  17:
        return 'afternoon'
    elif hour < 21:
        return 'evening'
    else:
        return 'night'

df['time_duration'] = df['ordered_hour'].apply(time)


# now calculating distance between restarunts and destination form the lattiude adn longitude
# but before that removin nan values from this columns for the clean approach

new_df = df.dropna(subset =['rest_lat','rest_long','destination_lat','destination_long'])


# new_column such as distance
def haversine(lat1, lon1, lat2, lon2):
    # Earth radius in km
    R = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

new_df = df.dropna(subset=[
    'rest_lat','rest_long','destination_lat','destination_long'
]).copy()

new_df['distance_km'] = haversine(
    new_df['rest_lat'],
    new_df['rest_long'],
    new_df['destination_lat'],
    new_df['destination_long']
).round(2)

# # now again cleanig the dataset
# # removing unwnated cols
new_df.drop(columns=['rest_lat','rest_long','destination_lat','destination_long','order_date','order_time','order_picked_time', 'year','order_picked_time_minute'], inplace = True)

# and again therr are some negative values in order_pick in minute so fixing the negative values
new_df['order_picked_in_minute'] = np.where(
    new_df['order_picked_in_minute'] < 0,
    new_df['order_picked_in_minute'] + 60,
    new_df['order_picked_in_minute']
)

# lowering all the values  and removing al  the unwnated spaces like trailing and leading spaces in all columns
def lower(column):
    return column.lower().strip()

# again need to convert all the categorical values into lower
new_df['month_name'] = new_df['month_name'].apply(lower)
new_df['day'] = new_df['day'].apply(lower)

# again creating new coulumn as distnce route
def route(distance):
    if distance < 5:
        return 'short'
    elif distance > 5 and distance < 10:
        return 'medium'
    else :
        return 'long'

new_df['distance_route']  = new_df['distance_km'].apply(route)

# dropping mentioned columns
new_df.drop(columns =['time_duration','ordered_miniut','order_picked_in_minute','month_name','day','order_type','distance_route'],inplace = True)

# dropping the null values and running exp- 1 AS experiment without imputation
new_df.dropna(inplace = True)

# dropping 
new_df.drop(columns = ['rider_id','ID'], inplace=True)

print(new_df.columns)


# now saving new_df in proceessed folder 

data_path = os.path.join("data","interim")
os.makedirs(data_path, exist_ok= True)
new_df.to_csv(os.path.join(data_path,"new_df.csv"), index=False)
