import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from utils.config import *


JFK_coords = [40.6106, -73.6688]
LGA_coords = [40.7475, -73.7629]


def log_trip_duration(df):
    df['trip_duration_log'] = np.log(1 + df['trip_duration'].values)
    df = df.drop('trip_duration', axis=1)
    return df


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h  # km


def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b


def encode_cat(df):
    for f in df.columns:
        if df[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[f].values))
            df[f] = lbl.transform(list(df[f].values))
    return df


def remove_outliers(df):
    # Suggested by headsortails' EDA (https://www.kaggle.com/headsortails/nyc-taxi-rides-eda)
    # remove trip_durations longer than 22 hours.
    df = df[df.trip_duration < 22*3600]
    # lower trip_duration limit of 10 seconds
    df = df[(df.trip_duration > 10)]
    # remove those zero-distance trips that took more than a minute.
    df = df.drop(df[(df['haversine_dist'] == 0) & (df['trip_duration'] > 60)].index)
    # remove pickup or dropoff locations more than 300 km away from NYC
    df = df[df['dist_JFK_pickup'] < 300]
    # speed limit of 100 km/h.
    df = df[(df.speed <= 100)]
    return df


def engineer_features(df, train=True):
    # Suggested by beluga's (gaborfodor) EDA (https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-376)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['pickup_hour'] = df.pickup_datetime.dt.hour
    df['day_week'] = df.pickup_datetime.dt.weekday
    df['haversine_dist'] = haversine_array(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])
    df['manhattan_dist'] = dummy_manhattan_distance(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])
    df['pickup_hour_weekofyear'] = df['pickup_datetime'].dt.weekofyear
    coords = np.vstack((df[['pickup_latitude', 'pickup_longitude']].values,
                        df[['dropoff_latitude', 'dropoff_longitude']].values))
    pca = PCA().fit(coords)
    df['pickup_pca0'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 0]
    df['pickup_pca1'] = pca.transform(df[['pickup_latitude', 'pickup_longitude']])[:, 1]
    df['dropoff_pca0'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    df['dropoff_pca1'] = pca.transform(df[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

    # Suggested by headsortails' EDA (https://www.kaggle.com/headsortails/nyc-taxi-rides-eda)
    if train:
        df['speed'] = df['haversine_dist']/(df['trip_duration'] / 3600)  # km/h
    df['dist_JFK_pickup'] = haversine_array(df['pickup_latitude'], df['pickup_longitude'], JFK_coords[0], JFK_coords[1])
    df['dist_JFK_dropoff'] = haversine_array(JFK_coords[0], JFK_coords[1], df['dropoff_latitude'], df['dropoff_longitude'])
    df['dist_LGA_pickup'] = haversine_array(df['pickup_latitude'], df['pickup_longitude'], LGA_coords[0], LGA_coords[1])
    df['dist_LGA_dropoff'] = haversine_array(LGA_coords[0], LGA_coords[1], df['dropoff_latitude'], df['dropoff_longitude'])
    df['trip_to_JFK'] = (df['dist_JFK_dropoff'] < 2) * 1
    df['trip_from_JFK'] = (df['dist_JFK_pickup'] < 2) * 1
    df['trip_from_LGA'] = (df['dist_LGA_pickup'] < 2) * 1
    df['trip_to_LGA'] = (df['dist_LGA_dropoff'] < 2) * 1
    df['work'] = ((df['day_week'] < 5) & (df['pickup_hour'] > 8) & (df['pickup_hour'] < 18)) * 1
    df['log_trip_duration'] = df.trip_duration.apply(np.log)

    return df


def select_features(df, train=True):
    if train:
        return df[[
            'passenger_count',
            'vendor_id',
            'store_and_fwd_flag',
            'pickup_hour',
            'day_week',
            'pickup_hour_weekofyear',
            'pickup_pca0',
            'pickup_pca1',
            'dropoff_pca0',
            'dropoff_pca1',
            'manhattan_dist',
            'haversine_dist',
            'trip_to_JFK',
            'trip_from_JFK',
            'trip_from_LGA',
            'trip_to_LGA',
            'work',
            'trip_duration',
            'log_trip_duration',
        ]]
    else:
        return df[[
            'passenger_count',
            'vendor_id',
            'store_and_fwd_flag',
            'pickup_hour',
            'day_week',
            'pickup_hour_weekofyear',
            'pickup_pca0',
            'pickup_pca1',
            'dropoff_pca0',
            'dropoff_pca1',
            'manhattan_dist',
            'haversine_dist',
            'trip_to_JFK',
            'trip_from_JFK',
            'trip_from_LGA',
            'trip_to_LGA',
            'work',
        ]]


def process(df, train=True):
    if train:
        df = engineer_features(df)
        df = remove_outliers(df)
        df = select_features(df)
    else:
        df = engineer_features(df, train=False)
        df = select_features(df, train=False)
    df = encode_cat(df)
    return df


def main():
    train_df = pd.read_csv(RAW_TRAINING_SET_FILE)
    train_df = process(train_df)
    train_df.to_csv(TRAINING_SET_FILE, index=False)
    test_df = pd.read_csv(RAW_TEST_SET_FILE)
    test_df = process(test_df, train=False)
    test_df.to_csv(TEST_SET_FILE, index=False)

if __name__ == '__main__':
        main()
