import numpy as np
import pandas as pd
import math
from scipy.stats import zscore
from geopy import distance
def save_data(df, path):
    df.to_csv(path, index=False)

def load_data(file_path):
    return pd.read_csv(file_path)

def drop_column(df, column_name):
    if column_name in df.columns:
        return df.drop(column_name, axis=1)
    return df

def convert_datetime(df, column_name='pickup_datetime'):
    if column_name in df.columns:
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    return df

def factorize_column(df, column_name):
    if column_name in df.columns:
        df[column_name] = pd.factorize(df[column_name])[0]
    return df

def create_time_features(df, datetime_column='pickup_datetime'):
    if datetime_column in df.columns:
        df[datetime_column] = pd.to_datetime(df[datetime_column])  # Ensure datetime format
        df['year'] = df[datetime_column].dt.year
        df['month'] = df[datetime_column].dt.month
        df['day'] = df[datetime_column].dt.day
        df['hour'] = df[datetime_column].dt.hour
        df['minute'] = df[datetime_column].dt.minute
        df['second'] = df[datetime_column].dt.second
        df['day_of_week'] = df[datetime_column].dt.weekday
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['day_of_year'] = df[datetime_column].dt.dayofyear
        df['week_of_year'] = df[datetime_column].dt.isocalendar().week
        df['quarter'] = df[datetime_column].dt.quarter
        df['is_month_start'] = df[datetime_column].dt.is_month_start.astype(int)
        df['is_month_end'] = df[datetime_column].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df[datetime_column].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df[datetime_column].dt.is_quarter_end.astype(int)
        df['is_year_start'] = df[datetime_column].dt.is_year_start.astype(int)
        df['is_year_end'] = df[datetime_column].dt.is_year_end.astype(int)
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)  # Cyclical encoding
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    return df


def log_transform(df, column_name, new_column_name):
    if column_name in df.columns:
        df[new_column_name] = np.log1p(df[column_name])
    return df

def geo_log_transform(df):
    geo_features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    for feature in geo_features:
        if feature in df.columns:
            min_value = df[feature].min() if 'longitude' in feature else 0
            df[f'{feature}_log1p'] = np.log1p(df[feature] - min_value + 1)
    return df

def compute_geopy_distance(df):
    def get_distance_km(row):
        pickup_coords = (row['pickup_latitude'], row['pickup_longitude'])
        dropoff_coords = (row['dropoff_latitude'], row['dropoff_longitude'])
        return distance.geodesic(pickup_coords, dropoff_coords).km

    if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']):
        df['distance'] = df.apply(get_distance_km, axis=1)
        df = log_transform(df, 'distance', 'distance_log1p')
    return df

def compute_manhattan_distance(df):
    def manhattan_distance(row):
        lat_distance = abs(row['pickup_latitude'] - row['dropoff_latitude']) * 111
        lon_distance = abs(row['pickup_longitude'] - row['dropoff_longitude']) * 111 * math.cos(math.radians(row['pickup_latitude']))
        return lat_distance + lon_distance

    if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']):
        df['man_distance'] = df.apply(manhattan_distance, axis=1)
        df = log_transform(df, 'man_distance', 'man_distance_log1p')
    return df

def remove_outliers(df, threshold=3):
    numeric_df = df.select_dtypes(include='number')
    z_scores = numeric_df.apply(zscore)
    mask = (abs(z_scores) <= threshold).all(axis=1)
    return df[mask].reset_index(drop=True)

def prepare_data(
    file_path,
    drop_id=True,
    parse_datetime=True,
    factorize_store_flag=True,
    create_time_features_flag=True,
    drop_datetime=True,
    log_transform_trip_duration=True,
    geo_log_transform_flag=True,
    compute_geopy_distance_flag=True,
    compute_manhattan_distance_flag=True,
    remove_outliers_flag=True,
    outlier_threshold=5
):
    df = load_data(file_path)
    print("data loaded")
    if drop_id:
        df = drop_column(df, 'id')
    print("id dropped")
    if parse_datetime:
        df = convert_datetime(df)
    print("date conversion")
    if factorize_store_flag:
        df = factorize_column(df, 'store_and_fwd_flag')
    print("converting store and fwd flag")
    if create_time_features_flag:
        df = create_time_features(df)
    print("creating time features")
    if drop_datetime:
        df = drop_column(df, 'pickup_datetime')
    print("dropping time")
    if log_transform_trip_duration:
        df = log_transform(df, 'trip_duration', 'trip_duration_log_transform')
    print("log duration")
    if geo_log_transform_flag:
        df = geo_log_transform(df)
    print("log longitude and latitude")
    if compute_geopy_distance_flag:
        df = compute_geopy_distance(df)
    print("compute distance")
    if compute_manhattan_distance_flag:
        df = compute_manhattan_distance(df)
    print("compute man distance")
    if remove_outliers_flag:
        df = remove_outliers(df, outlier_threshold)
    print("remove outliers")
    return df

