# to open and explore data
# initial code taken from https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from sklearn.linear_model import LinearRegression

# check files
import os
for dirname, _, filenames in os.walk('/Users/david/Desktop/kaggle_energy'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# load train files
base_dir = '/Users/david/Desktop/kaggle_energy/'
train_df = pd.read_csv(base_dir + 'train.csv')
train_df["timestamp"] = pd.to_datetime(
    train_df["timestamp"], format='%Y-%m-%d %H:%M:%S')
test_df = pd.read_csv(base_dir + 'test.csv')
test_df["timestamp"] = pd.to_datetime(
    test_df["timestamp"], format='%Y-%m-%d %H:%M:%S')
weather_train_df = pd.read_csv(base_dir + 'weather_train.csv')
weather_test_df = pd.read_csv(base_dir + 'weather_test.csv')
building_meta_df = pd.read_csv(base_dir + 'building_metadata.csv')
sample_submission = pd.read_csv(base_dir + 'sample_submission.csv')

# check dimensions
print('Size of train_df data', train_df.shape)
print('Size of test_df data', test_df.shape)
print('Size of weather_train_df data', weather_train_df.shape)
print('Size of weather_test_df data', weather_train_df.shape)
print('Size of building_meta_df data', building_meta_df.shape)

# Function to reduce the DF size


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# Reducing memory
train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)
weather_train_df = reduce_mem_usage(weather_train_df)
weather_test_df = reduce_mem_usage(weather_test_df)
building_meta_df = reduce_mem_usage(building_meta_df)

# Merge train frames together
train_site_df = weather_train_df.join(
    building_meta_df, on='site_id', rsuffix='_dup')
full_train_df = train_df.join(train_site_df, on='building_id', lsuffix='_dup')
full_train_df.drop(list(full_train_df.filter(
    regex='_dup$')), axis=1, inplace=True)

# Merge test frames together
test_site_df = weather_test_df.join(
    building_meta_df, on='site_id', rsuffix='_dup')
full_test_df = test_df.join(test_site_df, on='building_id', lsuffix='_dup')
full_test_df.drop(list(full_test_df.filter(
    regex='_dup$')), axis=1, inplace=True)

# delete unused frames
del train_df, test_df, weather_train_df,\
    weather_test_df, train_site_df, test_site_df, building_meta_df

# fill missing values with 0
full_train_df.fillna(0, inplace=True)
full_test_df.fillna(0, inplace=True)


# Run regression using temp
x_cols = ['air_temperature']
target = 'meter_reading'
x_train = full_train_df[x_cols].values
y_train = np.log1p(full_train_df["meter_reading"].values)
x_test = full_test_df[x_cols].values

# run regression, predict, set 0 as minimum
reg = LinearRegression().fit(x_train, y_train)
pred = reg.predict(x_test)
pred = list(map(lambda x: max(x, 0), pred))

# undo log1p transformation
pred = np.expm1(pred)

# create submission data format
submission = pd.read_csv(base_dir + 'sample_submission.csv')
submission['meter_reading'] = pred
submission.loc[submission['meter_reading'] < 0, 'meter_reading'] = 0
attempt = 1
submission.to_csv(base_dir + 'submissions/submission_' +
                  str(attempt) + '.csv', index=False)
