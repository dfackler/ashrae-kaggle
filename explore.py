# to open and explore data
# initial code taken from notebooks
# https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction
# https://www.kaggle.com/ryches/simple-lgbm-solution

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm

# check files
import os
for dirname, _, filenames in os.walk('/Users/david/Desktop/kaggle_energy'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# load train files
base_dir = '/Users/david/Desktop/kaggle_energy/'
train_df = pd.read_csv(base_dir + 'train.csv')
test_df = pd.read_csv(base_dir + 'test.csv')
weather_train_df = pd.read_csv(base_dir + 'weather_train.csv')
weather_test_df = pd.read_csv(base_dir + 'weather_test.csv')
building_df = pd.read_csv(base_dir + 'building_metadata.csv')
sample_submission = pd.read_csv(base_dir + 'sample_submission.csv')

# check dimensions
print('Size of train_df data', train_df.shape)
print('Size of test_df data', test_df.shape)
print('Size of weather_train_df data', weather_train_df.shape)
print('Size of weather_test_df data', weather_train_df.shape)
print('Size of building_df data', building_df.shape)

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
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
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
building_df = reduce_mem_usage(building_df)

# Merge train frames together
train_df = train_df.merge(building_df,
                          left_on="building_id",
                          right_on="building_id",
                          how="left")
train_df = train_df.merge(weather_train_df,
                          left_on=["site_id", "timestamp"],
                          right_on=["site_id", "timestamp"],
                          how="left")
# train_df.drop(list(train_df.filter(
#    regex='_dup$')), axis=1, inplace=True)
del weather_train_df

# Merge test frames together
test_df = test_df.merge(building_df,
                        left_on="building_id",
                        right_on="building_id",
                        how="left")
test_df = test_df.merge(weather_test_df,
                        left_on=["site_id", "timestamp"],
                        right_on=["site_id", "timestamp"],
                        how="left")
# test_df.drop(list(test_df.filter(
#    regex='_dup$')), axis=1, inplace=True)
del weather_test_df, building_df

# create time features
train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
train_df["hour"] = train_df["timestamp"].dt.hour
train_df["day"] = train_df["timestamp"].dt.day
train_df["weekday"] = train_df["timestamp"].dt.weekday
train_df["month"] = train_df["timestamp"].dt.month

# encode and define features
le = LabelEncoder()
train_df["primary_use"] = le.fit_transform(train_df["primary_use"])

categoricals = [
    "building_id", "primary_use", "hour", "day", "weekday", "month", "meter"
]
drop_cols = [
    "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed"
]
numericals = [
    "square_feet", "year_built", "air_temperature", "cloud_coverage",
    "dew_temperature"
]
feat_cols = categoricals + numericals

# set and transform target to handle long tail
target = np.log1p(train_df["meter_reading"])

# drop columns
del train_df["meter_reading"]
train_df = train_df.drop(drop_cols + ["site_id", "floor_count"], axis=1)

# Run LightGBM
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=False, random_state=123)
error = 0
models = []
for i, (train_index, val_index) in enumerate(kf.split(train_df)):
    if i + 1 < num_folds:
        continue
    print(train_index.max(), val_index.min())
    train_X = train_df[feat_cols].iloc[train_index]
    val_X = train_df[feat_cols].iloc[val_index]
    train_y = target.iloc[train_index]
    val_y = target.iloc[val_index]
    lgb_train = lgb.Dataset(train_X, train_y > 0)
    lgb_eval = lgb.Dataset(val_X, val_y > 0)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }
    gbm_class = lgb.train(params,
                          lgb_train,
                          num_boost_round=2000,
                          valid_sets=(lgb_train, lgb_eval),
                          early_stopping_rounds=20,
                          verbose_eval=20)

    lgb_train = lgb.Dataset(train_X[train_y > 0], train_y[train_y > 0])
    lgb_eval = lgb.Dataset(val_X[val_y > 0], val_y[val_y > 0])
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'rmse'},
        'learning_rate': 0.5,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }
    gbm_regress = lgb.train(params,
                            lgb_train,
                            num_boost_round=2000,
                            valid_sets=(lgb_train, lgb_eval),
                            early_stopping_rounds=20,
                            verbose_eval=20)
    #     models.append(gbm)

    y_pred = (gbm_class.predict(val_X, num_iteration=gbm_class.best_iteration) > .5) *\
    (gbm_regress.predict(val_X, num_iteration=gbm_regress.best_iteration))
    error += np.sqrt(mean_squared_error(y_pred, (val_y))) / num_folds
    print(np.sqrt(mean_squared_error(y_pred, (val_y))))
    break
print(error)

# setup test set
test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
test_df["hour"] = test_df["timestamp"].dt.hour
test_df["day"] = test_df["timestamp"].dt.day
test_df["weekday"] = test_df["timestamp"].dt.weekday
test_df["month"] = test_df["timestamp"].dt.month
test_df["primary_use"] = le.fit_transform(test_df["primary_use"])
test_df = test_df[feat_cols]

# predict on test set (with progress bar)
i = 0
pred = []
step_size = 50000
for j in tqdm(range(int(np.ceil(test_df.shape[0] / 50000)))):
    res.append(
        np.expm1(
            (gbm_class.predict(test_df.iloc[i:i + step_size],
                               num_iteration=gbm_class.best_iteration) > .5) *
            (gbm_regress.predict(test_df.iloc[i:i + step_size],
                                 num_iteration=gbm_regress.best_iteration))))
    i += step_size

# clear up data and combine pred
del test_df
res = np.concatenate(res)

# create submission data format
submission = pd.read_csv(base_dir + 'sample_submission.csv')
submission['meter_reading'] = res
submission.loc[submission['meter_reading'] < 0, 'meter_reading'] = 0
attempt = 2
submission.to_csv(base_dir + 'submissions/submission_' + str(attempt) + '.csv',
                  index=False)
