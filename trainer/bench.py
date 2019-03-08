import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.feature_column import numeric_column
from tensorflow.estimator import BoostedTreesRegressor
import utils
import time


FEATURE_COLS = [ 'assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'matchDuration', 'maxPlace',
       'numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills',
       'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',
       'weaponsAcquired', 'winPoints']
DROP_COLS = ['Id', 'groupId', 'matchId', 'matchType']
TARGET = 'winPlacePerc'


def create_feature_columns(feature_names=FEATURE_COLS):
    feature_columns = []
    for feature in feature_names:
        feature_columns.append(numeric_column(
            key=feature,
            dtype=tf.float32))
    return feature_columns

def input_fn(X, y, batch_size=128, shuffle=False, test=False):
    dataset = Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
        dataset = dataset.shuffle(batch_size)
    dataset = dataset.batch(batch_size)
    return dataset

def build_BoostedTreesRegressor(feature_columns, data_len, batch_size=128):
    params = {
        'n_trees': 50,
        'max_depth': 13,
        'n_batches_per_layer': 1,
        'model_dir': f'models/BoostedTrees/{time.time()}',
        'center_bias': True
    }
    return BoostedTreesRegressor(feature_columns, **params)

train = pd.read_csv('../data/train.csv', nrows=200000)

train_X, valid_X, train_y, valid_y = utils.load_data(train, TARGET, DROP_COLS)
train_X = utils.scale_features(train_X, FEATURE_COLS, utils.tanh_scalar)
valid_X = utils.scale_features(valid_X, FEATURE_COLS, utils.tanh_scalar)
train_X = utils.reduce_mem(train_X)
valid_X = utils.reduce_mem(valid_X)

fc = create_feature_columns()

estimator = build_BoostedTreesRegressor(fc,len(train_X))

train_input_fn = lambda: input_fn(train_X, train_y, shuffle=True)
valid_input_fn = lambda: input_fn(valid_X, valid_y)


for _ in range(1):
    estimator.train(train_input_fn, steps=1)
    results = estimator.evaluate(valid_input_fn, steps=1)
    print(pd.Series(results).to_frame())


test = pd.read_csv('../data/test.csv', nrows=10)

pred_labels = test.pop('Id')
test_X = utils.scale_features(test[FEATURE_COLS], FEATURE_COLS, utils.tanh_scalar)
test_X = utils.reduce_mem(test_X)

pred_input_fn = lambda: Dataset.from_tensors(dict(test_X))

pred_dicts = list(estimator.experimental_predict_with_explanations(pred_input_fn))

for pred in pred_dicts:
    for key, value in pred.items():
        print(f"{key} => {value}")

















#result = estimator.evaluate(valid_input_fn)
'''
for feature_batch, label_batch in ds.take(1):
  print('Some feature keys:', list(feature_batch.keys()))
  print()
  print('A batch of class:', feature_batch['assists'].numpy())
  print()
  print('A batch of Labels:', label_batch.numpy())  
'''
#for key, value in x.items():
#     print(f'{key} => {value}')

