import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.data import Dataset
from tensorflow.estimator import BoostedTreesRegressor
from tensorflow.feature_column import numeric_column
import utils

'''
before scaling, dtype mem reductions
{
    dtype('int8'): Index(['assists', 'boosts', 'DBNOs', 'headshotKills', 'heals', 'killPlace',
       'kills', 'killStreaks', 'maxPlace', 'numGroups', 'revives', 'roadKills',
       'teamKills', 'vehicleDestroys', 'weaponsAcquired'],
      
      dtype='object'), 
      dtype('float16'): Index([
        'damageDealt'
        'longestKill'
        'rideDistance'
        'swimDistance'
        'walkDistance'
        'winPlacePerc'])

      dtype('int16'): Index(['killPoints', 'matchDuration', 'rankPoints', 'winPoints'], dtype='object')}
'''


def create_feature_columns(feature_names):
    feature_columns = []
    for feature in feature_names:
        feature_columns.append(numeric_column(key=feature, dtype=tf.float16))
    return []

def prep_data(x, y):
    return (dict({'stats': x}), y)

def input_fn(X, y, batch_size=128, shuffle=False):
    dataset = Dataset.from_tensor_slices((X, y))
    dataset = dataset.map(lambda x, y: prep_data(x, y))

    if shuffle:
        dataset = dataset.shuffle(batch_size)
    dataset = dataset.batch(batch_size)

    return dataset

def build_BoostedTreesClassifier():
    feature_columns = [tf.feature_column.numeric_column(key='stats', shape=(24,), dtype=tf.float16)]
    params = {
        'n_trees': 50,
        'max_depth': 3,
        'n_batches_per_layer': 1,
        'model_dir': 'models/BoostedTrees',
        'center_bias': True
    }
    return BoostedTreesRegressor(feature_columns, **params)


NUMERIC_COLS = [ 'assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'matchDuration', 'maxPlace',
       'numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills',
       'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',
       'weaponsAcquired', 'winPoints']
DROP_COLS = ['Id', 'groupId', 'matchId', 'matchType']
TARGET = 'winPlacePerc'
FEATURE_COLUMNS = create_feature_columns(NUMERIC_COLS)

train = pd.read_csv('../data/train.csv', nrows=200000)
train_X, valid_X, train_y, valid_y = utils.load_data(train, TARGET, DROP_COLS)


train_X = utils.scale_features(train_X, NUMERIC_COLS, utils.tanh_scalar)
valid_X = utils.scale_features(valid_X, NUMERIC_COLS, utils.tanh_scalar)

train_X = utils.reduce_mem(train_X)
valid_X = utils.reduce_mem(valid_X)

classifier = build_BoostedTreesClassifier()

for _ in range(60):
    classifier.train(
        lambda: input_fn(train_X, train_y, shuffle=True),
        max_steps=100,
    )
    classifier.evaluate(
        lambda: input_fn(valid_X, valid_y),
        steps=100 
    )
# print(len(feature_columns))
# print(train_X.dtypes, train_X.shape)
# print(valid_X.dtypes, valid_X.shape)
