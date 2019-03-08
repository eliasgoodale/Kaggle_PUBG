import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.feature_column import numeric_column
from tensorflow.estimator import BoostedTreesRegressor
import utils

tf.enable_eager_execution()

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
            shape=(128, ),
            dtype=tf.float32,
            normalizer_fn=lambda x: utils.tanh_scalar.fit_transform(x)))
    return feature_columns

def input_fn(X, y, batch_size=128, shuffle=False):
    dataset = Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
        dataset = dataset.shuffle(batch_size)
    dataset = dataset.batch(batch_size)
    itr = dataset.make_one_shot_iterator()
    return itr.get_next()

def build_BoostedTreesRegressor(feature_columns):
    params = {
        'n_trees': 50,
        'max_depth': 3,
        'n_batches_per_layer': 1,
        'model_dir': 'models/BoostedTrees',
        'center_bias': True
    }
    return BoostedTreesRegressor(feature_columns, **params)

train = pd.read_csv('../data/train.csv', nrows=200000)

train_X, valid_X, train_y, valid_y = utils.load_data(train, TARGET, DROP_COLS)

fc = create_feature_columns()
estimator = build_BoostedTreesRegressor(fc)

train_input_fn = lambda: input_fn(train_X, train_y, shuffle=True)
valid_input_fn = lambda: input_fn(valid_X, valid_y)

for epoch in range(20):
    estimator.train(input_fn=train_input_fn, steps=100)
    estimator.evaluate(input_fn=valid_input_fn, steps=1)

'''
with tf.Session() as sess:
    input_tensor=sess.run(train_input_fn())
    x, y = input_tensor

    for key, value in x.items():
        print(len(value))
    print(len(y))
'''