import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
import keras
from tensorboard_customization import TrainValTensorBoard
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Dropout, Input
from keras.regularizers import L1L2
import time

'''
so we are doing PUBG challenge
I need 4 things to be submited:
1) EDA
2) keras API model
3) estimator API model
4) boosted trees model
please don't forget about Tensorboard  screenshots and kaggle submissions

Index([ 'assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'matchDuration', 'matchType', 'maxPlace',
       'numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills',
       'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',
       'weaponsAcquired', 'winPoints'],
      dtype='object')
Index(['Id', 'groupId', 'matchId', 'assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'matchDuration', 'matchType', 'maxPlace',
       'numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills',
       'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',
       'weaponsAcquired', 'winPoints'],
      dtype='object')
'''
def load_data(df, target, drop_cols):
    train_y = df[target]
    train_X = df.drop([*drop_cols,target], axis=1)

    return train_test_split(train_X, train_y, test_size=0.1, random_state=42)

def scale_features(df, cols, scalar):
    df[cols] = scalar.fit_transform(df[feature_cols])
    return df

feature_cols = [ 'assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'matchDuration', 'maxPlace',
       'numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills',
       'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',
       'weaponsAcquired', 'winPoints']

drop_cols = ['Id', 'groupId', 'matchId', 'matchType']


SAVE_DIR = f'models/keras/{time.time()}'
tensorboard = TrainValTensorBoard(log_dir=SAVE_DIR)

train= pd.read_csv('../data/train.csv', nrows=500000)

train_X, valid_X, train_y, valid_y = load_data(train, 'winPlacePerc', drop_cols)

tanh_scalar = MinMaxScaler(feature_range=(-1, 1))
train_X = scale_features(train_X, feature_cols, tanh_scalar)
valid_X = scale_features(valid_X, feature_cols, tanh_scalar)



inputs = Input(shape=(24,))
h = Dense(12, activation='sigmoid')(inputs)

h = BatchNormalization()(h)
h = Dropout(0.175)(h)
h = Dense(50, activation='sigmoid')(h)
h = BatchNormalization()(h)
h = Dropout(0.175)(h)
h = Dense(30, activation='sigmoid')(h)
h = BatchNormalization()(h)
h = Dropout(0.175)(h)

pred = Dense(1, activation='sigmoid')(h)

model = Model(inputs=inputs, outputs=pred)

model.compile(loss='mae', optimizer='rmsprop', metrics=['accuracy'])

hx = model.fit(
    train_X,
    train_y,
    validation_data=(valid_X, valid_y),
    epochs=10,
    callbacks=[tensorboard],
    batch_size = 128)

test = pd.read_csv('../data/test.csv')

ids = test.pop('Id')

predictions = list(np.reshape(model.predict(test_features), (len(test_data))))

submission = pd.DataFrame(np.transpose(np.array([ids, predictions])))

submission.columns = ['Id', 'winPlacePerc']

submission.to_csv('keras_submission.csv', index=False)
