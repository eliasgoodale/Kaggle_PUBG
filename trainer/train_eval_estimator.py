import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.float_format = '{:,.2f}'.format

import os

def selectedFeature(dataSet, train = True):
    trainSelecFeature = pd.DataFrame()
    trainSelecFeature['boosts'] = dataSet['boosts']
    trainSelecFeature['headshotKills'] = dataSet['headshotKills']
    trainSelecFeature['heals'] = dataSet['heals']
    trainSelecFeature['kills'] = dataSet['kills']
    trainSelecFeature['walkDistance'] = dataSet['walkDistance']
    trainSelecFeature['weaponsAcquired'] = dataSet['weaponsAcquired']
    if train:
        trainSelecFeature['winPlacePerc'] = dataSet['winPlacePerc']
    return trainSelecFeature


def featuresColumns():
    featureColumns = []
    featureColumns.append(tf.feature_column.numeric_column('boosts'))
    featureColumns.append(tf.feature_column.numeric_column('headshotKills'))
    featureColumns.append(tf.feature_column.numeric_column('heals'))
    featureColumns.append(tf.feature_column.numeric_column('kills'))
    featureColumns.append(tf.feature_column.numeric_column('walkDistance'))
    featureColumns.append(tf.feature_column.numeric_column('weaponsAcquired'))
    return featureColumns

def training(training_set, validate_set, periods, steps, name):
    my_optimizer = tf.train.ProximalAdagradOptimizer(
      learning_rate = 0.003,
      l1_regularization_strength = 0.001
    )
    estimator = tf.estimator.DNNRegressor(
        feature_columns = featuresColumns(),
        hidden_units = [1024, 128, 32],
        optimizer = my_optimizer,
        model_dir = name
    )
    
    training_target, training_features = training_set['winPlacePerc'], training_set.drop('winPlacePerc', axis = 1)
    validate_target, validate_features = validate_set['winPlacePerc'], validate_set.drop('winPlacePerc', axis = 1)
    
    train_fn = tf.estimator.inputs.pandas_input_fn(
        x = training_features,
        y = training_target,
        batch_size = 30,
        num_epochs = None,
        shuffle = True,
    )
    evalu_fn = tf.estimator.inputs.pandas_input_fn(
        x = validate_features,
        y = validate_target,
        batch_size = 1,
        num_epochs = 1,
        shuffle = True,
    )
    
    step_period = steps / periods
    print ("Period: ", end = ' ')
    for period in range(periods):
        training = estimator.evaluate(input_fn = train_fn, steps = step_period)
        training = estimator.train(input_fn = train_fn, steps = step_period)
        print (period + 1, end = ' ')
    training = estimator.evaluate(input_fn = train_fn, steps = step_period)
    print ("\nDone !")
    return estimator




trainData = pd.read_csv("../data/train_V2.csv", nrows=500000)
testData = pd.read_csv("../data/test_V2.csv")
trainData = trainData.reindex(np.random.permutation(trainData.index))

trainData['winPlacePerc'] = trainData['winPlacePerc'].fillna(trainData['winPlacePerc'].mean())

trainSelecSample = selectedFeature(trainData).sample(100000)

training_set = trainSelecSample.sample(frac = 0.8, replace = False)
validate_set = trainSelecSample.loc[set(trainSelecSample.index) - set(training_set.index)]

estimator = training(
    training_set = training_set,
    validate_set = validate_set,
    periods = 10,
    steps = 10000,
    name = "./models/DNNestimator/" + datetime.now().strftime("%H_%M")
)

'''
testSample = selectedFeature(testData, train = False)
testSample.isna().sum().loc[testSample.isna().sum() != 0]

predict_fn = tf.estimator.inputs.pandas_input_fn(
    x = testSample,
    y = None,
    batch_size = 1,
    num_epochs = 1,
    shuffle = False,
)
prediction = estimator.predict(input_fn = predict_fn)
prediction = [item['predictions'][0] for item in prediction]

output = pd.DataFrame(
    [(Id, pred) for Id, pred in zip(testData['Id'], prediction)],
    columns = ['Id', 'winPlacePerc']
)

output.to_csv("submission.csv", index = False)
'''