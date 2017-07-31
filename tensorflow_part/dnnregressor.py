# Imports
import pandas as pd
import tensorflow as tf
import numpy as np
from utils import submission, metrics
from utils.config import *

meantrain = []
stdtrain = []


def str_model_dir(hidden_units, learning_rate):
    subdir = "hidden_units " + str(hidden_units)
    subdir += " learning_rate " + str(learning_rate)
    model_dir = "/".join([MODELS_DIR, subdir])
    return model_dir


def normalize_predict(dataset, meantrain, stdtrain):
    k = 0
    for f in FEATURES:
        if f not in CATEGORICALS:
            dataset[f] = (dataset[f] - meantrain[k])/stdtrain[k]
        k += 1
    return dataset


def moment(dataset):
    for f in FEATURES:
        meantrain.append(dataset[f].mean())
        stdtrain.append(dataset[f].std())
    return meantrain, stdtrain


def input_fn(dataset, training=True):
    # Normalize data
    if training:
        for f in FEATURES:
            if f not in CATEGORICALS:
                dataset[f] = (dataset[f] - dataset[f].mean())/dataset[f].std()
    else:
        dataset = normalize_predict(dataset, meantrain, stdtrain)
    features_col = {
        k: tf.constant(dataset[k].values,
                       shape=[dataset[k].size, 1]) for k in FEATURES
    }
    labels = tf.constant((dataset[LABEL]).values.astype(np.float32), shape=[dataset[LABEL].size, 1])

    # if needed .astype(np.float32)
    return features_col, labels


def train_model(training_data, model, steps, validation_monitor):
    """
    Return the trained model.

    Parameters
    ----------
    training_data : pd.DataFrame
        DataFrame containing the data to train the model on.
    model : tensorflow model
        The model to train.
    steps : int
        The number of steps of the training.

    Returns
    -------
    tensorflow model
        The trained model.

    """
    model.fit(input_fn=lambda: input_fn(training_data), steps=steps, monitors=[validation_monitor])
    return model


def test_model(testing_data, model):
    """
    Return the metrics of the test.

    Parameters
    ----------
    testing_data : pd.DataFrame
        DataFrame containing the data to test the model on.
    model : tensorflow model
        The model to test.

    Returns
    -------
    dict
        The metrics of the test.

    """
    evaluation = model.evaluate(input_fn=lambda: input_fn(testing_data), steps=1)
    return evaluation


def predict_model(prediction_data, model):
    """
    Return the model's prediction for the new data.

    Parameters
    ----------
    prediction_data : pd.DataFrame
        DataFrame containing the data for the model to predict.
    model : tensorflow model
        The model used for the prediction.

    Returns
    -------
    array
        Numpy array of predicted scores.

    """
    predictions = model.predict(input_fn=lambda: input_fn(prediction_data))
    return predictions


def main(hidden_units, learning_rate):
    model_dir = str_model_dir(hidden_units, learning_rate)
    tf.logging.set_verbosity(tf.logging.INFO)

    cv_training_set = pd.read_csv(CV_TRAINING_SET_FILE, skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

    cv_test_set = pd.read_csv(CV_TEST_SET_FILE, skipinitialspace=True,
                           skiprows=1, names=COLUMNS)

    meantrain, stdtrain = moment(cv_training_set)

    # Feature cols
    feature_columns = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
    validation_metrics = {
        "rmse":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_root_mean_squared_error)}
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=lambda: input_fn(cv_training_set),
        every_n_steps=50,
        metrics=validation_metrics,
        early_stopping_metric="rmse",
        early_stopping_metric_minimize=True,
        early_stopping_rounds=50)
    # Build the model
    estimator = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=learning_rate,
            l1_regularization_strength=0.001),
        model_dir=model_dir,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=20)
        )
    # Training
    estimator = train_model(cv_training_set, estimator, 50, validation_monitor)

    # Prediction
    predictions = predict_model(cv_test_set, estimator)
    pred = np.array(predictions)
    real = np.array(cv_test_set['trip_duration'])
    print("Root Mean Square Logarithmic Error:")
    print(metrics.rmsle(real, pred))


def make_submission(hidden_units, learning_rate):
    model_dir = str_model_dir(hidden_units, learning_rate)
    training_set = pd.read_csv(TRAINING_SET_FILE, skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    # Feature cols
    feature_columns = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
    estimator = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=learning_rate,
            l1_regularization_strength=0.001),
        model_dir=model_dir
        )
    estimator = train_model(training_set, estimator, 500)

    test_set = pd.read_csv(TEST_SET_FILE, skipinitialspace=True,
                           skiprows=1, names=COLUMNS)

    prediction_set = pd.read_csv(PREDICTION_SET_FILE, skipinitialspace=True,
                                 skiprows=1, names=['id', 'trip_duration'])

    predictions = predict_model(test_set, estimator)
    prediction_set['trip_duration'] = predictions
    submission.write(prediction_set)
