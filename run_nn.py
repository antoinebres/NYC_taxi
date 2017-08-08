import numpy as np
import pandas as pd
from tensorflow_part import tf_nn as nn
from utils.config import *


def load_data():
    # Load data
    training_set = pd.read_csv(CV_TRAINING_SET_FILE, skipinitialspace=True,
                                  skiprows=1, names=COLUMNS)

    test_set = pd.read_csv(CV_TEST_SET_FILE, skipinitialspace=True,
                              skiprows=1, names=COLUMNS)
    return training_set, test_set


def write_predictions(dataset, predictions_tensor_log, name="predictions"):
    predictions = np.exp(np.array(predictions_tensor_log)) - 1
    trip_duration = np.transpose(predictions)
    predictions_file = dataset['id']
    predictions_file['trip_duration'] = trip_duration
    predictions_file.to_csv(name + ".csv", index=False)


def hyperparam_search():
    # Load data
    training_set, test_set = load_data()
    input_dim = training_set.shape[1] - 1
    # Hyperparameters
    learning_rate = 0.01
    hidden_units_set = [
        [10, 10],
        [10, 10, 10],
        [100, 50, 20],
        [1024, 512, 256],
        [10, 10, 10, 10],
    ]
    for hidden_units in hidden_units_set:
        model = nn.regressor(input_dim, hidden_units, learning_rate)
        model.train(training_set, test_set, LABEL, 127629, 23, 2)


def use_model(dataset, label, name, input_dim, hidden_units, learning_rate):
    model = nn.regressor(input_dim, hidden_units, learning_rate)
    predictions = model.predict(dataset, label)
    write_predictions(dataset, predictions, name)

if __name__ == '__main__':
    hyperparam_search()
