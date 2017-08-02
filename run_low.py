import numpy as np
import pandas as pd
from tensorflow_part import tf_low as low
from utils.config import *
from utils.preprocessing_data import log_trip_duration


def train(model):
    low.train_model(*model) 


def predict(dataset, model):
    dataset = log_trip_duration(dataset)
    trip_duration_log_tensor = low.predict_from_model(dataset, *model)
    trip_duration = np.exp(np.array(trip_duration_log_tensor)) - 1
    return np.transpose(trip_duration)


def write_pred(filename):
    dataset = pd.read_csv(filename, skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
    pred = predict(dataset, model)
    prediction_file = dataset['id']
    prediction_file['trip_duration'] = pred
    prediction_file.to_csv(filename[:-4] + 'predNN.csv')


def main():
    # Training
    hidden_units_set = [
        [10, 10],
        [10, 10, 10],
        [100, 50, 20],
        [1024, 512, 256],
        [10, 10, 10, 10],
    ]
    for hidden_units in hidden_units_set:
        model = (hidden_units, 0.102)
        train(model)

    # Prediction
    # write_pred(CV_SET_FUSION)
    # write_pred(TEST_SET_FUSION)
    # write_pred(TEST_SET_FILE_FUSION)

if __name__ == '__main__':
    main()
