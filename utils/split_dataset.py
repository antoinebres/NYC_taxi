import pandas as pd
from utils.config import *


def main(fileLocation):
    dataset = pd.read_csv(fileLocation)
    size = dataset.shape[0]
    training_size = int(0.7*size)
    shuffled_dataset = dataset.sample(frac=1)
    training_set = shuffled_dataset.iloc[:training_size]
    test_set = shuffled_dataset.iloc[training_size:]
    training_set.to_csv(CV_TRAINING_SET_FILE, index=False)
    test_set.to_csv(CV_TEST_SET_FILE, index=False)
