import numpy as np


def rmsle(real, predicted):
    _sum = 0.000
    length = len(predicted)
    for x in range(length):
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        _sum = _sum + (p - r)**2
    return (_sum/length)**0.5
