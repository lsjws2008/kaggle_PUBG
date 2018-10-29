import numpy as np


def normalizer(data):
    data = np.array(data)
    for i in range(data.shape[1]-1):
        data[:, i] = (data[:, i]-np.amin(data[:, i]))/ \
                     (np.amax(data[:, i])-np.amin(data[:, i]))

    return data

