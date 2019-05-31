import numpy as np
import glob


def read_files(path):
    return [p.split('-')[0] for p in sorted(glob.glob(path))]


def load_data():
    #Grabs the train, test, and submission folders
    train = np.unique(read_files('./train_data/**.csv'))
    test = np.unique(read_files('./test_data/**.csv'))
    sub = np.unique(read_files('./sample_submission/**.csv'))
    return train, test, sub