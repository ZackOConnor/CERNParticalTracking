import numpy as np
import glob


def read_files(path):
    """reads in and separates all the files with in the folder into a list"""
    return [p.split('-')[0] for p in sorted(glob.glob(path))]


def load_data():
    """Sets the train, test, and sub files from the files list created from read_files"""
    train = np.unique(read_files('./train_data/**.csv'))
    test = np.unique(read_files('./test_data/**.csv'))
    sub = np.unique(read_files('./sample_submission/**.csv'))
    return train, test, sub
