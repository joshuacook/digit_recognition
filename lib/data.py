"""Methods for saving and loading data dictionaries."""

import os
from six.moves import cPickle as pickle

def load_dataset(pickle_file):
    """Used to load a dataset from a pickled file."""

    print("Loading pickled data...")

    try:
        with open(pickle_file, 'rb') as this_file:
            data_dictionary = pickle.load(this_file)
    except Exception as error:
        print('Unable to process data from', pickle_file, ':', error)
        raise

    return data_dictionary

def pickle_data_dictionary(data_dictionary, pickle_file):
    """Used to pickle a data dictionary."""

    try:
        this_file = open(pickle_file, 'wb')
        save = data_dictionary
        pickle.dump(save, this_file, pickle.HIGHEST_PROTOCOL)
        this_file.close()
    except Exception as error:
        print('Unable to save data to', pickle_file, ':', error)
        raise

    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)
