import numpy as np
import os
import sys
import random

from scipy import misc
from six.moves import cPickle as pickle
from keras.utils import np_utils

TRAIN_SET_SIZE = 95000
VALID_SET_SIZE = 5000
TEST_SET_SIZE = 5000
MAX_SEQ_LEN = 5

np.random.seed(42)

def instantiate_datasets():
    train_set = np.ndarray(shape=(TRAIN_SET_SIZE, 64, 64),
                              dtype=np.float32)

    valid_set = np.ndarray(shape=(VALID_SET_SIZE, 64, 64),
                                  dtype=np.float32)

    test_set = np.ndarray(shape=(TEST_SET_SIZE, 64, 64),
                                 dtype=np.float32)

    return train_set, valid_set, test_set
            
def make_synthetic_datasets(train_set, valid_set, test_set, notmnist_datasets):

    dataset = np.ndarray(shape=(0, 28, 28))

    labels = np.ndarray(shape=0)

    print "loading data: ",
    for i in xrange(len(notmnist_datasets)):
        print ".",
        letters = open(notmnist_datasets[i])
        letters = pickle.load(letters)
        letters = _normalize(letters)
        new_labels = np.ndarray(shape=(letters.shape[0],1), dtype=int)
        new_labels.fill(i)
        labels = np.append(labels,new_labels)
        dataset = np.concatenate((dataset, letters),axis=0)

    print "\nshuffle data"
    shuffled_dataset, shuffled_labels = _randomize(dataset, labels)

    print "make dataset: ",
    print ".", 
    train_set, train_labels = _make_dataset(train_set, shuffled_dataset[:475000], TRAIN_SET_SIZE, shuffled_labels)
    print ".",
    valid_set, valid_labels = _make_dataset(valid_set, shuffled_dataset[457000:500000], VALID_SET_SIZE, shuffled_labels)
    print "."
    test_set, test_labels = _make_dataset(test_set, shuffled_dataset[500000:525000], TEST_SET_SIZE, shuffled_labels)

    print "reshape data for cnn: ",
    print ".",
    train_set = _reshape_for_cnn(train_set)
    print ".",
    valid_set = _reshape_for_cnn(valid_set)
    print "."
    test_set = _reshape_for_cnn(test_set)


    print "one-hot encode labels: ",
    
    print ".",
    
    y_trn_1 = train_labels[:,0]
    y_trn_2 = train_labels[:,1]
    y_trn_3 = train_labels[:,2]
    y_trn_4 = train_labels[:,3]
    y_trn_5 = train_labels[:,4]
    
    one_hot_train_labels = [np_utils.to_categorical(y_trn_1),
                            np_utils.to_categorical(y_trn_2),
                            np_utils.to_categorical(y_trn_3),
                            np_utils.to_categorical(y_trn_4),
                            np_utils.to_categorical(y_trn_5)]
    print ".",

    y_tst_1 = test_labels[:,0]
    y_tst_2 = test_labels[:,1]
    y_tst_3 = test_labels[:,2]
    y_tst_4 = test_labels[:,3]
    y_tst_5 = test_labels[:,4]
    
    one_hot_test_labels = [np_utils.to_categorical(y_tst_1),
                           np_utils.to_categorical(y_tst_2),
                           np_utils.to_categorical(y_tst_3),
                           np_utils.to_categorical(y_tst_4),
                           np_utils.to_categorical(y_tst_5)]
    print "."

    y_val_1 = valid_labels[:,0]
    y_val_2 = valid_labels[:,1]
    y_val_3 = valid_labels[:,2]
    y_val_4 = valid_labels[:,3]
    y_val_5 = valid_labels[:,4]
    
    one_hot_valid_labels = [np_utils.to_categorical(y_val_1),
                            np_utils.to_categorical(y_val_2),
                            np_utils.to_categorical(y_val_3),
                            np_utils.to_categorical(y_val_4),
                            np_utils.to_categorical(y_val_5)]

    data_dictionary = {
                       'train_set': train_set,
                       'valid_set': valid_set,
                       'test_set': test_set,
                       'train_labels': train_labels,
                       'valid_labels': valid_labels,
                       'test_labels': test_labels,
                       'one_hot_train_labels': one_hot_train_labels,
                       'one_hot_valid_labels': one_hot_valid_labels,
                       'one_hot_test_labels': one_hot_test_labels,
                       'single_digit_set':shuffled_dataset[:20000],
                       'single_digit_labels':shuffled_labels[:20000]
                      }

    return data_dictionary

def pickle_data_dictionary(data_dictionary):

    pickle_file = 'notMNIST_concat.pickle'

    try:
        f = open(pickle_file, 'wb')
        save = data_dictionary
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)

def _normalize(dataset):
    return dataset/255.  

def _randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def _reshape_for_cnn(dataset):
    return dataset.reshape(dataset.shape[0], dataset.shape[1], dataset.shape[2], 1)

def _make_dataset(return_set, dataset, data_set_size, input_labels):
        sequence_lengths = ([random.randint(1, MAX_SEQ_LEN) for i in range(0,return_set.shape[0])])
        data_set_size = return_set.shape[0]
        start=0
        end=0
        labels=np.ndarray(shape=(data_set_size, MAX_SEQ_LEN))
        labels.fill(10)
        for i in range(0,data_set_size):
                end += sequence_lengths[i]
                concatenation = np.concatenate((dataset[start:end]), axis=1)
                concatenation = misc.imresize(concatenation,0.41)
                leftPad = int((64 - concatenation.shape[0]) / 2)
                rightPad = int((64 - concatenation.shape[0]) - leftPad)
                topPad = int((64 - concatenation.shape[1]) / 2)
                bottomPad = int(64 - concatenation.shape[1] - topPad)
                pads = ((leftPad,rightPad),(topPad,bottomPad))
                concatenation=np.pad(concatenation, pads, 'constant',constant_values=0)
                concatenation=misc.imresize(concatenation,(64,64))
                
                these_input_labels = input_labels[start:end]
                while len(these_input_labels) < 5:
                    these_input_labels = np.append(these_input_labels, 10.)
                labels[i] = these_input_labels
                return_set[i]=concatenation
                start=end
        return return_set, labels
