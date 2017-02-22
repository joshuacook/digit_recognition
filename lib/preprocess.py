import numpy as np
import pandas as pd
import os
import sys
import random
import tarfile
import h5py

from scipy import ndimage
from scipy import misc
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from keras.utils import np_utils
import random
from PIL import Image

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

def download_progress_hook(count, blockSize, totalSize):
    global last_percent_reported
    last_percent_reported = None
    percent = int(count * blockSize * 100 / totalSize)
    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
          
        last_percent_reported = percent
        
def maybe_download(filename, url, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename) 
        filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(filename)
    return filename

def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    print(data_folders)
    return data_folders

def get_bounding_box(image_dict):
    top = min(image_dict['top'])
    left = min(image_dict['left'])
    
    top_and_height = [image_dict['top'], image_dict['height']]
    left_and_width = [image_dict['left'], image_dict['width']]
    
    bottom =max(np.sum(top_and_height, axis=0))
    right = max(np.sum(left_and_width,axis=0))
    
    return {'top': top, 'bottom': bottom, 'left':left, 'right':right}

def crop_and_resize(image_path, dimensions_dict):
    original = Image.open(image_path)
    left = int(dimensions_dict['left'])
    top = int(dimensions_dict['top'])
    right = int(dimensions_dict['right'])
    bottom =int(dimensions_dict['bottom'])
    box = (left,top,right,bottom)
    cropped = original.crop(box)
    resized = cropped.resize((54, 54))
    return resized

def process_all_images(source_folder, destination_folder, images_info):
    images_list = os.listdir(source_folder)
    images_list.remove('digitStruct.mat')
    images_list.remove('see_bboxes.m')
    for image_name in images_list:
        image_num = int(image_name[:-4]) - 1
        processed = crop_and_resize(source_folder+image_name, get_bounding_box(images_info[image_num]))
        processed.save(destination_folder + image_name) 

import os

from scipy import misc

def prepare_svhn_dataset(path, info):
    images_list = os.listdir(path)
    png = []
    labels = []
    
    for image in images_list:
        im = misc.imread(path+image)
        png.append(im)   
        
        image_num = int(image[:-4]) - 1
        try:
            image_label = info[image_num]['labels']
            image_length = info[image_num]['length']
        except IndexError:
            continue
        if image_length >5:
            image_label = image_label[:5]
        for i in range(image_length, 5):
            image_label = np.append(image_label,0)
        image_label = np.append(image_label,image_length)
        labels.append(image_label)
        
    images = np.asarray(png)
    labels =  np.asarray(labels)
    
    y_1 = labels[:,0]
    y_2 = labels[:,1]
    y_3 = labels[:,2]
    y_4 = labels[:,3]
    y_5 = labels[:,4]
    y_6 = labels[:,5]

    one_hot_labels = [np_utils.to_categorical(y_1),
                      np_utils.to_categorical(y_2),
                      np_utils.to_categorical(y_3),
                      np_utils.to_categorical(y_4),
                      np_utils.to_categorical(y_5),
                      np_utils.to_categorical(y_6)]
    
    return images, one_hot_labels
            
def download_and_extract_svhn_datasets():
    url = 'http://ufldl.stanford.edu/housenumbers/'
    train_filename = maybe_download('train.tar.gz', url=url)
    test_filename = maybe_download('test.tar.gz', url=url)

    train_folder = maybe_extract(train_filename)
    test_folder = maybe_extract(test_filename)

    
def svhn_meta_from_mat(filename):
    """ Reads and processes the mat files provided in the SVHN dataset. 
        Input: filename 
        Ouptut: list of python dictionaries 
    """
         
    f = h5py.File(filename, 'r')
    groups = f['digitStruct'].items()
    bbox_ds = np.array(groups[0][1]).squeeze()
    names_ds = np.array(groups[1][1]).squeeze()
 
    data_list = []
    num_files = bbox_ds.shape[0]
    count = 0
 
    for objref1, objref2 in zip(bbox_ds, names_ds):
 
        data_dict = {}
 
        # Extract image name
        names_ds = np.array(f[objref2]).squeeze()
        filename = ''.join(chr(x) for x in names_ds)
        data_dict['filename'] = filename
 
        #print filename
 
        # Extract other properties
        items1 = f[objref1].items()
 
        # Extract image label
        labels_ds = np.array(items1[1][1]).squeeze()
        try:
            label_vals = [int(f[ref][:][0, 0]) for ref in labels_ds]
        except TypeError:
            label_vals = [labels_ds]
        data_dict['labels'] = label_vals
        data_dict['length'] = len(label_vals)
 
        # Extract image height
        height_ds = np.array(items1[0][1]).squeeze()
        try:
            height_vals = [f[ref][:][0, 0] for ref in height_ds]
        except TypeError:
            height_vals = [height_ds]
        data_dict['height'] = height_vals
 
        # Extract image left coords
        left_ds = np.array(items1[2][1]).squeeze()
        try:
            left_vals = [f[ref][:][0, 0] for ref in left_ds]
        except TypeError:
            left_vals = [left_ds]
        data_dict['left'] = left_vals
 
        # Extract image top coords
        top_ds = np.array(items1[3][1]).squeeze()
        try:
            top_vals = [f[ref][:][0, 0] for ref in top_ds]
        except TypeError:
            top_vals = [top_ds]
        data_dict['top'] = top_vals
 
        # Extract image width
        width_ds = np.array(items1[4][1]).squeeze()
        try:
            width_vals = [f[ref][:][0, 0] for ref in width_ds]
        except TypeError:
            width_vals = [width_ds]
        data_dict['width'] = width_vals
 
        data_list.append(data_dict)
 
        count += 1
 
    return data_list
