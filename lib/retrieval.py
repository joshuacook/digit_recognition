"""Methods for retrieving datasets from remote locations."""

import os
import sys
import tarfile

import numpy as np
from scipy import ndimage

from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve

IMAGE_SIZE = 28
LAST_PERCENT_REPORTED = None
PIXEL_DEPTH = 255.0
URL = 'http://commondatastorage.googleapis.com/books1000/notMNIST_large.tar.gz'

def download_progress_hook(count, block_size, total_size):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 1% change in download progress.
    """
    global LAST_PERCENT_REPORTED
    percent = int(count * block_size * 100 / total_size)

    if LAST_PERCENT_REPORTED != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        LAST_PERCENT_REPORTED = percent

def maybe_download(filename, expected_bytes, force=False, url=URL):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url, filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

def maybe_extract(filename, force=False):
    """Extract files into folders if not present."""
    num_classes = 10
    root = os.path.splitext(os.path.splitext(filename)[0])[0]    # remove .tar.gz
    root_dir = root.split('/')[0]
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(root_dir)
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    """Pickle data files if not present."""
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as this_file:
                    pickle.dump(dataset, this_file, pickle.HIGHEST_PROTOCOL)
            except Exception as error:
                print('Unable to save data to', set_filename, ':', error)

    return dataset_names

def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), IMAGE_SIZE, IMAGE_SIZE),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          PIXEL_DEPTH / 2) / PIXEL_DEPTH
            if image_data.shape != (IMAGE_SIZE, IMAGE_SIZE):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as error:
            print('Could not read:', image_file, ':', error, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset
