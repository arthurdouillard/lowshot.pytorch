import os
import glob
import warnings

import numpy as np
from skimage import io, transform
import torch


def parse_omniglot(omniglot_folder):
    def parse(folder, c):
        x, y = [], []
        mapping = {}

        for alphabet in os.listdir(folder):
            if alphabet[0] == '.': continue # Hidden file.

            for character in os.listdir(os.path.join(folder, alphabet)):
                if character[0] == '.': continue # Hidden file.

                # A class is a unique character from an alphabet.
                mapping[c] = f'{alphabet}_{character}'

                for img in glob.glob(os.path.join(folder, alphabet, character, '*.png')):
                    y.append(c)
                    x.append(img)

                c += 1

        return (np.array(x), np.array(y)), mapping, c

    train, mapping, c = parse(os.path.join(omniglot_folder, 'images_background'), 0)
    test, test_mapping, _ = parse(os.path.join(omniglot_folder, 'images_evaluation'), c)
    mapping.update(test_mapping)

    return train, test, mapping


def extract_validation(x_train, y_train, percent=0.1, shuffle=True):
    classes = np.unique(y_train)
    if shuffle:
        np.random.shuffle(classes)
    first_n = int(len(classes) * percent)

    val_classes = classes[:first_n]
    val_idxes = np.where(np.isin(y_train, val_classes))[0]
    train_classes = classes[first_n:]
    train_idxes = np.where(np.isin(y_train, train_classes))[0]

    return (x_train[train_idxes], y_train[train_idxes]),\
           (x_train[val_idxes], y_train[val_idxes])


def set_random_seed(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_num_features(fp_size, kernel_size, padding, stride):
    """Computes the number of features created by a 2d convolution.

    See: http://cs231n.github.io/convolutional-networks/
    """
    return (fp_size - kernel_size + 2 * padding) / stride + 1


def load_image(path, target_size):
    img = io.imread(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = transform.resize(img, target_size)

    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)

    return img


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
