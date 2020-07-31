# partially based on https://github.com/hiram64/temporal-ensembling-semi-supervised

import argparse
import json
import os
from collections import Counter

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lib.globals import random_seed, rng


def parse_arguments():
    """
    Parse commandline arguments
    :return: parsed args object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', action='store_true', default=False)         # metacentrum flag
    parser.add_argument('--debug', action='store_true', default=False)    # debug flag
    parser.add_argument('--binary', action='store_true', default=False)   # load avast dataset as binary classification
    parser.add_argument('--data-path')                                    # path with the training data
    parser.add_argument('--out-path', default='output/')                   # output path
    parser.add_argument('--src-job-path')                                 # source path of the metacentrum job
    parser.add_argument('--runs', type=int, default=3)                    # number of repeated runs
    parser.add_argument('--num-train', type=int)                          # number of training samples
    parser.add_argument('--num-test', type=int)                           # number of test samples
    parser.add_argument('--num-unlabeled', type=int)                      # number of unlabeled samples
    parser.add_argument('--ratio', type=float)                            # ratio L : (L + U)
    parser.add_argument('--train-weeks', nargs='+', type=int)             # selected training weeks
    parser.add_argument('--unlabeled-weeks', nargs='+', type=int)         # selected unlabeled weeks
    parser.add_argument('--method', type=str)                             # semi-supervised method
    parser.add_argument('--options', type=json.loads)                     # additional algorithm options
    parser.add_argument('--hyper-par', type=json.loads)                   # hyper-parameters

    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    print(args)
    return args


args = parse_arguments()


def standardize(x):
    """
    Standardize the input data x
    """
    x32 = x.astype(np.float32)
    scaler = StandardScaler(copy=False)
    x32 = scaler.fit_transform(x32)
    x = x32.astype(np.float16)
    return x


def normalize(x, shift=None, scale=None):
    """
    Min-max normalize the input data x
    """
    if shift is None:
        shift = x.min(0)
    if scale is None:
        scale = x.max(0) - shift

    # np.save(args.out_path + 'min', shift)
    # np.save(args.out_path + 'ptp', scale)
    print('Min-Max normalizing {} records'.format(len(x)))
    xn = np.divide((x - shift), scale, out=np.zeros_like(x), where=scale != 0)
    xn = np.clip(xn, 0, 1)
    return xn


def normalize_together(*feat_sets):
    """
    Normalize multiple data sets using same min max params
    """
    mxs = [x.max(0) for x in feat_sets]
    mns = [x.min(0) for x in feat_sets]
    mn = np.min(mns, axis=0)
    mx = np.max(mxs, axis=0)
    normed = [normalize(s, mn, mx - mn) for s in feat_sets]
    return normed, mn, mx


def augmentation(inputs, noise_sd, mult=False):
    """
    Augment the data samples using Gaussian noise
    :param inputs: samples
    :param noise_sd: noise std
    :param mult: multiplicative or additive
    :return: augmented samples
    """
    if mult:
        noise = rng.normal(1, noise_sd, inputs.shape)
        return inputs * noise
    noise = rng.normal(0, noise_sd, inputs.shape)
    return inputs + noise


def to_onehot(label, num_class):
    """
    transform categorical labels to one-hot vectors
    """
    return np.identity(num_class)[label]


def to_numerical_labels(y_bytes, classes):
    """
    Converts label bytes to numerical values
    :param y_bytes: labels as bytes
    :param classes: list of classes bytes
    :return: numerical labels np array
    """
    return np.fromiter((classes.index(b) for b in y_bytes), dtype=np.int8)


def class_distribution(labels):
    """
    Calculates the distribution of classes in a list of labels
    :param labels: list of labels
    :return: array of class proportions
    """
    ctr = Counter(labels)
    counts = [ctr[l] for l in range(len(ctr))]
    return np.array(counts) / len(labels)


def load_moons(num_train, num_test):
    """
    Generate and normalize simple two moons dataset

    :param num_train: number of training data
    :param num_test: number of testing data
    :return: x_train, y_train, x_test, y_test
    """
    num_samples = num_train + num_test
    x, y = make_moons(n_samples=num_samples, shuffle=False, noise=0.08, random_state=random_seed)
    x = normalize(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=num_test, stratify=y, random_state=random_seed)
    return x_train, y_train, x_test, y_test


def make_moons_unlabeled(num_samples):
    """
    Generate and normalize simple two moons dataset and return without labels
    """
    x, _ = make_moons(n_samples=num_samples, shuffle=False, noise=0.08, random_state=random_seed)
    x = normalize(x)
    return x


def load_avast_weekly(file_keys, cut=None, num_test=None):
    """
    Loads avast malware data by weeks

    :param file_keys: selected weeks
    :param cut: number of the training samples
    :param num_test: number of test samples
    :return:
    """
    data_path = args.data_path
    path = data_path + 'avast-week/'
    feat_files = np.array(sorted(os.listdir(path + 'features/')))
    label_files = np.array(sorted(os.listdir(path + 'labels/')))

    weeks_feat = feat_files[file_keys]
    weeks_label = label_files[file_keys]

    x = np.concatenate([np.load(path + 'features/' + file, mmap_mode='r') for file in weeks_feat])
    y_b = np.concatenate([np.load(path + 'labels/' + file, mmap_mode='r') for file in weeks_label])

    if cut and cut < len(y_b):
        idx_l = rng.choice(len(y_b), cut, replace=False)
        x = x[idx_l]
        y_b = y_b[idx_l]

    classes = [b'a', b'c', b'i', b'm', b'p']
    y = to_numerical_labels(y_b, classes)

    if num_test is None:
        return x, y

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=num_test, stratify=y, random_state=random_seed)
    return x_train, y_train, x_test, y_test


def load_avast_weeks_pca(file_keys, cut=None, num_test=None, binary=False):
    """
    Loads PCA preprocessed avast malware data by weeks

    :param file_keys: selected weeks
    :param cut: number of the training samples
    :param num_test: number of test samples
    :param binary: convert to binary classification - clean or other
    :return:
    """
    print("Loading weeks", file_keys)

    data_path = args.data_path
    path = data_path + 'avast-week-pca/'
    feat_files = np.array(sorted(os.listdir(path + 'features/')))
    label_files = np.array(sorted(os.listdir(path + 'labels/')))

    weeks_feat = feat_files[file_keys]
    weeks_label = label_files[file_keys]

    x = np.concatenate([np.load(path + 'features/' + file, mmap_mode='r') for file in weeks_feat])
    y_b = np.concatenate([np.load(path + 'labels/' + file, mmap_mode='r') for file in weeks_label])

    # x = standardize(x)

    if cut and cut < len(y_b):
        idx_l = rng.choice(len(y_b), cut, replace=False)
        x = x[idx_l]
        y_b = y_b[idx_l]

    if binary:
        y = (y_b != b'c').astype(np.int8)
    else:
        classes = [b'a', b'c', b'i', b'm', b'p']
        y = to_numerical_labels(y_b, classes)

    if num_test is None:
        return x, y

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=num_test, stratify=y, random_state=random_seed)
    return x_train, y_train, x_test, y_test


def select_from_center(inputs, cat_idx, count, exp=8):
    """
    Specific for moon data - select only points close to center
    """
    x = inputs[cat_idx] - 0.5  # center to [0, 0]

    p = np.linalg.norm(x, axis=1)
    p = p.max() - p + 0.01
    p = np.power(p, exp)
    p /= p.sum()

    return rng.choice(cat_idx, count, replace=False, p=p)


def select_supervised_samples(inputs, labels, num_labeled_train, exp):
    """
    Choose randomly some samples to be used as labels
    """
    category = np.unique(labels)
    num_each_label = int(num_labeled_train / len(category))
    train_labeled_idx = []
    for cat in category:
        cat_idx = np.where(labels == cat)[0]
        if exp is None:
            train_labeled_idx.extend(rng.choice(cat_idx, num_each_label, replace=False))
        else:
            train_labeled_idx.extend(select_from_center(inputs, cat_idx, num_each_label, exp))

    return train_labeled_idx


def split_supervised_train(inputs, labels, num_labeled_train, exp=None):
    """
    Splits the data to labeled and unlabeled part
    """
    # list of unique category in labels
    assert labels.ndim == 1, "labels should be 1-dim array."

    train_labeled_idx = select_supervised_samples(inputs, labels, num_labeled_train, exp)

    # difference set between all-data indices and selected labeled data indices
    diff_set = list(np.setdiff1d(np.arange(labels.shape[0]), np.array(train_labeled_idx)))

    # labeled_x, labeled_y, unlabeled_x
    return inputs[train_labeled_idx], labels[train_labeled_idx], inputs[diff_set]


def make_train_test_dataset(inp_dic, num_class):
    """
    make train dataset and test dataset
    """
    train_x = np.concatenate((inp_dic['labeled_x'], inp_dic['unlabeled_x']), axis=0)

    # transform categorical labels to one-hot vectors
    supervised_label = to_onehot(inp_dic['labeled_y'], num_class)
    test_y = to_onehot(inp_dic['test_y'], num_class)
    num_train_unlabeled = inp_dic['unlabeled_x'].shape[0]

    # fill dummy 0 array and the size will corresponds to train dataset at axis 0
    supervised_label = np.concatenate((supervised_label, np.zeros((num_train_unlabeled, num_class))), axis=0)
    num_train_data = supervised_label.shape[0]

    # flag to indicate that supervised(1) or not(0) in train data
    supervised_flag = np.array([1] * (num_train_data - num_train_unlabeled) +
                               [0] * num_train_unlabeled)[:, np.newaxis]

    # initialize ensemble prediction label for unsupervised component. It corresponds to matrix Z
    unsupervised_target = np.zeros((num_train_data, num_class))

    # initialize weight of unsupervised loss component
    unsupervised_weight = np.zeros((num_train_data, 1))

    del inp_dic['labeled_x'], inp_dic['labeled_y'], inp_dic['unlabeled_x']
    inp_dic['train_x'] = train_x
    inp_dic['supervised_label'] = supervised_label
    inp_dic['unsupervised_target'] = unsupervised_target
    inp_dic['train_sup_flag'] = supervised_flag
    inp_dic['unsupervised_weight'] = unsupervised_weight
    inp_dic['test_y'] = test_y

    return inp_dic
