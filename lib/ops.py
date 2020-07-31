# partially based on https://github.com/hiram64/temporal-ensembling-semi-supervised

import sys

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.losses import mean_squared_error

from lib.utils import to_onehot


def create_loss_func(num_class, class_distr, ssl_method=None, to_return=None, enable_harden=False):
    """
    builds a custom tf loss function
    :param num_class: number of classes
    :param class_distr: class distribution for imbalanced datasets
    :param ssl_method: semi-supervised method
    :param to_return: which loss variant to return
    :param enable_harden: Experimantal
    :return:
    """
    epsilon = 1e-08
    pseudo_label_threshold = 0.55

    if class_distr is not None:
        inv_distr = 1 / class_distr
        class_weights = len(class_distr) * inv_distr / sum(inv_distr)

    def harden(x):
        return (tf.math.erf(8 * (x - 0.5)) + 1) / 2

    def weight_f(y_true, y_pred):
        weight = y_true[:, -1]
        weight = K.mean(weight)
        return weight

    def cross_entropy(prediction, one_hot_target, selection):
        c_entropy = one_hot_target * K.log(K.clip(prediction, epsilon, 1.0 - epsilon))
        if class_distr is not None:
            c_entropy *= class_weights
        # To sum over only supervised data on categorical_crossentropy, supervised_flag(1/0) is used
        supervised_loss = - K.mean(
            K.sum(c_entropy, axis=1) * selection
        )
        return supervised_loss

    def pi_model_f(y_true, y_pred):
        unsupervised_target = y_true[:, 0:num_class]
        model_pred = y_pred[:, 0:num_class]

        if enable_harden:
            unsupervised_target = harden(unsupervised_target)
            model_pred = harden(model_pred)

        return K.mean(mean_squared_error(unsupervised_target, model_pred))

    def pi_model_labeled_f(y_true, y_pred):
        unsupervised_target = y_true[:, 0:num_class]
        supervised_flag = y_true[:, num_class * 2]
        model_pred = y_pred[:, 0:num_class]

        return K.mean(mean_squared_error(unsupervised_target, model_pred) * supervised_flag)

    def pi_model_unlabeled_f(y_true, y_pred):
        return pi_model_f(y_true, y_pred) - pi_model_labeled_f(y_true, y_pred)

    def pseudo_label_f(y_true, y_pred):
        unsupervised_flag = 1 - y_true[:, num_class * 2]
        model_pred = y_pred[:, 0:num_class]
        batch_size = K.shape(model_pred)[0]

        max_confidence = K.max(model_pred, axis=1)
        max_confidence = K.reshape(max_confidence, (1, batch_size))

        pseudo_target = model_pred - K.transpose(max_confidence)
        pseudo_target = K.equal(pseudo_target, 0)
        pseudo_target = K.cast(pseudo_target, 'float32')

        cutoff = K.greater(max_confidence, pseudo_label_threshold)
        cutoff = K.cast(cutoff, 'float32')

        selection = unsupervised_flag * cutoff

        unsupervised_loss = cross_entropy(model_pred, pseudo_target, selection)

        return unsupervised_loss

    def unsupervised_f(y_true, y_pred):
        if ssl_method == 'pi-model':
            unsupervised_loss = pi_model_f(y_true, y_pred)
        elif ssl_method == 'pseudo-label':
            unsupervised_loss = pseudo_label_f(y_true, y_pred)
        else:
            unsupervised_loss = 0

        return unsupervised_loss

    def supervised_f(y_true, y_pred):
        supervised_label = y_true[:, num_class:num_class * 2]
        supervised_flag = y_true[:, num_class * 2]
        model_pred = y_pred[:, 0:num_class]

        supervised_loss = cross_entropy(model_pred, supervised_label, supervised_flag)
        return supervised_loss

    def loss_func(y_true, y_pred):
        """
        semi-supervised loss function
        the order of y_true:
        unsupervised_target(num_class), supervised_label(num_class), supervised_flag(1), unsupervised weight(1)
        """

        weight = y_true[:, -1]

        supervised_loss = supervised_f(y_true, y_pred)
        unsupervised_loss = unsupervised_f(y_true, y_pred)

        return supervised_loss + weight * unsupervised_loss

    if to_return == 'weight':
        return weight_f
    elif to_return == 'supervised':
        return supervised_f
    elif to_return == 'unsupervised':
        return unsupervised_f
    elif to_return == 'pi_model_labeled':
        return pi_model_labeled_f
    elif to_return == 'pi_model_unlabeled':
        return pi_model_unlabeled_f

    return loss_func


def wrap_print(x, message):
    """
    Help function for debug
    """
    # op = tf.print(message, x, output_stream='file://loss.log', summarize=-1)
    op = tf.print(message, x, output_stream=sys.stdout, summarize=-1)
    with tf.control_dependencies([op]):
        return 0 * tf.identity(tf.reduce_mean(x))


def ramp_up_weight(ramp_period, weight_max):
    """
    ramp-up weight generator.
    used in unsupervised component of loss.

    :param ramp_period: length of the ramp up period
    :param weight_max: maximal weight
    """

    cur_epoch = 0
    while True:
        if cur_epoch <= ramp_period - 1:
            T = (1 / (ramp_period - 1)) * cur_epoch
            yield np.exp(-5 * (1 - T) ** 2) * weight_max
        else:
            yield 1 * weight_max
        cur_epoch += 1


def ramp_down_weight(ramp_period):
    """
    ramp-down weight generator
    :param ramp_period: length of the ramp-down period
    """

    cur_epoch = 1
    while True:
        if cur_epoch <= ramp_period - 1:
            T = (1 / (ramp_period - 1)) * cur_epoch
            yield np.exp(-12.5 * T ** 2)
        else:
            yield 0
        cur_epoch += 1


def update_weight(y, unsupervised_weight, next_weight):
    """
    update weight of the unsupervised part of loss
    """
    y[:, -1] = next_weight
    unsupervised_weight[:] = next_weight

    return y, unsupervised_weight


def evaluate(model, num_class, test_x, test_y, hot=True):
    """
    Evaluate the models prediction on a test set
    :param model: Model
    :param num_class: number of classes
    :param test_x: test samples
    :param test_y: test targets
    :param hot: whether the target is one hot encoded
    :return:
    """
    assert len(test_x) == len(test_y)

    if not hot:
        test_y = to_onehot(test_y, num_class)

    num_test = len(test_y)

    test_supervised_label_dummy = np.zeros((num_test, num_class))
    test_supervised_flag_dummy = np.zeros((num_test, 1))
    test_unsupervised_weight_dummy = np.zeros((num_test, 1))

    test_x_ap = [test_x, test_supervised_label_dummy, test_supervised_flag_dummy, test_unsupervised_weight_dummy]
    p = model.predict(x=test_x_ap)
    pr = p[:, 0:num_class]
    pr_arg_max = np.argmax(pr, axis=1)
    tr_arg_max = np.argmax(test_y, axis=1)
    cnt = np.sum(pr_arg_max == tr_arg_max)
    acc = cnt / num_test

    return acc
