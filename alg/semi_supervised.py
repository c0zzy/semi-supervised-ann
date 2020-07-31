# partially based on https://github.com/hiram64/temporal-ensembling-semi-supervised

import os
import time

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from lib.globals import random_seed
from lib.model_mlp import build_model
from lib.ops import ramp_up_weight, create_loss_func, evaluate, ramp_down_weight, update_weight
from lib.utils import augmentation, class_distribution, to_onehot, parse_arguments

args = parse_arguments()

# Limit number of cpus used in Metacentrum environment
if args.m:
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1,
                            allow_soft_placement=True,
                            device_count={'CPU': 1})
    session = tf.Session(config=config)
    K.set_session(session)


class SemiSupervised:
    """
    Central class for the (semi)-supervised training
    """

    def __init__(
            self,
            dataset_params,
            # Training parameters
            num_epoch=100,
            batch_size=100,
            learning_rate=0.001,
            # Algorithm
            ssl_method=None,
            options=None,
            hyper_par=None,
            # Other parameters
            imbalanced=False,
            save_results=False,
            should_evaluate=True,
    ):
        """
        Save params to class properties and initialize it
        """
        # Dataset parameters
        self.ds_name = dataset_params['name']
        self.input_size = dataset_params['input_size']
        self.num_classes = dataset_params['num_classes']
        self.mlp_arch = dataset_params['mlp_arch']
        self.vis_funcs_names = dataset_params['visualise_funcs']

        # Training parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch

        # Algorithm
        self.ssl_method = ssl_method
        if options is None:
            options = {}
        # increasing consistency
        self.augment = options.get('augment', False)
        self.dropout = options.get('dropout', False)
        self.temporal = options.get('temporal', False)

        # Hyperparameters with their default values
        if hyper_par is None:
            hyper_par = {}
        #    pi-model
        self.noise = hyper_par.get('noise', 0.05)
        self.weight_max = hyper_par.get('weight_max', 10)
        self.ramp_up = hyper_par.get('ramp_up', 70)
        self.ramp_down = hyper_par.get('ramp_down', 20)
        #    pseudo-label
        self.threshold = hyper_par.get('threshold', 0.9)

        # Other parameters
        self.imbalanced = imbalanced
        self.save_results = save_results
        self.should_evaluate = should_evaluate

        # Data
        self.train_x, self.train_y, self.test_x, self.test_y = 4 * [None]
        self.labeled_x, self.labeled_y, self.unlabeled_x = 3 * [None]
        self.supervised_label, self.supervised_flag, self.unsupervised_target, self.unsupervised_weight = 4 * [None]
        self.distances = None

        # Metrics
        self.metrics_names = ['loss']
        if args.debug:
            if self.ssl_method:
                self.metrics_names = ['loss', 'supervised', 'unsupervised', 'weight']
            if self.ssl_method == 'pi-model':
                self.metrics_names += ['pi_model_labeled', 'pi_model_unlabeled']
        self.metrics_sums = [0] * len(self.metrics_names)

        # Other
        self.num_labeled_train = None
        self.model_predict = None
        self.visualizer = None
        self.result_path = args.out_path

    def set_train_data(self, x_train, y_train):
        self.train_x, self.train_y, = x_train, y_train

    def set_test_data(self, x_test, y_test):
        self.test_x, self.test_y = x_test, y_test

    def set_unsupervised_data(self, unlabeled_x=None):
        self.labeled_x = self.train_x
        self.labeled_y = self.train_y
        self.unlabeled_x = unlabeled_x
        if unlabeled_x is None:
            self.unlabeled_x = np.empty((0, self.labeled_x.shape[1]))

    def split_sup_unsup(self, ratio=0.5, select_func=None):
        """
        split the trining date to labeled and unlabeled in certain ratio = L : (L + U)
        """
        x, y = self.train_x, self.train_y
        assert y.ndim == 1, "labels should be 1-dim array."

        labeled_idx = []
        if select_func:
            category = np.unique(y)
            num_each_label = int(ratio * len(category))
            for cat in category:
                cat_idx = np.where(y == cat)[0]
                labeled_idx.extend(select_func(x, cat_idx, num_each_label))
            # Subtract labeled ids to get unlabeled
            diff_set = list(np.setdiff1d(np.arange(y.shape[0]), np.array(labeled_idx)))
            self.labeled_x, self.labeled_y = x[labeled_idx], y[labeled_idx]
            self.unlabeled_x = x[diff_set]
        else:
            if ratio >= 1.0:
                # Make it whole labeled
                self.labeled_x, self.labeled_y = x, y
                self.unlabeled_x = np.empty((0, self.labeled_x.shape[1]))
            else:
                self.unlabeled_x, self.labeled_x, _, self.labeled_y = train_test_split(
                    x, y, test_size=ratio, stratify=y, random_state=random_seed)

    def prepare_train_test_data(self):
        """
        Prepare data to form used in loss calculation
        """
        self.train_x = np.concatenate((self.labeled_x, self.unlabeled_x), axis=0)

        # Transform categorical labels to one-hot vectors
        self.supervised_label = to_onehot(self.labeled_y, self.num_classes)
        self.test_y = to_onehot(self.test_y, self.num_classes)
        num_train_unlabeled = self.unlabeled_x.shape[0]

        # Fill dummy 0 array and the size will corresponds to train dataset at axis 0
        self.supervised_label = np.concatenate(
            (self.supervised_label, np.zeros((num_train_unlabeled, self.num_classes))), axis=0)

        num_train_data = self.supervised_label.shape[0]

        # Flag to indicate that supervised(1) or not(0) in train data
        self.supervised_flag = np.array([1] * (num_train_data - num_train_unlabeled) +
                                        [0] * num_train_unlabeled)[:, np.newaxis]

        # Initialize ensemble prediction label for unsupervised component. It corresponds to matrix Z
        self.unsupervised_target = np.zeros((num_train_data, self.num_classes))

        # initialize weight of unsupervised loss component
        self.unsupervised_weight = np.zeros((num_train_data, 1))

    def batch_metrics(self, metrics):
        """
        Save metrics after precessing each mini-batch
        """
        try:
            for i in range(len(metrics)):
                self.metrics_sums[i] += metrics[i]
        except TypeError:
            self.metrics_sums[0] += metrics

    def log_avg_metrics(self):
        values = [s / self.batch_size for s in self.metrics_sums]
        self.metrics_sums = [0] * len(self.metrics_sums)
        return dict(zip(self.metrics_names, values))

    def calculate_distances(self):
        """
        Experimental - unused
        """
        split = [[] for i in range(self.num_classes)]
        for i, y in enumerate(self.labeled_y):
            split[y].append(i)

        dist_to_classes = []
        for s in split:
            same_class = self.labeled_x[s]
            dist = pairwise_distances(self.train_x, same_class, metric='cosine')
            dist_to_classes.append(np.percentile(dist, 10, axis=1))

        mins = np.min(dist_to_classes, axis=0)
        return mins * len(mins) / mins.sum()

    def create_model(self, build_model_func):
        """
        Create a model and compile it with optimizer and metrics
        :param build_model_func: A function that returns a keras model
        :return: compiled keras model
        """
        class_distr = class_distribution(self.train_y) if self.imbalanced else None
        loss = create_loss_func(self.num_classes, class_distr, self.ssl_method)
        model = build_model_func(self.input_size, self.num_classes, self.mlp_arch, self.dropout)

        optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999)
        metrics = [create_loss_func(self.num_classes, class_distr, self.ssl_method, part)
                   for part in self.metrics_names[1:]]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model_predict = K.function(inputs=[model.layers[0].input, K.learning_phase()],
                                        outputs=[model.get_layer('output-softmax').output])

        model.summary()
        return model

    def dist_to_weight(self, distances):
        """
        Experimental - unused
        """

        def curve(x, a):
            return a * 4 * x ** 3 + (1 - a) * x

        def smoothed(x, a, b):
            return b * curve(x, a) + (b - 1) * curve(x, a) + 0.5

        def dist2w(x, a, b, mx):
            return smoothed(x - mx / 2, a, b)

        def log_for_dist(x):
            return np.clip(-np.log(1 - x), 0, 3)

        def log_for_near(x):
            return np.clip(-np.log(x), 0, 3)

        a = 0.5
        b = 1
        mx = 1

        distances = (distances - distances.min()) / distances.ptp()
        # w = np.fromiter((dist2w(d, a, b, mx) for d in distances), dtype=np.float32)
        w = np.fromiter((log_for_dist(d) for d in distances), dtype=np.float32)
        return w * (len(w) / w.sum())

    def make_batch_data(self, y, target_idx):
        """
        prepare the batch of date to form used in loss calculation
        :param y: array of all labels
        :param target_idx: selected indices for the mini-batch
        """
        y_t = y[target_idx]

        x1 = self.train_x[target_idx]
        x2 = self.supervised_label[target_idx]
        x3 = self.supervised_flag[target_idx]
        x4 = self.unsupervised_weight[target_idx]

        # increasing the consistency
        if self.ssl_method == 'pi-model' and not self.temporal:
            if self.augment:
                x1 = augmentation(self.train_x[target_idx], self.noise, mult=True)

            learning_phase = 1
            if self.dropout:
                _ = self.model_predict(inputs=[x1[:1], learning_phase])[0]  # Different dropout hack
                # get the first prediction
            y_t[:, 0:self.num_classes] = self.model_predict(inputs=[x1, learning_phase])[0]

        if self.augment:
            x1 = augmentation(self.train_x[target_idx], self.noise, mult=True)

        x_t = [x1, x2, x3, x4]
        return x_t, y_t

    def train(self, run_nmb):
        """
        The main method with the training loop
        :param run_nmb: number of repeated run - for saving of models
        :return:
        """
        self.print_training_settings()

        # Prepare the data to the form used in loss calculation
        y_columns = (self.unsupervised_target, self.supervised_label, self.supervised_flag, self.unsupervised_weight)
        y = np.concatenate(y_columns, axis=1)

        num_train = len(self.train_x)
        w_max = self.weight_max * (len(self.labeled_x) / num_train)

        # Init Visualizer
        if len(self.vis_funcs_names) > 0:
            from lib.visualizer import Visualizer
            self.visualizer = Visualizer(self.result_path, self.vis_funcs_names, self.num_classes, self.train_x, y,
                                         w_max)

        # Create model
        model = self.create_model(build_model)
        model_weights_path = self.result_path + str(run_nmb) + '-model.h5'

        # Prepare ramp-up and ramp-down generators
        gen_weight = gen_lr_weight = None
        if self.ssl_method:  # semi
            gen_weight = ramp_up_weight(self.ramp_up, w_max)
            gen_lr_weight = ramp_down_weight(self.ramp_down)
        else:  # Supervised only
            num_train = len(self.labeled_x)

        # Training
        best_loss = 1e10
        idx_list = list(range(num_train))
        for epoch in range(self.num_epoch):
            print('epoch: ', epoch, end='\t')
            start_time = time.time()

            idx_list = shuffle(idx_list)
            if self.ssl_method and epoch > self.num_epoch - self.ramp_down:
                self.update_train_params(gen_lr_weight, model)

            # Train on batches
            for i in range(0, num_train, self.batch_size):
                target_idx = idx_list[i:i + self.batch_size]
                x_t, y_t = self.make_batch_data(y, target_idx)
                metrics = model.train_on_batch(x=x_t, y=y_t)
                self.batch_metrics(metrics)

            # Log
            metrics_avgs = self.log_avg_metrics()
            if self.ssl_method and 'losses' in self.vis_funcs_names:
                self.visualizer.add_losses(metrics_avgs)
            duration = np.round(time.time() - start_time, 2)
            print(metrics_avgs, 'time: ', duration, flush=True)

            # Elitism
            if self.ssl_method and args.debug:
                max_w_loss = metrics_avgs['supervised'] + w_max * metrics_avgs['unsupervised']
                if max_w_loss < best_loss:
                    model.save_weights(model_weights_path)

            # Update params phase
            if self.ssl_method:
                y, unsupervised_weight = update_weight(y, self.unsupervised_weight, next(gen_weight))

            # Evaluation
            if self.should_evaluate and (epoch) % 5 == 0:
                test_acc = evaluate(model, self.num_classes, self.test_x, self.test_y)
                train_acc = evaluate(model, self.num_classes, self.labeled_x, self.labeled_y,
                                     hot=False)
                print('Evaluate epoch : {} Test acc: {} Train acc {}'.format(epoch, test_acc, train_acc), flush=True)
                if self.visualizer:
                    self.visualizer.visualize(epoch, model, metrics_avgs, test_acc, show=True)

        if self.save_results and 'acc' in self.vis_funcs_names:
            self.visualizer.acc_save()

        if self.ssl_method and args.debug:
            model.load_weights(model_weights_path)
        else:
            model.save_weights(model_weights_path)
        return model

    def update_train_params(self, gen_lr_weight, model):
        weight_down = next(gen_lr_weight)
        K.set_value(model.optimizer.lr, weight_down * self.learning_rate)
        K.set_value(model.optimizer.beta_1, 0.4 * weight_down + 0.5)

    def print_training_settings(self):
        print('### TRAINING ###')
        print('ssl_method', self.ssl_method)
        print('augment', self.augment)
        print('dropout', self.dropout)
        print('temporal', self.temporal)
        print('noise', self.noise)
        print('weight_max', self.weight_max)
        print('ramp_up', self.ramp_up)
        print('ramp_down', self.ramp_down)
        print('threshold', self.threshold)

    def make_result_path(self, changed_params):
        learn_type = self.ssl_method if self.ssl_method else 'super'
        if changed_params is None:
            run_name = folder_name = 'default'
        else:
            run_name = '_'.join(sorted(changed_params))
            folder_name = '_'.join([k + str(v) for k, v in sorted(changed_params.items())])
        result_path = './results/{}/{}/{}/{}/'.format(self.ds_name, learn_type, run_name, folder_name)
        os.makedirs(result_path + 'visual', exist_ok=True)
        return result_path
