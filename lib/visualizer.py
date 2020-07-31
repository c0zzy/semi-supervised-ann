import os

import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """
    Class for visualisations of the learning process
    """

    def __init__(self, result_path, funcs_names, num_class, x, y, w_max=0, h=0.01):
        """
        Initialize the Visualizer

        :param result_path: path to save the output to
        :param funcs_names: names of visualizer functions to be used
        :param num_class: number of classes
        :param x: data samples
        :param y: data labels
        :param w_max: maximal weight of unsupervised loss
        :param h: density of the mesh grid
        """
        self.result_path = result_path
        self.history = []
        self.loss_history = []
        self.w_max = w_max

        self.visualise_funcs = [getattr(self, name) for name in funcs_names]

        if self.result_path:
            os.makedirs(result_path + 'visual', exist_ok=True)

        # Only for 2D datasets
        if x.shape[1] == 2:
            self.x = x

            labels = y[:, num_class:num_class * 2]
            self.super_ids = np.nonzero(labels)[0]
            self.unsup_ids = np.where(labels == 0)[0]
            colors = [[1, 0.7, 0], [0, 1, 1]]
            self.c = [colors[l] for l in np.nonzero(labels[self.super_ids])[1]]

            y_min, y_max = x_min, x_max = (0, 1)
            self.xx, self.yy = np.meshgrid(np.arange(x_min, x_max + h, h),
                                           np.arange(y_min, y_max + h, h))

            grid = np.c_[self.xx.ravel(), self.yy.ravel()]
            grid_len = len(grid)

            test_supervised_label_dummy = np.zeros((grid_len, num_class))
            test_supervised_flag_dummy = np.zeros((grid_len, 1))
            test_unsupervised_weight_dummy = np.zeros((grid_len, 1))
            self.grid_ap = [grid, test_supervised_label_dummy, test_supervised_flag_dummy,
                            test_unsupervised_weight_dummy]

    def visualize(self, epoch, model, metrics_avgs, test_acc, show=False):
        """
        Produce all specified plots
        """
        for func in self.visualise_funcs:
            func(model=model,
                 epoch=epoch,
                 loss_metrics=metrics_avgs,
                 test_acc=test_acc,
                 show=show
                 )

    @staticmethod
    def _scatter(x, y=None, c=None, s=5, a=0.4):
        """
        Scatter plot wrapper
        """
        if c is None:
            c = y
        plt.scatter(x[:, 0], x[:, 1], c=c, alpha=a, s=s)

    def border(self, model, epoch, **kwargs):
        """
        Visualize the decision border for 2D datasets
        """
        z = model.predict_on_batch(self.grid_ap)[:, 0]
        z = z.reshape(self.xx.shape)

        fig = plt.figure()
        plt.contourf(self.xx, self.yy, z, cmap='seismic', alpha=1, levels=16, linestyles=None)
        self._scatter(self.x[self.unsup_ids], c=[3 * [0.4]])
        self._scatter(self.x[self.super_ids], c=self.c, a=0.4, s=6)

        plt.axis('off')
        # plt.savefig("test.png", bbox_inches='tight')

        if kwargs['show']:
            fig.show()
        if self.result_path and epoch % 50 == 0 and epoch > 0:
            fig.savefig(self.result_path + 'visual/epoch{:03d}.png'.format(epoch), bbox_inches='tight', pad_inches=0)

    def add_losses(self, losses):
        """
        Memorize current losses for later use in plots
        """
        self.loss_history.append(losses)

    def losses(self, epoch, **kwargs):
        """
        Plot the progress of losses
        :param epoch:

        """
        if len(self.loss_history) == 0:
            return

        semi = np.array([list(self.loss_history[i].values()) for i in range(len(self.loss_history))])
        n = len(semi)
        epochs = list(range(n))

        fig, ax1 = plt.subplots(figsize=(8, 6))

        colors = {
            'loss': 'blue',
            'supervised': 'green',
            'pi_model_labeled': 'cyan',
            'pi_model_unlabeled': 'orange',
            'unsupervised': 'magenta',
            'weight': 'red',
            'score': 'black'
        }

        def get_curve(x):
            return np.array([l[x] for l in self.loss_history])

        for key in self.loss_history[0].keys():
            if key == 'weight':
                ax2 = ax1.twinx()
                ax2.plot(epochs, get_curve(key), color=colors[key], label=key)
                ax2.legend(loc='upper right')
                ax2.set_ylabel('weight', color='red')
            else:
                curve = get_curve(key)
                label = key
                if key == 'unsupervised':
                    curve *= get_curve('weight')
                    label = 'w * unsupervised'
                ax1.plot(epochs, curve, color=colors[key], label=label)

        score = get_curve('supervised') + self.w_max * get_curve('unsupervised')
        ax1.plot(epochs, score, color=colors['score'], label='sup + w_max * unsup')
        ax1.legend(loc='upper right')

        if kwargs['show']:
            fig.show()
        if self.result_path:
            fig.savefig(self.result_path + 'visual/epoch{:03d}.png'.format(epoch))

    def _plot_acc(self):
        epochs, accs, losses = np.hsplit(np.array(self.history), 3)

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training loss', color=color)
        ax1.plot(epochs, losses, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        color = 'tab:blue'
        ax2 = ax1.twinx()
        ax2.set_ylabel('Test accuracy', color=color)
        ax2.plot(epochs, accs, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title('Training loss and test accuracy')

        return fig

    def acc(self, epoch, test_acc, loss, **kwargs):
        """
        Plot accuracy and loss
        """
        self.history.append((epoch, test_acc, loss))

        if kwargs['show']:
            fig = self._plot_acc()
            fig.show()

    def acc_save(self):
        """
        Save the accuracy plot
        """
        fig = self._plot_acc()
        fig.savefig(self.result_path + 'plot.png')
        pass
