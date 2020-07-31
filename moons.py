from alg.semi_supervised import SemiSupervised
from lib.utils import load_moons, split_supervised_train

dataset_moons = {
    'name': 'moons',
    'input_size': 2,
    'num_classes': 2,
    'mlp_arch': [64, 32],
    'visualise_funcs': ['border'],
}


def train_model(x_sup, y_sup, x_sup_test, y_sup_test, x_un):
    semi_sup = SemiSupervised(
        dataset_params=dataset_moons,
        num_epoch=101,
        ssl_method='pi-model',  # or 'pseudo-label' or None
        options={'augment': True, 'dropout': False, 'noise': 0.05},
        hyper_par={'weight_max': 20, 'ramp_up': 30},
        save_results=False,
        imbalanced=False,
        should_evaluate=True
    )

    semi_sup.set_train_data(x_sup, y_sup)
    semi_sup.set_test_data(x_sup_test, y_sup_test)
    semi_sup.set_unsupervised_data(x_un)
    semi_sup.prepare_train_test_data()
    model = semi_sup.train(1)

    return model


def main():
    x_sup_train, y_sup_train, x_sup_test, y_sup_test = load_moons(1000, 400)
    x_sup_train, y_sup_train, x_unsup = split_supervised_train(x_sup_train, y_sup_train, 200, 1)

    train_model(x_sup_train, y_sup_train, x_sup_test, y_sup_test, x_unsup)
    return


if __name__ == '__main__':
    main()
