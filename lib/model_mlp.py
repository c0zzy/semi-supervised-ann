from keras.layers import Input, Dense, concatenate, Dropout

from keras.models import Model


def build_model(input_size, num_classes, mlp_arch, dropout=False):
    """
    Builds the underlying model - MLP with specified architecture

    :param input_size: size of the input
    :param num_classes: number of target classes
    :param mlp_arch: MLP architecture
    :param dropout: whether to use dropout
    :return: MLP model
    """

    input_data = Input(shape=(input_size,), name='input-data')
    supervised_label = Input(shape=(num_classes,), name='input-label')
    supervised_flag = Input(shape=(1,), name='input-flag')
    unsupervised_weight = Input(shape=(1,), name='input-unsup-w')

    net = input_data
    for i, units in enumerate(mlp_arch):
        net = Dense(units, activation='relu', input_shape=(2,))(net)
        if dropout:
            net = Dropout(0.5)(net)

    net = Dense(num_classes, activation='softmax', name='output-softmax')(net)
    net = concatenate([net, supervised_label, supervised_flag, unsupervised_weight])

    return Model([input_data, supervised_label, supervised_flag, unsupervised_weight], net)
