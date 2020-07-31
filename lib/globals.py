import numpy as np

random_seed = 42
rng = np.random.RandomState(random_seed)
state = rng.get_state()


def reset_random_state():
    """
    Reset the random stat - for testing purposes
    """
    rng.set_state(state)
