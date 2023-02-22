import numpy as np
import numbers

__all__ = ["compute_proportions", "check_random_state"]


def compute_proportions(bags, y):
    """
    Compute the proportion of positive samples in each bag.

    Parameters
    ----------
    bags : array-like, shape (n_samples,)
        The bag that each sample belongs to.
    
    y : array-like, shape (n_samples,)
        The labels of the samples.
    
    Returns
    -------
    proportions : array, shape (n_bags,)
        The proportion of positive samples in each bag.
    """
    num_bags = int(max(bags)) + 1
    proportions = np.empty(num_bags, dtype=float)
    for i in range(num_bags):
        bag = np.where(bags == i)[0]
        if len(bag) == 0:
            proportions[i] = np.nan
        else:
            proportions[i] = np.count_nonzero(y[bag] == 1) / len(bag)
    return proportions


def check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.

        If seed is an int, return a new RandomState instance seeded with
        seed.

        If seed is already a RandomState instance, return it.

        Otherwise raise ValueError.

    Returns
    -------
    random_state : RandomState instance
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
