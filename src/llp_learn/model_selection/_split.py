import collections
import numpy as np
from math import ceil
import numbers
from llp_learn.util import check_random_state
from collections import defaultdict

__all__ = ["SplitBagKFold", "FullBagStratifiedKFold", "SplitBagShuffleSplit", "SplitBagBootstrapSplit"]

class SplitBagKFold():
    """
        Split-Bag K-Fold cross-validator
        Provides train/test indices to split data in train/test sets. Split
        each bag into k consecutive folds (without shuffling by default).
        The union of fold i per bag is used once as validation while the union of the k - 1 remaining
        folds form the training set.

        Args:
            n_splits (int):
                Number of folds. Must be at least 2.

            shuffle (bool): 
                Whether to shuffle the data before splitting into batches.

            random_state (int, RandomState instance or None):
                If int, random_state is the seed used by the random number generator;

                If RandomState instance, random_state is the random number generator;

                If None, the random number generator is the RandomState instance used
                by `np.random`. Only used when ``shuffle`` is True. This should be left
                to None if ``shuffle`` is False.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError("The number of folds must be of Integral type. "
                             "%s of type %s was passed."
                             % (n_splits, type(n_splits)))
        n_splits = int(n_splits)
        self.n_splits = n_splits

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits))

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False;"
                            " got {0}".format(shuffle))

        if not shuffle and random_state is not None:
            raise ValueError(
                "Setting a random_state has no effect since shuffle is "
                "False. This will raise an error in 0.24. You should leave "
                "random_state to its default (None), or set shuffle=True.")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, bags):
        """
        Generate indices to split data into training and test set.

        Args:
            X (array-like, shape (n_samples, n_features)):
                Training data, where n_samples is the number of samples and n_features is the number of features.
            
            bags (array-like, shape (n_samples,)):
                The bag that each element is into.
            
        Yields:
            train (ndarray):
                The training set indices for that split.

            test (ndarray):
                The testing set indices for that split.
        """

        num_bags = bags.max() + 1

        all_bags = []
        all_fold_sizes = []

        # Computing the index of each bag and the size of each fold per bag
        for i in range(num_bags):
            bag_i = np.where(bags == i)[0]
            all_bags.append(bag_i)

            n_samples = len(bag_i)
            if self.n_splits > n_samples:
                raise ValueError(
                    ("Cannot have number of splits n_splits={0} greater"
                     " than the number of samples: n_samples={1}.")
                    .format(self.n_splits, n_samples))

            fold_sizes = np.full(self.n_splits, n_samples //
                                 self.n_splits, dtype=int)
            fold_sizes[:n_samples % self.n_splits] += 1
            all_fold_sizes.append(fold_sizes)

        if self.shuffle:
            random = check_random_state(self.random_state)

        # Generating the train/test index
        for i in range(self.n_splits):
            train_index = np.empty(0, dtype=int)
            test_index = np.empty(0, dtype=int)
            for j in range(num_bags):
                bag_j = all_bags[j]
                if self.shuffle:
                    random.shuffle(bag_j)
                fold_size_bag_j = all_fold_sizes[j]
                start = fold_size_bag_j[:i].sum()
                end = start + fold_size_bag_j[i]
                test_index = np.concatenate((test_index, bag_j[start:end]))
                train_index = np.concatenate(
                    (train_index, np.setdiff1d(bag_j, bag_j[start:end])))
            if self.shuffle:
                yield random.permutation(train_index), random.permutation(test_index)
            else:
                yield train_index, test_index

class SplitBagShuffleSplit():
    """
    Split-Bag Shuffle cross-validator per bag
    Yields indices to split data into training and test sets. The train and
    test sizes are used to sample elements per bag.
    The splits are made by preserving the percentage of samples for each class per bag.

    Args:
        n_splits (int):
            Number of re-shuffling & splitting iterations.
        
        test_size (float):
            Should be between 0.0 and 1.0 and represent the proportion
            of the each bag to include in the test split.

        random_state (int, RandomState instance or None):
            If int, random_state is the seed used by the random number generator;

            If RandomState instance, random_state is the random number generator;

            If None, the random number generator is the RandomState instance used
            by `np.random`.

    """

    def __init__(self, n_splits=5, test_size=0.25, random_state=None):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError("The number of folds must be of Integral type. "
                             "%s of type %s was passed."
                             % (n_splits, type(n_splits)))

        if not isinstance(test_size, numbers.Real) and not isinstance(test_size, (collections.abc.Sequence, np.ndarray)):
            raise ValueError("The test size must be of Real type or an array. "
                             "%s of type %s was passed."
                             % (test_size, type(test_size)))
        
        if isinstance(test_size, numbers.Real):
            test_size = np.array([test_size])
        else:
            test_size = np.array(test_size)
        
        if np.any(test_size <= 0.0) or np.any(test_size >= 1.0):
            raise ValueError("The test size must be between 0 and 1. "
                             "%s was passed."
                             % (test_size))

        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, bags):
        """
        Generate indices to split data into training and test set.

        Args:
            X (array-like, shape (n_samples, n_features)):
                Training data, where n_samples is the number of samples and n_features is the number of features.
            
            bags (array-like, shape (n_samples,)):
                The bag that each element is into.
            
        Yields:
            train (ndarray):
                The training set indices for that split.

            test (ndarray):
                The testing set indices for that split.
        """

        num_bags = len(np.unique(bags))

        if len(self.test_size) == 1:
            self.test_size = np.full(num_bags, self.test_size[0])

        if len(self.test_size) != num_bags:
            raise ValueError("The length of test_size is not the same as the number of bags. "
                             "%d != %d" % (len(self.test_size), num_bags))

        all_bags = []
        n_test_bags = []

        # Computing the index of each bag and the number of test examples in each bag
        for i in range(num_bags):
            bag_i = np.where(bags == i)[0]
            all_bags.append(bag_i)
            n_test = round(self.test_size[i] * len(bag_i))
            n_test_bags.append(n_test)

        random = check_random_state(self.random_state)

        # Generating the train/test index
        for _ in range(self.n_splits):
            train_index = np.empty(0, dtype=int)
            test_index = np.empty(0, dtype=int)
            for i in range(num_bags):
                bag_i = random.permutation(all_bags[i])
                test_index = np.concatenate(
                    (test_index, bag_i[:n_test_bags[i]]))
                train_index = np.concatenate(
                    (train_index, bag_i[n_test_bags[i]:]))

            yield random.permutation(train_index), random.permutation(test_index)


class SplitBagBootstrapSplit():
    """
    Split-Bag Boostrap cross-validator per bag
    Yields indices to split data into training and test sets. The train and
    test sizes are used to sample elements per bag.
    The splits are made by random sampling with replacement per bag.
    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.
    Args:
        n_splits (int):
            Number of re-shuffling & splitting iterations.
        
        test_size (float):
            Should be between 0.0 and 1.0 and represent the proportion
            of the each bag to include in the test split.

        random_state (int, RandomState instance or None):
            If int, random_state is the seed used by the random number generator;

            If RandomState instance, random_state is the random number generator;

            If None, the random number generator is the RandomState instance used
            by `np.random`.

    """

    def __init__(self, n_splits=5, test_size=0.2, random_state=None):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError("The number of folds must be of Integral type. "
                             "%s of type %s was passed."
                             % (n_splits, type(n_splits)))

        if not isinstance(test_size, numbers.Real) and not isinstance(test_size, (collections.abc.Sequence, np.ndarray)):
            raise ValueError("The test size must be of Real type or an array. "
                             "%s of type %s was passed."
                             % (test_size, type(test_size)))
        
        if isinstance(test_size, numbers.Real):
            test_size = np.array([test_size])
        else:
            test_size = np.array(test_size)
        
        if np.any(test_size <= 0.0) or np.any(test_size >= 1.0):
            raise ValueError("The test size must be between 0 and 1. "
                             "%s was passed."
                             % (test_size))
        
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, bags):
        """
        Generate indices to split data into training and test set.

        Args:
            X (array-like, shape (n_samples, n_features)):
                Training data, where n_samples is the number of samples and n_features is the number of features.
            
            bags (array-like, shape (n_samples,)):
                The bag that each element is into.
            
        Yields:
            train (ndarray):
                The training set indices for that split.

            test (ndarray):
                The testing set indices for that split.
        """

        num_bags = len(np.unique(bags))

        if len(self.test_size) == 1:
            self.test_size = np.full(num_bags, self.test_size[0])

        if len(self.test_size) != num_bags:
            raise ValueError("The length of test_size is not the same as the number of bags. "
                             "%d != %d" % (len(self.test_size), num_bags))

        all_bags = []
        n_test_bags = []

        # Computing the index of each bag and the number of test examples in each bag
        for i in range(num_bags):
            bag_i = np.where(bags == i)[0]
            all_bags.append(bag_i)
            n_test = round(self.test_size[i] * len(bag_i))
            n_test_bags.append(n_test)

        random = check_random_state(self.random_state)

        # Generating the train/test index
        for _ in range(self.n_splits):
            train_index = np.empty(0, dtype=int)
            test_index = np.empty(0, dtype=int)
            for i in range(num_bags):
                bag_i = all_bags[i]
                test_size_bag = n_test_bags[i]
                train_size_bag = len(bag_i) - test_size_bag
                test_index = np.concatenate(
                    (test_index, random.choice(all_bags[i], test_size_bag)))
                train_index = np.concatenate(
                    (train_index, random.choice(all_bags[i], train_size_bag)))

            yield random.permutation(train_index), random.permutation(test_index)

class FullBagStratifiedKFold():
    """
        Full-Bag K-Fold cross-validator. Implementation of the paper 
        "A framework for evaluation in learning from label proportions".
        Provides train/test indices to split data in train/test sets. Split
        each bag into k consecutive folds (without shuffling by default).
        The union of fold i per bag is used once as validation while the union of the k - 1 remaining
        folds form the training set.

        Args:
            n_splits (int):
                Number of folds. Must be at least 2.
            
            shuffle (boolean):
                Whether to shuffle the data before splitting into batches.

            random_state (int, RandomState instance or None):
                If int, random_state is the seed used by the random number generator;

                If RandomState instance, random_state is the random number generator;

                If None, the random number generator is the RandomState instance used
                by `np.random`. Only used when ``shuffle`` is True. This should be left
                to None if ``shuffle`` is False.
    """

    def __init__(self, n_splits=5, random_state=None):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError("The number of folds must be of Integral type. "
                             "%s of type %s was passed."
                             % (n_splits, type(n_splits)))
        n_splits = int(n_splits)
        self.n_splits = n_splits

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits))

        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, bags, proportions):
        """
        Generate indices to split data into training and test set.

        Args:
            X (array-like, shape (n_samples, n_features)):
                Training data, where n_samples is the number of samples and n_features is the number of features.
            
            bags (array-like, shape (n_samples,)):
                The bag that each element is into.
            
            proportions (array-like, shape (n_bags,)):
                The proportion of each bag.
            
        Yields:
            train (ndarray):
                The training set indices for that split.

            test (ndarray):
                The testing set indices for that split.
        """

        if proportions.ndim == 1:
            n_classes = 2
        else:
            n_classes = proportions.shape[1]

        random = check_random_state(self.random_state)

        num_bags = len(np.unique(bags))
        len_bags = np.zeros(num_bags, int)
        bag_list = list(range(num_bags))

        for i in range(num_bags):
            len_bags[i] = len(np.where(bags == i)[0])


        if n_classes == 2:
            m_plus = np.round(len_bags * proportions).astype(int)
            m_minus = len_bags - m_plus
            p_plus = (len_bags * proportions).sum() / len_bags.sum()

            m_plus_squared_sum = (m_plus ** 2).sum()
            m_minus_squared_sum = (m_minus ** 2).sum()
            assigned_bags = []
            f_sizes = np.zeros(self.n_splits, dtype=int)
            f_plus = np.zeros(self.n_splits, dtype=int)
            folds = defaultdict(set)

            # Creating the folds
            while len(assigned_bags) != num_bags:
                f = np.argmin(f_sizes)
                f_p = f_plus[f] / f_sizes[f] if f_sizes[f] != 0 else 0.0
                if f_p <= p_plus:
                    c = (m_plus ** 2) / m_plus_squared_sum
                else:
                    c = (m_minus ** 2) / m_minus_squared_sum

                c[assigned_bags] = 0

                b = random.choice(bag_list, size = 1, replace = False, p = c)[0]

                folds[f].add(b)
                assigned_bags.append(b)
                f_sizes[f] += len_bags[b]
                f_plus[f] += m_plus[b]
                m_plus_squared_sum -= m_plus[b] ** 2
                m_minus_squared_sum -= m_minus[b] ** 2
        else:
            # Multiclass
            m = np.round(len_bags.reshape(-1, 1) * proportions).astype(int)
            global_prop = np.sum(m, axis=0) / np.sum(m)
            assigned_bags = []
            f_sizes = np.zeros((self.n_splits, n_classes), dtype=int)
            folds = defaultdict(set)

            # Creating the folds
            while len(assigned_bags) != num_bags:
                f = np.argmin(np.sum(f_sizes, axis=1))
                c = m + f_sizes[f]
                c_prop = c / np.sum(c, axis=1).reshape(-1, 1)
                c_norm = np.linalg.norm(c_prop - global_prop, axis=1)
                probs = 1 - c_norm 
                probs[assigned_bags] = 0 
                probs = probs / np.sum(probs)
                b = random.choice(bag_list, size = 1, replace = False, p = probs)[0]
                folds[f].add(b)
                assigned_bags.append(b)
                f_sizes[f] += m[b]

        # Generating the train/test indexes
        for i in range(self.n_splits):
            bags_test = list(folds[i])
            test_index = np.where(np.isin(bags, bags_test))[0]
            train_index = np.setdiff1d(np.array(range(len(bags))), test_index)

            yield random.permutation(train_index), random.permutation(test_index)