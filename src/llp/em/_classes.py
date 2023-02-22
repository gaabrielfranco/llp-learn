import numpy as np
from copy import copy
from llp.base import baseLLPClassifier
from abc import ABC
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression

class EM(baseLLPClassifier, ABC):
    """
    EM based algorithm for LLP (KDD 18' paper: Assessing Candidate Preference through Web Browsing History).
    """

    def __init__(self, model, init_y = "random", max_iter=100, random_state=None):
        """
        Args:
            model (sklearn model):
                Model used to fit the data. Can be "SVC", "LinearSVC" or "LogisticRegression".
            
            init_y (str):
                Method used to initialize the labels of the bags.
            
            max_iter (int):
                Maximum number of iterations.
        
            random_state (int):
                Random state seed used in the algorithm.
        """

        self.model = model
        self.max_iter = max_iter
        self.thresholds = None
        self.random = np.random.RandomState(random_state)
        self.model.set_params(random_state=random_state)
        self.init_y_type = init_y

        # Different scoring functions for SVM and LR.
        # For LR, we use the positive class probability
        # FOR SVM, we use the decision function score
        if isinstance(self.model, SVC):
            self.score_function = self.model.decision_function
        elif isinstance(self.model, LogisticRegression):
            self.score_function = self.score_function_LR
        elif isinstance(self.model, LinearSVC):
            self.score_function = self.model.decision_function
        else:
            raise Exception("Invalid Model for EM based approach!")
    
    def score_function_LR(self, X):
        return self.model.predict_proba(X)[:, 1]

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Args:
            **params (dict):
                Estimator parameters.
        """
        self.model.set_params(**params)

    def get_params(self):
        """
        Get parameters for this estimator.

        Args:

        Returns:
            params (dict):
                Estimator parameters.        
        """
        return self.__dict__
    
    def init_y(self, bags, proportions=None):
        if self.init_y_type == "majority":
            return self._create_y_majority(bags, proportions)
        elif self.init_y_type == "random":
            return self._create_y(bags)
        else:
            raise Exception("Invalid init_y for EM based approach")

    def _create_y(self, bags):
        n = len(bags)
        return 2 * self.random.randint(2, size=n) - 1

    def _create_y_majority(self, bags, proportions):
        y = np.full(len(bags), -1)
        num_bags = len(proportions)
        for i in range(num_bags):
            bag = np.where(bags == i)[0]
            if len(bag) != 0:
                if proportions[i] >= 0.5:
                    y[bag] = 1
                else:
                    y[bag] = -1
        return y

    def predict(self, X, bags=None):
        """
        Predict the labels of the samples

        Args:
            X (array-like, shape (n_samples, n_features) ): 
                Training vectors, where n_samples is the number of samples and
                n_features is the number of features.

            bags (array-like, shape (n_samples,) ):
                The bag that each sample belongs to.

        Returns:
            y (array, shape (n_samples,) ):
                The predicted labels of the samples.
        """
        if bags is None:
            return np.array(self.model.predict(X))
        else:
            rows, _ = X.shape
            y = np.full(rows, -1)
            score = self.score_function(X)
            num_bags = int(max(bags) + 1)
            for i in range(num_bags):
                bag = np.where(bags == i)[0]
                if len(bag) != 0:
                    I = [x for x in bag if score[x] >= self.thresholds[i]]
                    if len(I) != 0:
                        y[I] = 1
            return y

    def _optimize_y(self, score, bags, proportions):
        new_y = np.full(len(bags), -1)
        num_bags = len(proportions)
        for i in range(num_bags):
            bag = np.where(bags == i)[0]
            if len(bag) != 0:
                bag_sorted = sorted(bag, key=lambda w: score[w], reverse=True)
                ones = int(round(proportions[i] * len(bag)))
                new_y[bag_sorted[:ones]] = 1
                new_y[bag_sorted[ones:]] = -1
                self.thresholds[i] = score[bag_sorted[ones - 1]]
        return new_y

    def fit(self, X, bags, proportions):
        """
        Fit the model according to the given training data.

        Args:
            X (array-like, shape (n_samples, n_features) ):
                Training vectors, where n_samples is the number of samples and
                n_features is the number of features.
            
            bags (array-like, shape (n_samples,) ):
                The bag that each sample belongs to.

            proportions (array-like, shape (n_bags,) ):
                The proportion of positive samples in each bag.
        """

        self.thresholds = np.full(len(proportions), 0.5)
        y = self.init_y(bags, proportions)

        past_y = y.copy()
        past_error = None
        for _ in range(self.max_iter):
            self.model.fit(X, y)
            score = self.score_function(X)
            y, past_y = past_y, y
            y = self._optimize_y(score, bags, proportions)
            error = np.linalg.norm(y - past_y, ord=0)

            if past_error is not None:
                if abs(past_error - error) < 0.05 * len(y):
                    break
            past_error = error