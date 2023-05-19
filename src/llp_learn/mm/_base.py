from llp_learn.base import baseLLPClassifier
from abc import ABC, abstractmethod
import numpy as np
from scipy.special import expit

class MMBaseClassifier(baseLLPClassifier, ABC):
    """
    Base class for all MM classifiers - (Patriani, 2014) paper.
    """
    def __init__(self, lmd=1, random_state=None):
        self.lmd = lmd
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, bags, proportions):
        pass

    def predict(self, X):
        return np.where(self.predict_proba(X) >= 0.5, 1, -1)
    
    def predict_proba(self, X):
        return expit(2 * X @ self.w)

    def set_params(self, **params):
        for param in params:
            self.__dict__[param] = params[param]

    def get_params(self):
        return self.__dict__