from abc import ABC, abstractmethod

class baseLLPClassifier(ABC):
    """
    Base class for LLP classifiers.

    All LLP classifiers should inherit from this class.
    """
    @abstractmethod
    def fit(self, X, bags, proportions):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        
        bags : array-like, shape (n_samples,)
            The bag that each sample belongs to.

        proportions : array-like, shape (n_bags,)
            The proportion of positive samples in each bag.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y : array, shape (n_samples,)
            Class labels for each data sample.
        """
        pass
