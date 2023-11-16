import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score
import pandas as pd
import itertools
from joblib import Parallel, delayed, parallel_backend
import pandas as pd
import os
import warnings

import torch
from ._split import SplitBagKFold, SplitBagBootstrapSplit, SplitBagShuffleSplit, FullBagStratifiedKFold
import numbers


__all__ = ["gridSearchCV"]


class gridSearchCV():
    """
    Exhaustive search over specified parameter values for an estimator.
    """
    def __init__(self, estimator, param_grid, refit=True, cv=None, splitter="split-bag-shuffle", loss_type="abs", validation_size=0.2, cv_type="std", central_tendency_metric="mean", n_jobs=1, random_state=None):
        """
        Args:
            estimator (object): 
                This is assumed to implement the baseLLPClassifier estimator interface.
            
            param_grid (dict): 
                Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.

            refit (bool): 
                Refit an estimator using the best found parameters on the whole dataset. Defaults to True.

            cv (int, cross-validation generator or an iterable): 
                Determines the cross-validation splitting strategy.

            splitter (str): 
                The splitter used to split the dataset. The default value is "split-bag-shuffle". 

            loss_type (str): 
                The loss type used to evaluate the performance of the model. The default value is "abs".

            validation_size (float): 
                The size of the validation dataset. The default value is 0.2. The value must be between 0 and 1.

            cv_type (str): 
                The type of cross validation used to evaluate the performance of the model. The default value is "std".

            central_tendency_metric (str): 
                The central tendency metric used to evaluate the performance of the model. The default value is "mean".

            n_jobs (int): 
                Number of jobs to run in parallel. -1 means using all processors. Defaults to 1.

            random_state (int): 
                Controls the randomness of the estimator. The default value is None.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.refit = refit
        self.cv = cv
        self.splitter = splitter
        self.loss_type = loss_type
        self.validation_size = validation_size
        self.cv_type = cv_type
        self.central_tendency_metric = central_tendency_metric
        self.n_jobs = n_jobs
        self.best_params_ = None
        self.best_estimator_ = None
        self.random_state = random_state

    def _product_dict(self, **kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    def _bag_loss(self, proportions, predicted_proportions):
        if self.loss_type == "abs":
            proportions_error = np.abs(predicted_proportions - proportions)
            proportions_error[np.isnan(proportions_error)] = 0.0
            return proportions_error
        else:
            raise Exception(
                "There was not possible to compute the error. Verify the loss_type parameter. The value used was: %s" % self.loss_type)

    def _evaluate_candidate(self, X, bags, proportions, arg):
        est, param_id, param, train_index, validation_index = arg
        estimator = deepcopy(est)
        estimator.set_params(**param)
        try:
            estimator.fit(X[train_index], bags[train_index], proportions)
        except ValueError:
            return None
        y_pred_validation = estimator.predict(X[validation_index])

        if self.cv_type == "std":
            predicted_proportions = np.empty(len(proportions))
            bag_size_validation = np.empty(len(proportions), int)
            num_bags = len(proportions)

            # Computing the predicted proportions and the size of bags in the validation set
            for i in range(num_bags):
                bag_validation = np.where(bags[validation_index] == i)[0]
                bag_size_validation[i] = len(bag_validation)
                y_pred_bag_validation = y_pred_validation[bag_validation]
                if len(bag_validation) == 0:
                    predicted_proportions[i] = np.nan
                else:
                    predicted_proportions[i] = np.count_nonzero(
                        y_pred_bag_validation == 1) / len(y_pred_bag_validation)

            err = self._bag_loss(proportions, predicted_proportions).sum()
        elif self.cv_type == "oracle":
            err = 1 - \
                accuracy_score(self.y[validation_index], y_pred_validation)
        else:
            raise Exception(
                "There was not possible to compute the error. Verify the cv_type parameter. The value used was: %s" % self.cv_type)

        return (estimator, param_id, param, train_index, validation_index, err)
    
    def _aggregate_results(self, r):
        df = pd.DataFrame(
            r, columns="model id params train_index validation_index error".split())
        
        # Removing hyperparameters that do not converged in all folds
        df = df.groupby("id").filter(lambda x: len(x) == self.cv)

        if self.central_tendency_metric == "mean":
            df_results = df["id error".split()].groupby("id").mean()
        elif self.central_tendency_metric == "median":
            df_results = df["id error".split()].groupby("id").median()
        else:
            raise Exception(
                "There was not possible to computate the error. Verify the central_tendency_metric parameter.")

        return df, df_results
    
    def _fit_best_estimator(self, X, bags, proportions, df, df_results):
        best_estimator_id = int(df_results.idxmin())
        df_best_estimator = df[df.id == best_estimator_id]
        self.best_params_ = df_best_estimator["params"].iloc[0]

        if self.refit:
            self.best_estimator_ = deepcopy(self.estimator)
            self.best_estimator_.set_params(**self.best_params_)

            try:
                self.best_estimator_.fit(X, bags, proportions)
            except ValueError:
                self.best_estimator_ = df_best_estimator[df_best_estimator.error == df_best_estimator.error.min(
                )].iat[0, 0]
                warnings.warn("The best hyperparameters found by the CV process did not converge in the refit process. \
                    The best hyperparameters are " + str(self.best_params_))
        else:
            self.best_estimator_ = df_best_estimator[df_best_estimator.error == df_best_estimator.error.min(
            )].iat[0, 0]


    def fit(self, X, bags, proportions, y=None):
        """
        Fit the model according to the given training data.

        Args:
            X (array-like, shape (n_samples, n_features)): Training vector, where n_samples is the number of samples and n_features is the number of features.
        """
        self.best_params_ = None
        self.best_estimator_ = None
        self.y = y

        if self.y is None and self.cv_type == "oracle":
            raise Exception("y must be passed when cv_type = %s" %
                            (self.cv_type))

        # Step 1 - Compute the folds that will be used in CV
        if self.cv is None:
            self.cv = 5

        if isinstance(self.cv, numbers.Integral):
            if self.splitter == "split-bag-shuffle":
                folds = list(SplitBagShuffleSplit(
                    n_splits=self.cv, test_size=self.validation_size, random_state=self.random_state).split(X, bags))
            elif self.splitter == "split-bag-k-fold":
                folds = list(SplitBagKFold(
                    n_splits=self.cv, shuffle=True, random_state=self.random_state).split(X, bags))
            elif self.splitter == "split-bag-bootstrap":
                folds = list(SplitBagBootstrapSplit(
                    n_splits=self.cv, test_size=self.validation_size, random_state=self.random_state).split(X, bags))
            elif self.splitter == "full-bag-stratified-k-fold":
                folds = list(FullBagStratifiedKFold(
                    n_splits=self.cv, random_state=self.random_state).split(X, bags, proportions))
            else:
                raise ValueError("splitter %s do not exist" % (self.splitter))
        elif isinstance(self.cv, list):
            folds = self.cv
        else:
            # Receive a CV generator (must have the split method)
            try:
                folds = list(self.cv.split(X))
            except:
                try:
                    folds = list(self.cv.split(X, bags))
                except:
                    raise Exception(
                        "There was not possible to create the folds. Verify the cv parameter.")

        self.n_splits = len(folds)
        self.bag_size = np.empty(len(proportions), int)
        num_bags = len(proportions)

        # Computing the size of bags
        for i in range(num_bags):
            bag_i = np.where(bags == i)[0]
            self.bag_size[i] = len(bag_i)

        if isinstance(self.param_grid, dict):
            params = list(self._product_dict(**self.param_grid))
            params = list(zip(range(len(params)), params))  # [(id, {})]
        elif isinstance(self.param_grid, list):
            params = list(zip(range(len(self.param_grid)),
                              self.param_grid))  # [(id, {})]
        else:
            raise Exception(
                "There was not possible to create the parameters combination. Verify the param_grid parameter.")

        # Step 2 - Create a arguments list with folds and hyperparameters
        candidates = []
        for param, fold in itertools.product(params, folds):
            param_id, param = param
            train_index, validation_index = fold
            arg = [self.estimator, param_id, param,
                   train_index, validation_index]
            candidates.append(arg)

        # Step 3 - Call the Parallel method
        if self.n_jobs == -1:
            if os.cpu_count() is None:
                inner_max_num_threads = 1
            else:
                inner_max_num_threads = (os.cpu_count() // len(candidates))
        else:
            inner_max_num_threads = (self.n_jobs // len(candidates))

        if inner_max_num_threads == 0:
            inner_max_num_threads = 1

        with parallel_backend("loky", inner_max_num_threads=inner_max_num_threads):
            r = Parallel(n_jobs=self.n_jobs, verbose=10)(
                delayed(self._evaluate_candidate)(X, bags, proportions, arg) for arg in candidates)

        # Step 4 - Filter the results and verify the convergence
        r = [x for x in r if x is not None]

        if len(r) == 0:
            raise ValueError("There was no (C, C_p) with convergence!")
        
        # Step 5 - Aggregate the results by hyperparameters and compute the mean for each one
        df, df_results = self._aggregate_results(r)

        # Step 6 - Choose the best estimator
        self._fit_best_estimator(X, bags, proportions, df, df_results)

    def predict(self, X, bags=None):
        if bags is None:
            return self.best_estimator_.predict(X)
        else:
            return self.best_estimator_.predict(X, bags)