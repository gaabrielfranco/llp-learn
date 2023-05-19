from ._base import alterClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import MinMaxScaler


class alterSVMRBF(alterClassifier):
    """
    Alter SVM with RBF kernel implementation
    """
    def __init__(self, C=1.0, gamma='scale', C_p=1.0, d=0.5, max_iter=100, num_exec=10, llp_loss_function_type="abs", weight_method=None, initialization_method="random", random_state=None):
        """
        Args:
            C (float):
                Parameter C from Alter-SVM.

            gamma (float):
                Parameter Gamma from the RBF kernel.

            C_p (float):
                Parameter C_p from the Alter-SVM.

            d (float):
                Parameter Delta from the Alter-SVM.

            max_iter (int):
                Maximum number of iterations.

            num_exec (int):
                Number of times that the algorithm will be repeated.

            llp_loss_function_type (str):
                Type of loss function used in the LLP.

                Currently, the following methods are supported:

                - abs: Absolute value difference.

            weight_method (str):
                Method used to compute the weights of the bags.

                Currently, the following methods are supported:

                - None: No weight is used.

            initialization_method (str):
                Method used to initialize the weights of the bags.

                Currently, the following methods are supported:

                - random: Random initialization.
            
            random_state (int):
                Random state seed used in the algorithm.
            """
        super().__init__(SVC(kernel='rbf', gamma=gamma,
                             random_state=random_state), C=C, C_p=C_p, d=d, max_iter=max_iter,
                         num_exec=num_exec, proba=False, initialization_method=initialization_method,
                         llp_loss_function_type=llp_loss_function_type, weight_method=weight_method,
                         random_state=random_state)
        self.gamma = gamma

    def fit(self, X, bags, proportions):        
        self.X = X
        self.bags = bags
        self.proportions = proportions

        # Selecting the loss function
        if self.llp_loss_function_type == "abs":
            self.llp_loss_function = self.llp_loss_function_abs
        else:
            raise ValueError("Invalid loss function type")
    
        super().alter_classifier()

    def obj_function(self, y, C_star, proportions_pred):
        sv = self.model.support_vectors_
        l = np.ravel(np.ravel(self.model.dual_coef_))
        K = pairwise_kernels(sv, metric='rbf')
        first = 0.5 * np.dot(l, np.dot(K, l))
        second = C_star * \
            np.maximum(0, 1 - y * self.model.decision_function(self.X)).sum()
        third = self.C_p * \
            self.llp_loss_function(proportions_pred, self.proportions).sum()
        return first + second + third

    def model_loss_function(self, X, y):
        return np.maximum(0, 1 - y * self.model.decision_function(X))

    def llp_loss_function(self, pk_estimated, pk, bag_id=None):
        pass

    def llp_loss_function_abs(self, pk_estimated, pk, bag_id=None):
        return np.abs(pk_estimated - pk)

    def llp_loss_function_mse(self, pk_estimated, pk, bag_id=None):
        return (pk_estimated - pk) ** 2

    def llp_loss_function_logcosh(self, pk_estimated, pk, bag_id=None):
        return np.log(np.cosh(pk_estimated - pk))

    def llp_loss_function_weighted_mse(self, pk_estimated, pk, bag_id=None):
        if bag_id is None:
            return self.bag_weight * \
                (self.llp_loss_function_mse(pk_estimated, pk))
        else:
            return self.bag_weight[bag_id] * \
                (self.llp_loss_function_mse(pk_estimated, pk))

    def llp_loss_function_weighted_logcosh(self, pk_estimated, pk, bag_id=None):
        if bag_id is None:
            return self.bag_weight * \
                self.llp_loss_function_logcosh(pk_estimated, pk)
        else:
            return self.bag_weight[bag_id] * \
                self.llp_loss_function_logcosh(pk_estimated, pk)

    def llp_loss_function_weighted_abs(self, pk_estimated, pk, bag_id=None):
        if bag_id is None:
            return self.bag_weight * \
                self.llp_loss_function_abs(pk_estimated, pk)
        else:
            return self.bag_weight[bag_id] * \
                self.llp_loss_function_abs(pk_estimated, pk)

    def llp_loss_function_ada_weighted_abs(self, pk_estimated, pk, bag_id=None):
        if bag_id is None:
            loss_prior = (self.w_size_bag * pk -
                          self.w_size_bag * pk_estimated) ** 2
            loss_mse = self.bag_weight * \
                self.llp_loss_function_mse(pk_estimated, pk)
        else:
            loss_prior = (self.w_size_bag[bag_id] * pk -
                          self.w_size_bag[bag_id] * pk_estimated) ** 2
            loss_mse = self.bag_weight[bag_id] * \
                self.llp_loss_function_mse(pk_estimated, pk)

        return loss_prior + loss_mse

    def delta_computation(self, X_bag):
        delta = self.model_loss_function(X_bag, np.full(
            len(X_bag), -1)) - self.model_loss_function(X_bag, np.full(len(X_bag), 1))
        delta_sort_order = np.argsort(delta)[::-1]
        return delta_sort_order


class alterSVM(alterClassifier):
    """
    Alter SVM with Linear kernel implementation

    Args:
        C (float):
            Parameter C from Alter-SVM.

        C_p (float):
            Parameter C_p from the Alter-SVM.

        d (float):
            Parameter Delta from the Alter-SVM.

        max_iter (int):
            Maximum number of iterations.

        num_exec (int):
            Number of times that the algorithm will be repeated.

        llp_loss_function_type (str):
            Type of loss function used in the LLP.

            Currently, the following methods are supported:

            - abs: Absolute value difference.

        weight_method (str):
            Method used to compute the weights of the bags.

            Currently, the following methods are supported:

            - None: No weight is used.

        initialization_method (str):
            Method used to initialize the weights of the bags.

            Currently, the following methods are supported:

            - random: Random initialization.
        
        random_state (int):
            Random state seed used in the algorithm.
    """
    def __init__(self, C=1.0, C_p=1.0, d=0.5, max_iter=100, num_exec=10, llp_loss_function_type="abs", weight_method=None, initialization_method="random", random_state=None):
        super().__init__(LinearSVC(random_state=random_state, max_iter=10000,
                                   loss='hinge'), C=C, C_p=C_p, d=d, max_iter=max_iter,
                         num_exec=num_exec, proba=False, initialization_method=initialization_method,
                         llp_loss_function_type=llp_loss_function_type, weight_method=weight_method,
                         random_state=random_state)

    def fit(self, X, bags, proportions):
        self.X = X
        self.bags = bags
        self.proportions = proportions

        # Selecting the loss function
        if self.llp_loss_function_type == "abs":
            self.llp_loss_function = self.llp_loss_function_abs
        else:
            raise ValueError("Loss function not implemented")

        super().alter_classifier()

    def obj_function(self, y, C_star, proportions_pred):
        return (0.5 * np.inner(np.ravel(self.model.coef_).transpose(),
                               np.ravel(self.model.coef_))) + (C_star *
                                                               self.model_loss_function(self.X, y).sum()) + (self.C_p
                                                                                                             * self.llp_loss_function(proportions_pred,
                                                                                                                                      self.proportions).sum())

    def model_loss_function(self, X, y):
        return np.maximum(0, 1 - (y * (np.inner(np.ravel(self.model.coef_).transpose(), X) +
                                       self.model.intercept_[0])))

    def llp_loss_function(self, pk_estimated, pk, bag_id=None):
        pass

    def llp_loss_function_abs(self, pk_estimated, pk, bag_id=None):
        return np.abs(pk_estimated - pk)

    def delta_computation(self, X_bag):
        delta = self.model_loss_function(X_bag, np.full(
            len(X_bag), -1)) - self.model_loss_function(X_bag, np.full(len(X_bag), 1))
        delta_sort_order = np.argsort(delta)[::-1]
        return delta_sort_order