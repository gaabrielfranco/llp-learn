from llp_learn.base import baseLLPClassifier
from llp_learn.util import check_random_state
from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy

class alterClassifier(baseLLPClassifier, ABC):
    def __init__(self, model, C, C_p, d=0.5, max_iter=100, num_exec=10, proba=False, llp_loss_function_type = "abs", initialization_method = "random", weight_method = None, random_state=None):
        self.C = C
        self.C_p = C_p
        self.d = d
        self.max_iter = max_iter
        self.num_exec = num_exec
        self.model = model
        self.proba = proba
        self.initialization_method = initialization_method
        self.llp_loss_function_type = llp_loss_function_type
        self.weight_method = weight_method
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, bags, proportions):
        pass

    def predict(self, X):
        return np.array(self.model.predict(X))

    @abstractmethod
    def obj_function(self, y, proportions_pred):
        pass

    @abstractmethod
    def model_loss_function(self, X, y):
        pass

    @abstractmethod
    def llp_loss_function(self, pk_estimated, pk, bag_id):
        pass

    def set_params(self, **params):
        for param in params:
            self.__dict__[param] = params[param]

    def get_params(self):
        return self.__dict__

    @abstractmethod
    def delta_computation(self, X_bag):
        pass

    def _optimize_y_per_bag(self, bag, bag_id, pk, C_star):
        if self.proba:
            X_bag = self.predicted_proba[bag]
        else:
            X_bag = self.X[bag]

        delta_sort_order = self.delta_computation(X_bag)
        new_y = np.full(len(X_bag), -1)

        solution_array = np.zeros(len(bag) + 1)

        negative_flips = self.model_loss_function(
            X_bag, np.full(len(X_bag), -1))
        positive_flips = self.model_loss_function(
            X_bag, np.full(len(X_bag), 1))

        solution_array[0] = negative_flips.sum(
        ) + (self.C_p / C_star) * self.llp_loss_function(0, pk, bag_id)

        R = np.arange(1, len(bag) + 1)
        top_R = delta_sort_order[R - 1]

        sol_with_flip = positive_flips[top_R] + \
            ((self.C_p / C_star) * self.llp_loss_function(R / len(bag), pk, bag_id))
        sol_without_flip = negative_flips[top_R] + (
            (self.C_p / C_star) * self.llp_loss_function((R - 1) / len(bag), pk, bag_id))

        
        for i in range(1, len(bag) + 1):
            solution_array[i] = solution_array[i - 1] - \
                sol_without_flip[i - 1] + sol_with_flip[i - 1]

        R = np.argmin(solution_array)
        min_theta = R / len(bag)
        top_R = delta_sort_order[:R]

        new_y[top_R] *= -1

        return new_y, min_theta

    def _optimize_y(self, y, C_star):
        new_y = np.full(len(y), -1)

        num_bags = len(self.proportions)
        proportions_pred = np.zeros(len(self.proportions))

        if self.proba:
            self.predicted_proba = self.model.predict_proba(self.X)

        for i in range(num_bags):
            bag = np.where(self.bags == i)[0]
            if len(bag) != 0:
                y_bag, proportions_pred[i] = self._optimize_y_per_bag(
                    bag, i, self.proportions[i], C_star)
                new_y[bag] = y_bag

        return new_y, proportions_pred

    def _initialize_y(self):
        if self.initialization_method == "random":
            new_y = self.random.randint(2, size=len(self.X), dtype="int8")
            new_y[new_y == 0] = -1
        else:
            raise ValueError("Invalid initialization method")
        return new_y

    def alter_classifier(self):
        min_obj_function = np.inf
        best_model = None

        self.random = check_random_state(self.random_state)

        # Execute K times to avoid bad solution (Yu et al., 2013)
        for i in range(self.num_exec):
            y = self._initialize_y()

            C_star = 10e-5 * self.C

            # Values of objetive function
            old_obj_function = 0.0
            new_obj_function = 0.0

            # Avoid dots to improve the performance
            set_params = self.model.set_params
            fit = self.model.fit
           
            # if there are convergence problems (all y's become the same)
            # this flag becomes True and the external while is broken
            convergence_error = False

            best_obj_function = np.inf

            while C_star < self.C and not convergence_error:
                C_star = min((1 + self.d) * C_star, self.C)
                
                old_obj_function = np.inf
                new_obj_function = 0.0

                set_params(**{"C": C_star})

                # Internal loop
                num_it = 0  # The number of maximum iterations

                best_y = None
                best_obj_function = np.inf
                best_model_loop = None

                while True:
                    # Step 1 - Fix y to solve beta
                    try:
                        fit(self.X, y)
                    except ValueError:
                        convergence_error = True
                        break

                    # Step 2 - Fix beta to solve y
                    y, proportions_pred = self._optimize_y(y, C_star)
                    new_obj_function = self.obj_function(
                        y, C_star, proportions_pred)

                    # Saving the best execution results in the internal loop
                    if new_obj_function <= best_obj_function:
                        best_obj_function = new_obj_function
                        best_y = deepcopy(y)
                        best_model_loop = deepcopy(self.model)

                    if np.abs(new_obj_function - old_obj_function) <= 10e-4:
                        # Picking the best y in the internal loop
                        y = deepcopy(best_y)
                        break
                    old_obj_function = new_obj_function
                    num_it += 1
                    if num_it == self.max_iter:
                        print("\t", num_it, "iterations without convergence")
                        break

            # The final solution is the execution with the lowest objective function value
            if best_obj_function < min_obj_function and not convergence_error:
                min_obj_function = best_obj_function
                best_model = deepcopy(best_model_loop)

        # Setting the model
        if best_model is None:
            raise ValueError("This combination of (C = %.5f, C_p = %.5f) does not converge!" % (self.C, self.C_p))
        self.model = best_model