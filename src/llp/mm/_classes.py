import numpy as np
import pandas as pd
from ._base import MMBaseClassifier
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from urllib.request import urlopen
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
import rpy2

class MM(MMBaseClassifier):
    def __init__(self, lmd, random_state=None):
        super().__init__(lmd, random_state)

    def fit(self, X, bags, proportions):
        """
        Fit the model according to the given training data.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The training input samples.
        bags : array-like, shape = (n_samples,)
            The training bags.
        proportions : array-like, shape = (n_bags,)
            The bags proportions.
        """
        
        bioc_url = urlopen('https://raw.githubusercontent.com/giorgiop/almostnolabel/master/mean.map.R')
        string = bioc_url.read().decode('utf-8')
        mm = SignatureTranslatedAnonymousPackage(string, "map.mean")
        self.map_mean = mm.mean_map


        pandas2ri.activate()

        # Creating the R object expected by the R function
        y_proportions = np.array([proportions[bag] for bag in bags])
        trainset = pd.DataFrame(np.concatenate((y_proportions.reshape(-1, 1), bags.reshape(-1, 1), X), axis=1), columns=["label", "bag"] + ["x" + str(i) for i in range(X.shape[1])])
        trainset = trainset.astype({"bag": int})

        # Calling the R function
        self.w = self.map_mean(trainset, self.lmd)

class LMM(MMBaseClassifier):
    def __init__(self, lmd, gamma, sigma, similarity="G,s", random_state=None):
        super().__init__(lmd, random_state)
        self.gamma = gamma
        self.sigma = sigma
        self.similarity = similarity

    def fit(self, X, bags, proportions):
        """
        Fit the model according to the given training data.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The training input samples.
        bags : array-like, shape = (n_samples,)
            The training bags.
        proportions : array-like, shape = (n_bags,)
            The bags proportions.
        """
        bioc_url = urlopen('https://raw.githubusercontent.com/giorgiop/almostnolabel/master/laplacian.mean.map.R')
        string = bioc_url.read().decode('utf-8')
        lmm = SignatureTranslatedAnonymousPackage(string, "laplacian.mean.map")
        self.laplacian = lmm.laplacian
        self.laplacian_mean_map = lmm.laplacian_mean_map

        pandas2ri.activate()

        # Creating the R object expected by the R function
        y_proportions = np.array([proportions[bag] for bag in bags])
        trainset = pd.DataFrame(np.concatenate((y_proportions.reshape(-1, 1), bags.reshape(-1, 1), X), axis=1), columns=["label", "bag"] + ["x" + str(i) for i in range(X.shape[1])])
        trainset = trainset.astype({"bag": int})

        N_bags = len(np.unique(bags))

        # Computing the laplacian 
        laplacian = self.laplacian(self.similarity, trainset, N_bags, self.sigma)

        # Calling the R function
        self.w = self.laplacian_mean_map(trainset, laplacian, self.lmd, self.gamma)

class AMM(MMBaseClassifier):
    def __init__(self, lmd, gamma, sigma, similarity="G,s", init="LMM", random_state=None):
        super().__init__(lmd, random_state)
        self.gamma = gamma
        self.sigma = sigma
        if similarity not in ["G,s", "NC,s"]:
            raise ValueError("Similarity must be either 'G,s' or 'NC,s'")
        self.similarity = similarity
        if init not in ["LMM", "MM"]:
            raise ValueError("Init must be either 'LMM' or 'MM'")
        self.init = init

    def fit(self, X, bags, proportions):
        """
        Fit the model according to the given training data.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The training input samples.
        bags : array-like, shape = (n_samples,)
            The training bags.
        proportions : array-like, shape = (n_bags,)
            The bags proportions.
        """
        bioc_url = urlopen('https://raw.githubusercontent.com/giorgiop/almostnolabel/master/laplacian.mean.map.R')
        string_lmm = bioc_url.read().decode('utf-8')
        bioc_url = urlopen('https://raw.githubusercontent.com/giorgiop/almostnolabel/master/alternating.mean.map.R')
        string_amm = bioc_url.read().decode('utf-8')
        # Concate the two strings
        string = string_lmm + "\n" + string_amm
        amm = SignatureTranslatedAnonymousPackage(string, "amm")
        self.laplacian = amm.laplacian
        self.alternating_mean_map = amm.alternating_mean_map

        pandas2ri.activate()

        # Creating the R object expected by the R function
        y_proportions = np.array([proportions[bag] for bag in bags])
        trainset = pd.DataFrame(np.concatenate((y_proportions.reshape(-1, 1), bags.reshape(-1, 1), X), axis=1), columns=["label", "bag"] + ["x" + str(i) for i in range(X.shape[1])])
        trainset = trainset.astype({"bag": int})

        N_bags = len(np.unique(bags))

        if self.init == "LMM":
            # Computing the laplacian 
            laplacian = self.laplacian(self.similarity, trainset, N_bags, self.sigma)
        else:
            laplacian = None

        # Calling the R function
        self.w = self.alternating_mean_map(trainset, laplacian, False, self.lmd, self.gamma, rpy2.rinterface.NULL, self.init, 30)
        self.w = robjects.conversion.rpy2py(self.w)
        self.w = np.array(self.w.theta)