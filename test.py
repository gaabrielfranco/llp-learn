from copy import deepcopy
import sys
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs, make_circles, make_moons

from llp_learn.mm import MM, LMM, AMM
from llp_learn.dllp import DLLP
from llp_learn.model_selection import gridSearchCV, SplitBagShuffleSplit, SplitBagBootstrapSplit

random = np.random.RandomState(42)

# Creating a syntetic dataset using sklearn
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_samples=1000, random_state=42)
bags = random.randint(0, 5, size=X.shape[0])

# Creating the proportions
proportions = np.zeros(5)
for i in range(5):
    bag_i = np.where(bags == i)[0]
    proportions[i] = y[bag_i].sum() / len(bag_i)
    
#y[y == 0] = -1

#mm = LMM(lmd=1, gamma=1, sigma=1)
#mm = AMM(lmd=1, gamma=1, sigma=1)
#gs = gridSearchCV(mm, param_grid={"lmd": [0.1, 1], "gamma": [0.1], "sigma": [0.1]}, cv=5, n_jobs=-1, random_state=42)
#mm = MM(lmd=100)
mm = DLLP(lr=0.0001, n_epochs=1000, hidden_layer_sizes=(100, 100))
gs = gridSearchCV(mm, param_grid={"lr": [0.1, 0.01, 0.001, 0.0001]}, cv=5, validation_size=0.5, n_jobs=-1, random_state=42)


# Train/test split
train_idx = random.choice(np.arange(X.shape[0]), size=int(X.shape[0] * 0.8), replace=False)
test_idx = np.setdiff1d(np.arange(X.shape[0]), train_idx)

gs.fit(X[train_idx], bags[train_idx], proportions)

print(classification_report(y[test_idx], gs.predict(X[test_idx])))