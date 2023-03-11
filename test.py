from copy import deepcopy
import sys
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs, make_circles, make_moons
sys.path.append('src')

from llp.mm import MM, LMM, AMM
from llp.dllp import DLLP
from llp.model_selection import gridSearchCV

random = np.random.RandomState(42)

# Creating a syntetic dataset using sklearn
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_samples=1000, random_state=42)
bags = random.randint(0, 5, size=X.shape[0])

# Creating the proportions
proportions = np.zeros(5)
for i in range(5):
    bag_i = np.where(bags == i)[0]
    proportions[i] = y[bag_i].sum() / len(bag_i)

# y[y == 0] = -1 

#mm = LMM(lmd=1, gamma=1, sigma=1)
#mm = AMM(lmd=1, gamma=1, sigma=1)
#gs = gridSearchCV(mm, param_grid={"lmd": [0.1, 1], "gamma": [0.1], "sigma": [0.1]}, cv=5, n_jobs=-1, random_state=42)
#mm = MM(lmd=1)
mm = DLLP(lr=0.001, n_epochs=100)
gs = gridSearchCV(mm, param_grid={"lr": [1e-5, 1e-3, 1e-1]}, cv=5, n_jobs=-1, random_state=42)


# Train/test split
train_idx = random.choice(np.arange(X.shape[0]), size=int(X.shape[0] * 0.8), replace=False)
test_idx = np.setdiff1d(np.arange(X.shape[0]), train_idx)

gs.fit(X[train_idx], bags[train_idx], proportions)

print(classification_report(y[test_idx], gs.predict(X[test_idx])))