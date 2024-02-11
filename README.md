# llp-learn

[![PyPI - Version](https://img.shields.io/pypi/v/llp-learn.svg)](https://pypi.org/project/llp-learn)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/llp-learn.svg)](https://pypi.org/project/llp-learn)

LLP-learn is a library that provides implementation of methods for Learning from Label Proportions.

-----

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

```console
pip install llp-learn
```

## Usage
```py
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from llp_learn.dllp import DLLP
from llp_learn.model_selection import gridSearchCV

random = np.random.RandomState(42)

# Creating a syntetic dataset using sklearn
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_samples=1000, random_state=42)

# Generating 5 bags randomly
bags = random.randint(0, 5, size=X.shape[0])

# Creating the proportions
proportions = np.zeros(5)
for i in range(5):
    bag_i = np.where(bags == i)[0]
    proportions[i] = y[bag_i].sum() / len(bag_i)

# LLP model (DLLP)
llp_model = DLLP(lr=0.0001, n_epochs=1000, hidden_layer_sizes=(100, 100))

# Grid Search the lr parameter
gs = gridSearchCV(llp_model, param_grid={"lr": [0.1, 0.01, 0.001, 0.0001]}, cv=5, validation_size=0.5, n_jobs=-1, random_state=42)

# Train/test split
train_idx = random.choice(np.arange(X.shape[0]), size=int(X.shape[0] * 0.8), replace=False)
test_idx = np.setdiff1d(np.arange(X.shape[0]), train_idx)

# Fitting the model
gs.fit(X[train_idx], bags[train_idx], proportions)

# Predicting the labels of the test set
y_pred_test = gs.predict(X[test_idx])

# Reporting the performance of the model in the test set
print(classification_report(y[test_idx], y_pred_test))
```

## License

`llp-learn` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
