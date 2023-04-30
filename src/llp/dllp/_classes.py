import numpy as np
from copy import copy, deepcopy
import torch
from torch import nn
from tqdm import tqdm
from sklearn.utils import shuffle
import sys

from llp.base import baseLLPClassifier
from abc import ABC, abstractmethod

np.set_printoptions(threshold=sys.maxsize)

class BatchAvgLayer(nn.Module):
    def __init__(self):
        super(BatchAvgLayer, self).__init__()

    def forward(self, x):
        return torch.mean(input=x, dim=0).repeat(x.shape[0], 1)

class MLPBatchAvg(nn.Module):
    def __init__(self, in_features, out_features, hidden_layer_sizes=(100,)):
        super(MLPBatchAvg, self).__init__()  
        self.layers = nn.ModuleList() 
        for size in hidden_layer_sizes:
            self.layers.append(nn.Linear(in_features, size))
            self.layers.append(nn.ReLU())
            in_features = size
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], out_features))
        self.layers.append(nn.LogSoftmax(dim=1))
        self.batch_avg = BatchAvgLayer()

    def forward(self, x): 
        for layer in self.layers:
            x = layer(x)
        prop = x.clone()
        if self.training:
            x = self.batch_avg(x)

        return x, prop
    
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

class DLLP(baseLLPClassifier, ABC):
    
    def __init__(self, lr, n_epochs, in_features=2, out_features=2, hidden_layer_sizes=(100,), verbose=False, random_state=None):
        self.n_epochs = n_epochs
        self.model = MLPBatchAvg(in_features=in_features, out_features=out_features, hidden_layer_sizes=hidden_layer_sizes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_batch = torch.nn.KLDivLoss(reduction='batchmean')
        self.verbose = verbose
        self.random_state = random_state

    def set_params(self, **params):
        self.model.set_params(**params)
        if 'lr' in params:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'])

    def get_params(self):
        return self.__dict__

    def predict(self, X):
        x_test = torch.FloatTensor(X)
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(x_test)

        return outputs.argmax(dim=1).numpy()

    def fit(self, X, bags, proportions):
        unique_bags = np.unique(bags)
        y_proportions = np.array([proportions[bags[i]] for i in range(len(bags))])

        self.model.train()
        loss_epoch = []
        with tqdm(range(self.n_epochs), desc='Training model', unit='epoch', disable=not self.verbose) as t:
            for _ in t:
                shuffled_bags = shuffle(unique_bags, random_state=self.random_state)
                loss_sum = []
                for bag in shuffled_bags:
                    # prepare bag data
                    bag_idx = np.where(bags == bag)[0]
                    x_train = torch.FloatTensor(X[bag_idx])
                    y_train = torch.FloatTensor(y_proportions[bag_idx])
                    
                    # compute outputs
                    outputs, _ = self.model(x_train)                
                    
                    # compute loss and backprop
                    self.optimizer.zero_grad()
                    loss = self.loss_batch(outputs[0], torch.Tensor([(1 - y_train[0]), y_train[0]]))
                    loss_sum.append(loss.item())
                    loss.backward()
                    self.optimizer.step()                
                    t.set_postfix(loss=np.round(np.mean(loss_sum), 4))
                loss_epoch.append(np.mean(loss_sum))