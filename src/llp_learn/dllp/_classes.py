from multiprocessing import cpu_count
import numpy as np
from copy import copy, deepcopy
import torch
from torch import nn
from tqdm import tqdm
from sklearn.utils import shuffle
import sys

from llp_learn.base import baseLLPClassifier
from abc import ABC
from torchvision.models import resnet18

from ._loaders import LLPDataset

np.set_printoptions(threshold=sys.maxsize)

# Source: https://github.com/lucastassis/dllp/blob/main/net.py
class BatchAvgLayer(nn.Module):
    def __init__(self):
        super(BatchAvgLayer, self).__init__()

    def forward(self, x):
        return torch.mean(input=x, dim=0)

class MLPBatchAvg(nn.Module):
    def __init__(self, model_type, in_features=2, out_features=2, hidden_layer_sizes=(100,), channels=3, classes=2, pretrained=True):
        super(MLPBatchAvg, self).__init__()
        if model_type == "simple-mlp":
            self.layers = nn.ModuleList() 
            for size in hidden_layer_sizes:
                self.layers.append(nn.Linear(in_features, size))
                self.layers.append(nn.ReLU())
                in_features = size
            self.layers.append(nn.Linear(hidden_layer_sizes[-1], out_features))
            self.layers.append(nn.LogSoftmax(dim=1))
            self.batch_avg = BatchAvgLayer()
        elif model_type == "resnet18":
            model = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            if model:
                if channels != 3:
                    model.conv1 = nn.Conv2d(
                        channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                    )
                model.fc = nn.Linear(model.fc.in_features, classes)
            self.layers = nn.ModuleList(list(model.children()))
            self.batch_avg = BatchAvgLayer()

    def forward(self, x): 
        for layer in self.layers:
            x = layer(x)
        softmax = x.clone()
        if self.training:
            x = self.batch_avg(x)
        return x, softmax
    
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

class DLLP(baseLLPClassifier, ABC):
    def __init__(self, lr, n_epochs, model_type, pretrained=True, hidden_layer_sizes=(100,), verbose=False, device="auto", n_jobs=-1, random_state=None):
        self.n_epochs = n_epochs
        self.lr = lr
        self.hidden_layer_sizes = hidden_layer_sizes
        self.loss_batch = torch.nn.KLDivLoss(reduction='batchmean')
        self.verbose = verbose
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if model_type != "resnet18" and model_type != "simple-mlp":
            raise NameError("Unknown model type.")
        self.model_type = model_type
        self.model = None
        self.optimizer = None
        self.pretrained = pretrained
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.seed = random_state if random_state is not None else self.random.randint(2**32-1)
        # Setting the seed for reproducibility in PyTorch
        torch.manual_seed(self.seed)  # fix the initial value of the network weight
        torch.cuda.manual_seed(self.seed)  # for cuda
        torch.cuda.manual_seed_all(self.seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True  # choose the determintic algorithm

    def worker_init_fn(self):
        np.random.seed(self.seed)

    def set_params(self, **params):
        self.model.set_params(**params)

    def get_params(self):
        return self.__dict__

    def predict(self, X):
        with torch.no_grad():
            X = torch.tensor(X).to(self.device)
            _, outputs = self.model(X)
            y_pred = outputs.argmax(dim=1).cpu().tolist()

        return y_pred

    def fit(self, X, bags, proportions):
        if len(proportions.shape) == 1:
            classes = 2
        else:
            classes = proportions.shape[1]

        # We will have to convert the proportions array to a 2D array
        if classes == 2:
            proportions = np.array([1 - proportions, proportions]).T

        if self.model_type == "resnet18":
            self.model = MLPBatchAvg(model_type=self.model_type, channels=X.shape[1], classes=classes, pretrained=self.pretrained)
        elif self.model_type == "simple-mlp":
            self.model = MLPBatchAvg(model_type=self.model_type, in_features=X.shape[1], out_features=classes, hidden_layer_sizes=self.hidden_layer_sizes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        train_dataset = LLPDataset(X=X, bags=bags, proportions=proportions)
        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            worker_init_fn=self.worker_init_fn(),
            num_workers=self.n_jobs,
        )

        self.model.train()
        with tqdm(range(self.n_epochs), desc='Training model', unit='epoch') as tepoch:
            for i in tepoch:
                losses = []
                for X, bag_prop in data_loader:
                    # prepare bag data
                    X, bag_prop = X[0].to(self.device), bag_prop[0].to(self.device)
                    # compute outputs
                    batch_avg, outputs = self.model(X) 
                    # compute loss and backprop
                    self.optimizer.zero_grad()
                    loss = self.loss_batch(batch_avg, bag_prop)
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
                train_loss = np.mean(losses)
                print("[Epoch: %d/%d] train loss: %.4f" % (i + 1, self.n_epochs, train_loss))
        
