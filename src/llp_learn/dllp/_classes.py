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
import torch.nn.functional as F

np.set_printoptions(threshold=sys.maxsize)

class KLLossLLP(nn.Module):
    def __init__(self):
        """
            KL LLP loss
        """
        super().__init__()

    def forward(self, pred, target):
        # Take the log softmax of the model output
        pred = F.log_softmax(pred, dim=1)
        # Batch averager
        batch_avg = torch.mean(pred, dim=0)
        # KL divergence loss
        loss = F.kl_div(batch_avg, target, reduction="batchmean")
        return loss

# Source: https://github.com/lucastassis/dllp/blob/main/net.py
class SimpleMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_layer_sizes=(100,)):
        super(SimpleMLP, self).__init__()  
        self.layers = nn.ModuleList() 
        for size in hidden_layer_sizes:
            self.layers.append(nn.Linear(in_features, size))
            self.layers.append(nn.Dropout(0.5))
            self.layers.append(nn.ReLU())
            in_features = size
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], out_features))

    def forward(self, x): 
        for layer in self.layers:
            x = layer(x)
        return x

class DLLP(baseLLPClassifier, ABC):
    def __init__(self, lr, n_epochs, model_type, pretrained=True, hidden_layer_sizes=(100,), verbose=False, device="auto", n_jobs=-1, random_state=None):
        self.n_epochs = n_epochs
        self.lr = lr
        self.hidden_layer_sizes = hidden_layer_sizes
        self.loss_train = KLLossLLP()
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
        self.seed = random_state if random_state is not None else np.random.randint(2**32-1)
        # Setting the seed for reproducibility in PyTorch
        torch.manual_seed(self.seed)  # fix the initial value of the network weight
        torch.cuda.manual_seed(self.seed)  # for cuda
        torch.cuda.manual_seed_all(self.seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True  # choose the determintic algorithm

    def model_import(self, model_type, classes, **kwargs):
        if model_type == "resnet18":
            try:
                pretrained = kwargs["pretrained"]
                channels = kwargs["channels"]
            except KeyError:
                raise NameError("Missing parameters for the model.")

            model = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            if model:
                if channels != 3:
                    model.conv1 = nn.Conv2d(
                        channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                    )
                model.fc = nn.Linear(model.fc.in_features, classes)
                model = model.to(self.device)
        elif model_type == "simple-mlp":
            try:
                in_features = kwargs["in_features"]
                hidden_layer_sizes = kwargs["hidden_layer_sizes"]
            except KeyError:
                raise NameError("Missing parameters for the model.")
            model = SimpleMLP(in_features, classes, hidden_layer_sizes=hidden_layer_sizes)
            model = model.to(self.device)
        return model

    def worker_init_fn(self):
        np.random.seed(self.seed)

    def set_params(self, **params):
        self.__dict__.update(params)

    def get_params(self):
        return self.__dict__

    def predict(self, X, batch_size=512):
        test_loader = torch.utils.data.DataLoader(
            X,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.n_jobs,
        )
        y_pred = []

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x = batch.to(self.device)
                pred = self.model(x)
                y_pred += pred.argmax(dim=1).cpu().tolist()

        return np.array(y_pred).reshape(-1)

    def fit(self, X, bags, proportions):
        if len(proportions.shape) == 1:
            classes = 2
        else:
            classes = proportions.shape[1]

        # We will have to convert the proportions array to a 2D array
        if classes == 2:
            proportions = np.array([1 - proportions, proportions]).T

        # Create model
        if self.model_type == "resnet18":
            self.model = self.model_import(self.model_type, classes, pretrained=self.pretrained, channels=X.shape[1]) # We expect the channels to be the first dimension
        elif self.model_type == "simple-mlp":
            self.model = self.model_import(self.model_type, classes, in_features=X.shape[1], hidden_layer_sizes=self.hidden_layer_sizes)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Create dataset and dataloader
        unique_bags = sorted(np.unique(bags))

        # We have to map the unique bags to the range of 0 to len(unique_bags)-1
        bags = np.array([unique_bags.index(bag) for bag in bags])

        # Keep the proportions of the unique bags
        proportions = proportions[unique_bags]

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
                    outputs = self.model(X)
                    # compute loss and backprop
                    loss = self.loss_train(outputs, bag_prop)
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
                train_loss = np.mean(losses)
                print("[Epoch: %d/%d] train loss: %.4f" % (i + 1, self.n_epochs, train_loss))
        
