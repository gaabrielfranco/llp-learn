import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import sys
from multiprocessing import cpu_count

from llp_learn.base import baseLLPClassifier
from abc import ABC

from ._loaders import LLPFC_DATASET
from ._make_groups import make_groups_forward

import torch.nn.functional as F
from torchvision.models import resnet18
from torch.distributions.constraints import simplex
from torch.utils.data import SubsetRandomSampler

np.set_printoptions(threshold=sys.maxsize)

class SimpleMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_layer_sizes=(100,)):
        super(SimpleMLP, self).__init__()  
        self.layers = nn.ModuleList() 
        for size in hidden_layer_sizes:
            self.layers.append(nn.Linear(in_features, size))
            self.layers.append(nn.ReLU())
            in_features = size
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], out_features))

    def forward(self, x): 
        for layer in self.layers:
            x = layer(x)
        return x

class LLPFC(baseLLPClassifier, ABC):
    """
    LLPFC implementation
    """
    
    def __init__(self, lr, n_epochs, model_type="resnet18", device="auto", pretrained=True, hidden_layer_sizes=(100,), verbose=False, n_jobs=-1, random_state=None):
        self.n_epochs = n_epochs
        self.lr = lr
        self.model = None
        self.optimizer = None
        self.verbose = verbose
        self.random_state = random_state
        self.pretrained = pretrained
        self.hidden_layer_sizes = hidden_layer_sizes
        self.model_type = model_type
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
                
        if model_type != "resnet18" and model_type != "simple-mlp":
            raise NameError("Unknown model type.")
        
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

        self.seed = random_state if random_state is not None else self.random.randint(2**32-1)
        # Setting the seed for reproducibility in PyTorch
        torch.manual_seed(self.seed)  # fix the initial value of the network weight
        torch.cuda.manual_seed(self.seed)  # for cuda
        torch.cuda.manual_seed_all(self.seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True  # choose the determintic algorithm

    def worker_init_fn(self):
        np.random.seed(self.seed)
        
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
    
    def loss_train(self, x, y, weights, epsilon=1e-8):
        assert torch.all(simplex.check(x))
        x = torch.clamp(x, epsilon, 1 - epsilon)
        unweighted = nn.functional.nll_loss(torch.log(x), y, reduction='none')
        weights /= weights.sum()
        return (unweighted * weights).sum()

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
        # Get the number of classes
        if len(proportions.shape) == 1:
            classes = 2
        else:
            classes = proportions.shape[1]

        # Since they are using softmax for binary, we will have to convert the proportions array to a 2D array
        if classes == 2:
            proportions = np.array([1 - proportions, proportions]).T

        # Batch size 
        batch_size = 64 #TODO: change this to a parameter

        # Create model
        if self.model_type == "resnet18":
            self.model = self.model_import(self.model_type, classes, pretrained=self.pretrained, channels=X.shape[1]) # We expect the channels to be the first dimension
        elif self.model_type == "simple-mlp":
            self.model = self.model_import(self.model_type, classes, in_features=X.shape[1], hidden_layer_sizes=self.hidden_layer_sizes)
         
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        unique_bags = sorted(np.unique(bags))

        # Create loaders
        bag2indices = {}
        bag2prop = {}
        bag2size = {}
        for bag in unique_bags:
            bag2indices[bag] = np.where(bags == bag)[0]
            bag2prop[bag] = proportions[bag].astype(np.float32)
            bag2size[bag] = len(bag2indices[bag])

        # TODO: transform them to parameters
        self.noisy_prior_choice = "approx"
        self.weights = "uniform"
        self.num_epoch_regroup = 20

        self.model.train()
        
        with tqdm(range(self.n_epochs), desc='Training model', unit='epoch', disable=not self.verbose) as t:
            for epoch in t:
                losses = []
                if epoch % self.num_epoch_regroup == 0:
                    instance2group, group2transition, instance2weight, noisy_y = make_groups_forward(classes, bag2indices, bag2size, bag2prop, self.noisy_prior_choice, self.weights)

                    # Create datasets
                    train_dataset = LLPFC_DATASET(
                        data=X,
                        noisy_y=noisy_y, 
                        group2transition=group2transition, 
                        instance2weight=instance2weight, 
                        instance2group=instance2group,
                        transform=None,
                    )

                    train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        worker_init_fn=self.worker_init_fn(),
                        num_workers=self.n_jobs,
                    )

                for i, (batch, noisy_y, trans_m, weights) in enumerate(tqdm(train_loader, leave=False)):
                    batch = batch.to(self.device)
                    noisy_y = noisy_y.to(self.device)
                    trans_m = trans_m.to(self.device)
                    weights = weights.to(self.device)

                    outputs = self.model(batch)
                    prob = nn.functional.softmax(outputs, dim=1)
                    prob_corrected = torch.bmm(trans_m.float(), prob.reshape(prob.shape[0], -1, 1)).reshape(prob.shape[0], -1)

                    loss = self.loss_train(prob_corrected, noisy_y, weights)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
                train_loss = np.array(losses).mean()
                print("[Epoch: %d/%d] train loss: %.4f" % (epoch + 1, self.n_epochs, train_loss))