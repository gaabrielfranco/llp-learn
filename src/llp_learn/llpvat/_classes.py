import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import sys
from multiprocessing import cpu_count

from llp_learn.base import baseLLPClassifier
from abc import ABC

from ._vat_loss import VATLoss
from ._loaders import LLP_DATASET
import torch.nn.functional as F
from torchvision.models import resnet18

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

class LLPVAT(baseLLPClassifier, ABC):
    """
    LLPVAT implementation
    """
    
    def __init__(self, lr, n_epochs, xi, eps, ip, model_type="resnet18", device="auto", pretrained=True, hidden_layer_sizes=(100,), verbose=False, n_jobs=-1, random_state=None):
        self.n_epochs = n_epochs
        self.lr = lr
        self.model = None
        self.optimizer = None
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.verbose = verbose
        self.random_state = random_state
        self.pretrained = pretrained
        self.hidden_layer_sizes = hidden_layer_sizes
        self.model_type = model_type
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.loss_train = VATLoss(xi=xi, eps=eps, ip=ip).to(self.device)
        
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

    def set_params(self, **params):
        self.__dict__.update(params)

    def get_params(self):
        return self.__dict__

    def compute_kl_loss_on_bagbatch(self, images, props, epsilon=1e-8):
        # Forward pass
        data_info = tuple(images.size())
        batch_size = data_info[0] # batch_size (number of bags in the batch)
        bag_size = data_info[1] # bag size
        shape = data_info[2:] # shape of the data
        shape = tuple([batch_size * bag_size] + list(shape))
        images = images.reshape(shape)
        outputs = self.model(images)
        prob = nn.functional.softmax(outputs, dim=-1).reshape((batch_size, bag_size, -1))
        avg_prob = torch.mean(prob, dim=1)
        avg_prob = torch.clamp(avg_prob, epsilon, 1 - epsilon)
        loss = torch.sum(-props * torch.log(avg_prob), dim=-1).mean()
        return loss
    
    def sigmoid_rampup(self, current, rampup_length):
        # modified from https://github.com/kevinorjohn/LLP-VAT/blob/a111d6785e8b0b79761c4d68c5b96288048594d6/llp_vat/
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))


    def get_rampup_weight(self, weight, iteration, rampup):
        # modified from https://github.com/kevinorjohn/LLP-VAT/blob/a111d6785e8b0b79761c4d68c5b96288048594d6/llp_vat/
        alpha = weight * self.sigmoid_rampup(iteration, rampup)
        return alpha

    def llp_loss_f(self, images, props, iteration):
        prop_loss = self.compute_kl_loss_on_bagbatch(images, props)
        alpha = self.get_rampup_weight(0.05, iteration, -1)  # hard-coded based on tsai and lin's implementation
        shape = tuple(images.size())
        shape = shape[2:] # shape of the data
        shape = tuple([-1] + list(shape))
        vat_loss = self.loss_train(self.model, images.reshape(shape))
        return prop_loss, alpha, vat_loss

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

        # Batch size - bags as batches
        batch_size = 1

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
        for bag in unique_bags:
            bag2indices[bag] = np.where(bags == bag)[0]
            bag2prop[bag] = proportions[bag].astype(np.float32)

        # Create datasets
        train_dataset = LLP_DATASET(
            data=X,
            bag2indices=bag2indices,
            bag2prop=bag2prop,
            transform=None,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=self.worker_init_fn(),
            num_workers=self.n_jobs,
        )

        self.model.train()
        
        with tqdm(range(self.n_epochs), desc='Training model', unit='epoch', disable=not self.verbose) as t:
            for epoch in t:
                losses = []
                for i, (batch, props) in enumerate(tqdm(train_loader, leave=False)):
                    batch = batch.to(self.device)
                    props = props.to(self.device)

                    prop_loss, alpha, vat_loss = self.llp_loss_f(batch, props, i)
                    loss = prop_loss + alpha * vat_loss

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
                train_loss = np.array(losses).mean()
                print("[Epoch: %d/%d] train loss: %.4f" % (epoch + 1, self.n_epochs, train_loss))