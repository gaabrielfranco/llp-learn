import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import sys
from multiprocessing import cpu_count

from llp_learn.base import baseLLPClassifier
from abc import ABC

from ._losses import ConfidentialIntervalLoss, PiModelLoss, VATLoss, consistency_loss_function
from ._loaders import Dataset_Mixbag
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

class MixBag(baseLLPClassifier, ABC):
    """
    MixBag implementation based on the original implementation from https://github.com/asanomitakanori/Mixbag.
    """
    
    def __init__(self, lr, n_epochs, consistency, choice, confidence_interval, model_type="resnet18", device="auto", pretrained=True, hidden_layer_sizes=(100,), verbose=False, n_jobs=-1, random_state=None):
        self.n_epochs = n_epochs
        self.lr = lr
        self.model = None
        self.optimizer = None
        self.loss_train = ConfidentialIntervalLoss()
        self.verbose = verbose
        self.random_state = random_state
        self.choice = choice
        self.confidence_interval = confidence_interval
        self.pretrained = pretrained
        self.hidden_layer_sizes = hidden_layer_sizes
        self.model_type = model_type
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
        
        if model_type != "resnet18" and model_type != "simple-mlp":
            raise NameError("Unknown model type.")
        
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

        self.seed = random_state if random_state is not None else self.random.randint(2**32-1)
        # Setting the seed for reproducibility in PyTorch
        torch.manual_seed(self.seed)  # fix the initial value of the network weight
        torch.cuda.manual_seed(self.seed)  # for cuda
        torch.cuda.manual_seed_all(self.seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True  # choose the determintic algorithm

        # Consistency loss
        self.consistency = consistency
        if consistency == "none":
            self.consistency_criterion = None
        elif consistency == "vat":
            self.consistency_criterion = VATLoss()
        elif consistency == "pi":
            self.consistency_criterion = PiModelLoss()
        else:
            raise NameError("Unknown consistency criterion")

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
        if "consistency" in params:
            if params['consistency'] == "none":
                self.consistency_criterion = None
            elif params['consistency'] == "vat":
                self.consistency_criterion = VATLoss()
            elif params['consistency'] == "pi":
                self.consistency_criterion = PiModelLoss()
            else:
                raise NameError("Unknown consistency criterion")

    def get_params(self):
        return self.__dict__

    def calculate_prop(self, output, nb, bs):
        output = F.softmax(output, dim=1)
        output = output.reshape(nb, bs, -1)
        lp_pred = output.mean(dim=1)
        return lp_pred

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

        train_bags = []
        for bag in unique_bags:
            bag_i = np.where(bags == bag)[0]
            train_bags.append(X[bag_i, :])

        # Create loaders
        train_dataset = Dataset_Mixbag(
            data=train_bags,
            lp=proportions[unique_bags],
            choice=self.choice,
            confidence_interval=self.confidence_interval,
            random_state=self.random_state
        )

        # TODO: check the shuffle parameter
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=self.worker_init_fn(),
            num_workers=self.n_jobs,
        )

        self.model.train()

        best_loss = np.inf
        
        with tqdm(range(self.n_epochs), desc='Training model', unit='epoch', disable=not self.verbose) as t:
            for epoch in t:
                losses = []
                for batch in tqdm(train_loader, leave=False):
                    data_info = tuple(batch["data"].size())
                    nb = data_info[0] # number of bags
                    bs = data_info[1] # bag size
                    shape = data_info[2:] # shape of the data
                    shape = tuple([-1] + list(shape))

                    data = batch["data"].reshape(shape).to(self.device)
                    lp_gt = batch["label_prop"].to(self.device)
                    ci_min_value, ci_max_value = batch["ci_min_value"], batch["ci_max_value"]

                    # Consistency loss
                    consistency_loss = consistency_loss_function(
                        self.consistency,
                        self.consistency_criterion,
                        self.model,
                        train_loader,
                        data,
                        epoch,
                        batch_size,
                    )

                    output = self.model(data)
                    lp_pred = self.calculate_prop(output, nb, bs)

                    loss = self.loss_train(
                        lp_pred,
                        lp_gt,
                        ci_min_value.to(self.device),
                        ci_max_value.to(self.device),
                    )

                    loss += consistency_loss
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    losses.append(loss.item())
                train_loss = np.array(losses).mean()
                print("[Epoch: %d/%d] train loss: %.4f" % (epoch + 1, self.n_epochs, train_loss))

        # TODO: how can we improve this? The filename must be unique, otherwise it will overwrite the previous one. Take this decision.
        #         if train_loss < best_loss:
        #             torch.save(self.model.state_dict(), "Best_CP.pkl")
        #             best_loss = train_loss

        # # Using the best model
        # self.model.load_state_dict(torch.load("Best_CP.pkl", map_location=self.device))
