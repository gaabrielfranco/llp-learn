import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import sys
from multiprocessing import cpu_count

from llp_learn.base import baseLLPClassifier
from abc import ABC

from ._loaders import LLP_DATASET
import torch.nn.functional as F
from torch.autograd import Variable

np.set_printoptions(threshold=sys.maxsize)


#TODO: change it to a more general class
class LLPGAN_GEN_MNIST(nn.Module):

    def __init__(self, noise_size=100, out_h=28, out_w=28):
        self.out_h, self.out_w = out_h, out_w
        super(LLPGAN_GEN_MNIST, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_size, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500, eps=1e-05, momentum=0.1, ),

            nn.Linear(500, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500, eps=1e-05, momentum=0.1, ),

            nn.Linear(500, self.out_h * self.out_w),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_h * self.out_w, eps=1e-05, momentum=0.1, ),
        )

    def forward(self, noise):
        return self.model(noise).reshape(-1, 1, self.out_h, self.out_w)

#TODO: change it to a more general class
class LLPGAN_GEN_COLOR(nn.Module):
    def __init__(self, noise_size=32*32, channels=3):
        super(LLPGAN_GEN_COLOR, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(noise_size, 4*4*512, bias=False),
            nn.BatchNorm1d(4*4*512),
            nn.ReLU()
        )

        self.trans_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, channels, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.linear(z)
        h = h.view(-1, 512, 4, 4)
        return self.trans_conv(h)
    
# class LLPGAN_DIS(nn.Module):
# 	# use the same discriminator as LLP-GAN paper
# 	def __init__(self, num_class, image_size, in_channel=3, return_features=False):
# 		super(LLPGAN_DIS, self).__init__()
# 		self.conv_layers = nn.Sequential(
# 			nn.Dropout(p=0.2, ),
# 			nn.Conv2d(in_channel, 64, 3, padding=1, stride=1),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, padding=1, stride=1),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, padding=1, stride=2),
# 			nn.ReLU(),
# 			nn.Dropout(p=0.5, ),
# 			nn.Conv2d(64, 128, 3, padding=1, stride=1),
# 			nn.ReLU(),
# 			nn.Conv2d(128, 128, 3, padding=1, stride=1),
# 			nn.ReLU(),
# 			nn.Conv2d(128, 128, 3, padding=1, stride=2),
# 			nn.ReLU(),
# 			nn.Dropout(p=0.5, ),
# 			nn.Conv2d(128, 256, 3, padding=1, stride=1),
# 			nn.ReLU(),
# 			nn.Conv2d(256, 128, 1, padding=0, stride=1),
# 			nn.ReLU(),
# 			nn.Conv2d(128, 64, 1, padding=0, stride=1),
# 			nn.ReLU(),
# 		)
# 		if isinstance(image_size, int):
# 			pool_size = round(round(image_size/2.0)/2.0)
# 		else:
# 			pool_size = (round(round(image_size[0]/2.0)/2.0), round(round(image_size[1]/2.0)/2.0))
# 		self.pool_layer = nn.AvgPool2d(pool_size, stride=pool_size, )
# 		self.fc_layer = nn.Linear(64, num_class, bias=True)
# 		self.return_features = return_features

# 	def forward(self, x):
# 		x = self.conv_layers(x)
# 		features = self.pool_layer(x).reshape(-1, 64)
# 		out = self.fc_layer(features)
# 		if self.return_features:
# 			return out, features
# 		return out

class LLPGAN_DIS(nn.Module):
	# use the same discriminator as LLP-GAN paper
	def __init__(self, num_class, image_size, in_channel=3, return_features=False):
		super(LLPGAN_DIS, self).__init__()
		self.conv_layers = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channel, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 1, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1, padding=0, stride=1),
            nn.ReLU(),
		)
		if isinstance(image_size, int):
			pool_size = round(round(image_size/2.0)/2.0)
		else:
			pool_size = (round(round(image_size[0]/2.0)/2.0), round(round(image_size[1]/2.0)/2.0))
		self.pool_layer = nn.AvgPool2d(pool_size, stride=pool_size, )
		self.fc_layer = nn.Linear(64, num_class, bias=True)
		self.return_features = return_features

	def forward(self, x):
		x = self.conv_layers(x)
		features = self.pool_layer(x).reshape(-1, 64)
		out = self.fc_layer(features)
		if self.return_features:
			return out, features
		return out


class LLPGAN(baseLLPClassifier, ABC):
    """
    LLPGAN implementation
    """
    
    def __init__(self, lr, n_epochs, lambda_, noise_dim, n_jobs=-1, verbose=False, device="auto", random_state=None):
        self.n_epochs = n_epochs
        self.lr = lr
        self.noise_dim = noise_dim
        self.lambda_ = lambda_
        self.optimizer = None
        self.verbose = verbose
        self.random_state = random_state
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
                        
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

        self.seed = random_state if random_state is not None else self.random.randint(2**32-1)
        # Setting the seed for reproducibility in PyTorch
        torch.manual_seed(self.seed)  # fix the initial value of the network weight
        torch.cuda.manual_seed(self.seed)  # for cuda
        torch.cuda.manual_seed_all(self.seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True  # choose the determintic algorithm

    def worker_init_fn(self):
        np.random.seed(self.seed)

    def compute_dis_loss(self, true_images, fake_images, props, lambd=1, epsilon=1e-8):
        # Forward pass
        #TODO: make it more generic
        batch_size, bag_size, channel, height, width = true_images.shape
        true_images = true_images.reshape((batch_size * bag_size, channel, height, width))
        true_outputs, _ = self.dis(true_images)
        fake_outputs, _ = self.dis(fake_images)

        # compute the lower bound of kl
        prob = nn.functional.softmax(true_outputs, dim=-1).reshape((batch_size, bag_size, -1))
        clamped_prob = torch.clamp(prob, epsilon, 1 - epsilon)
        log_prob = torch.log(clamped_prob)
        avg_log_prop = torch.mean(log_prob, dim=1)
        lower_kl_loss = -torch.sum(-props * avg_log_prop, dim=-1).mean() * lambd

        # compute the true/fake binary loss
        true_outputs_cat = torch.cat((true_outputs, torch.zeros(true_outputs.shape[0], 1).to(self.device)), dim=1)
        true_prob = 1 - nn.functional.softmax(true_outputs_cat, dim=1)[:, -1]
        clamped_true_prob = torch.clamp(true_prob, epsilon, 1 - epsilon)
        log_true_prob = torch.log(clamped_true_prob)
        avg_log_true_prop = -torch.mean(log_true_prob)

        fake_outputs_cat = torch.cat((fake_outputs, torch.zeros(fake_outputs.shape[0], 1).to(self.device)), dim=1)
        fake_prob = nn.functional.softmax(fake_outputs_cat, dim=1)[:, -1]
        clamped_fake_prob = torch.clamp(fake_prob, epsilon, 1 - epsilon)
        log_fake_prob = torch.log(clamped_fake_prob)
        avg_log_fake_prop = -torch.mean(log_fake_prob)
        return lower_kl_loss + avg_log_true_prop + avg_log_fake_prop


    def compute_gen_loss(self, true_images, fake_images):
        #TODO: make it more generic
        batch_size, bag_size, channel, height, width = true_images.shape
        true_images = true_images.reshape((batch_size * bag_size, channel, height, width))
        true_outputs, true_features = self.dis(true_images)
        fake_outputs, fake_features = self.dis(fake_images)
        loss = F.mse_loss(fake_features, true_features)
        return loss  # also return feature_maps to compute generator loss
        
    def model_import(self, **kwargs):
        #TODO: implement it properly
        raise NotImplementedError("This method is not implemented yet.")

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

        self.dis.eval()
        with torch.no_grad():
            for batch in test_loader:
                x = batch.to(self.device)
                pred, _ = self.dis(x)
                y_pred += pred.argmax(dim=1).cpu().tolist()

        return np.array(y_pred).reshape(-1)

    def fit(self, X, bags, proportions):

        torch.cuda.empty_cache()

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

        # Create model TODO
        #self.gen, self.dis = self.model_import()
        self.gen = LLPGAN_GEN_COLOR(self.noise_dim, X.shape[1]).to(self.device)
        # from torchvision.models import resnet18
        # self.dis = resnet18(weights="IMAGENET1K_V1")
        # channels = X.shape[1]
        # if self.dis:
        #     if channels != 3:
        #         self.dis.conv1 = nn.Conv2d(
        #             channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        #         )
        #     self.dis.fc = nn.Linear(self.dis.fc.in_features, classes)
        #     self.dis = self.dis.to(self.device)

        self.dis = LLPGAN_DIS(classes, X.shape[2:], X.shape[1], return_features=True).to(self.device)
        
        # TODO: check which one the paper used and change it here
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=self.lr)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr)

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

        self.gen.train()
        self.dis.train()
        
        with tqdm(range(self.n_epochs), desc='Training model', unit='epoch', disable=not self.verbose) as t:
            for epoch in t:
                gen_losses = []
                dis_losses = []
                for i, (batch, props) in enumerate(tqdm(train_loader, leave=False)):
                    batch = batch.to(self.device)
                    props = props.to(self.device)

                    batch_data_points = batch.shape[0] * batch.shape[1]
                    noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_data_points, self.noise_dim))).to(self.device))

                    self.dis_opt.zero_grad()
                    fake_batch = self.gen(noise).detach()
                    fake_batch = torch.tensor(fake_batch).float()
                    fake_batch = fake_batch.to(self.device)
                    dis_loss = self.compute_dis_loss(batch, fake_batch, props, lambd=self.lambda_)
                    dis_loss.backward()
                    self.dis_opt.step()

                    self.gen_opt.zero_grad()
                    gen_loss = self.compute_gen_loss(batch, self.gen(noise))
                    gen_loss.backward()
                    self.gen_opt.step()

                    gen_losses.append(gen_loss.item())
                    dis_losses.append(dis_loss.item())
                
                gen_loss = np.mean(gen_losses)
                dis_loss = np.mean(dis_losses)
                print("[Epoch: %d/%d] train GEN loss: %.4f, train DIS loss: %.4f" % (epoch + 1, self.n_epochs, gen_loss, dis_loss))