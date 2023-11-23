from PIL import Image
import torch
import numpy as np

def truncate_data_group(x, y, instance2group):
	idx_list = []
	for i in range(x.shape[0]):
		if instance2group[i] != -1:
			idx_list.append(i)
	x_truncated = x[idx_list]
	y_truncated = y[idx_list]
	idx2new = {idx_list[i]: i for i in range(len(idx_list))}
	instance2group_new = {}
	for old, new in idx2new.items():
		instance2group_new[new] = instance2group[old]
	new2idx = {idx2new[idx]: idx for idx in idx2new.keys()}
	return x_truncated, y_truncated, instance2group_new, new2idx

class LLPFC_DATASET_BASE(torch.utils.data.Dataset):
	def __init__(self, data, noisy_y, group2transition, instance2weight, instance2group, transform):
		self.data, self.noisy_y, self.instance2group, self.new2idx = truncate_data_group(data, noisy_y, instance2group)
		self.group2transition = group2transition
		self.instance2weight = instance2weight
		self.transform = transform

	def __len__(self):
		return len(self.data)
    
class LLPFC_DATASET(LLPFC_DATASET_BASE):
	def __getitem__(self, index):
		trans_m = self.group2transition[self.instance2group[index]]
		weight = self.instance2weight[self.new2idx[index]]
		return self.data[index], int(self.noisy_y[index]), torch.tensor(trans_m, dtype=None), weight


# class KL_CIFAR10(KL_DATASET_BASE):
#     def __getitem__(self, bag_index):
#         indices = self.bag2indices[bag_index]
#         images = torch.zeros((len(indices), self.data[0].shape[2], self.data[0].shape[0], self.data[0].shape[1]),
#                              dtype=torch.float32)
#         for i in range(len(indices)):
#             idx = indices[i]
#             img = self.data[idx]
#             img = Image.fromarray(img)
#             if self.transform is not None:
#                 img = self.transform(img)
#             images[i] = img
#         return images, self.bag2prop[bag_index]


# class KL_SVHN(KL_DATASET_BASE):
#     def __getitem__(self, bag_index):
#         indices = self.bag2indices[bag_index]
#         images = torch.zeros((len(indices), self.data[0].shape[0], self.data[0].shape[1], self.data[0].shape[2],),
#                              dtype=torch.float32)
#         for i in range(len(indices)):
#             idx = indices[i]
#             img = self.data[idx]
#             img = Image.fromarray(np.transpose(img, (1, 2, 0)))
#             if self.transform is not None:
#                 img = self.transform(img)
#             images[i] = img
#         return images, self.bag2prop[bag_index]


# class KL_EMNIST(KL_DATASET_BASE):
#     def __init__(self, data, bag2indices, bag2prop, transform):
#         super(KL_EMNIST, self).__init__(data, bag2indices, bag2prop, transform)
#         img = self.transform(Image.fromarray(self.data[0].numpy(), mode='L'))
#         self.new_h = img.shape[1]
#         self.new_w = img.shape[2]  # need this for resized emnist

#     def __getitem__(self, bag_index):
#         indices = self.bag2indices[bag_index]
#         images = torch.zeros((len(indices), 1, self.new_h, self.new_w,), dtype=torch.float32)
#         for i in range(len(indices)):
#             idx = indices[i]
#             img = self.data[idx]
#             img = Image.fromarray(img.numpy(), mode='L')
#             if self.transform is not None:
#                 img = self.transform(img)
#             images[i] = img
#         return images, self.bag2prop[bag_index]
