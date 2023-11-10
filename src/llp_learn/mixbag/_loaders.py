import numpy as np
import torch
from scipy.stats import norm

class Dataset_Base(torch.utils.data.Dataset):
    """
        Base class for Dataset loader. 
        Different from the original implementation, the normalization step is not included in the Dataset class.
        In the original implementation, they also have the label associated to each instance. We removed it because in LLP we do not have access to the labels.
    """

    def __init__(self, data):
        self.data = data
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """Overload this function in your Dataset."""
        raise NotImplementedError

class Dataset_Mixbag(Dataset_Base):
    """
    Training Dataset. This is a MixBag dataloader,
    so we can use MixBag by applying this dataloader.
    CI loss is not applied in this module.
    The original implementation did not support bags with different sizes. We implemented the support for it following the paper definition.
    Also, the original implementation expects the data to be always an image. We removed this constraint in this implementation.
    """

    def __init__(self, data, lp, choice, confidence_interval, random_state=None):
        super().__init__(data)
        self.lp = lp
        self.classes = lp.shape[1]
        self.choice = choice
        self.CI = confidence_interval
        self.random = np.random.RandomState(random_state)
        seed = random_state if random_state is not None else self.random.randint(2**32-1)
        # Setting the seed for reproducibility in PyTorch
        torch.manual_seed(seed)  # fix the initial value of the network weight
        torch.cuda.manual_seed(seed)  # for cuda
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True  # choose the determintic algorithm


    def ci_loss_interval(self, proportion1: list, proportion2: list, sampling_num1: int, sampling_num2: int, confidence_interval: float):
        a: float = sampling_num1 / (sampling_num1 + sampling_num2)
        b: float = sampling_num2 / (sampling_num1 + sampling_num2)
        t = norm.isf(q=confidence_interval)
        cover1 = t * np.sqrt(proportion1 * (1 - proportion1) / sampling_num1)
        cover2 = t * np.sqrt(proportion2 * (1 - proportion2) / sampling_num2)
        expected_plp = a * proportion1 + b * proportion2
        confidence_area = t * cover1 + b * cover2
        ci_min_value = expected_plp - confidence_area
        ci_max_value = expected_plp + confidence_area
        return ci_min_value, ci_max_value, expected_plp

    def sampling(self, index_i: list, index_j: list, lp_i: float, lp_j: float):
        """Sampling methods
        Args:
            index_i (list):
            index_j (list):
            lp_i (float):
            lp_j (float):
        Returns:
            expected_lp (float):
            index_i (list):
            index_j (list):
            min_error (float):
            max_error (float):
        """

        # Considering now that bags can have different sizes
        if self.choice == "half":
            new_bag_size = (len(index_i) + len(index_j)) // 2

            index_i, index_j = (
                self.random.choice(index_i, size=new_bag_size // 2, replace=False),
                self.random.choice(index_j, size=new_bag_size // 2, replace=False),
            )

        elif self.choice == "uniform":
            # re-implementing folllowing the paper defn.: sampling gamma \in [0, 1] from uniform distribution
            gamma = np.random.uniform(low=0.0, high=1.0)
            index_i = self.random.choice(index_i, size=max(1, int(np.round(gamma * len(index_i)))), replace=False)
            index_j = self.random.choice(index_j, size=max(1, int(np.round((1 - gamma) * len(index_j)))), replace=False)

        elif self.choice == "gauss":
            # re-implementing folllowing the paper defn.: sampling gamma from gaussian distribution
            gamma = np.random.normal(loc=0.5, scale=0.1, size=1)

            if gamma < 0:
                gamma = 0
            elif gamma > 1:
                gamma = 1
            
            index_i = self.random.choice(index_i, size=max(1, int(np.round(gamma * len(index_i)))), replace=False)
            index_j = self.random.choice(index_j, size=max(1, int(np.round((1 - gamma) * len(index_j)))), replace=False)

        ci_min, ci_max, expected_lp = self.ci_loss_interval(
            lp_i, lp_j, len(index_i), len(index_j), self.CI
        )

        return expected_lp, index_i, index_j, ci_min, ci_max

    def __getitem__(self, idx):
        data_i, lp_i = self.data[idx], self.lp[idx]
        MixBag = self.random.choice([True, False])
        if MixBag:
            j = np.random.randint(0, self.len)
            data_j, lp_j = (
                self.data[j],
                self.lp[j],
            )

            id_i = list(range(data_i.shape[0]))
            id_j = list(range(data_j.shape[0]))

            # expected_lp: mixed_bag's label proportion
            # id_i: index used for creating subbag_i from data_i
            # id_j: index used for creating subbag_j from data_j
            # ci_min: minimam value of confidence interval
            # ci_max: maximam value of confidence interval
            expected_lp, id_i, id_j, ci_min, ci_max = self.sampling(id_i, id_j, lp_i, lp_j)
                        
            subbag_i = data_i[id_i]
            subbag_j = data_j[id_j]

            mixed_bag = np.concatenate([subbag_i, subbag_j], axis=0)

            # bs: bag size, w: width, h: height, c: channel

            return {
                "data": mixed_bag,  # data: [10, 3, 32, 32]
                "label_prop": torch.tensor(expected_lp).float(),  # label_prop: [10]
                "ci_min_value": torch.tensor(ci_min).float(),  # ci_min_value: [10]
                "ci_max_value": torch.tensor(ci_max).float(),  # ci_max_value: [10]
            }

        else:
            data, lp = self.data[idx], self.lp[idx]

            ci_min, ci_max = (
                torch.full((1, self.classes), -1).reshape(self.classes).float(),
                torch.full((1, self.classes), -1).reshape(self.classes).float(),
            )

            return {
                "data": data,  # data: [10, 3, 32, 32]
                "label_prop": torch.tensor(lp).float(),  # label_prop: [10]
                "ci_min_value": ci_min,  # ci_min_value: [10]
                "ci_max_value": ci_max,  # ci_max_value: [10]
            }
