import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import ShuffleSplit
from llp_learn.mixbag import MixBag

def compute_proportions(bags, y):
    """
    Compute the proportion of positive samples in each bag.

    Parameters
    ----------
    bags : array-like, shape (n_samples,)
        The bag that each sample belongs to.
    
    y : array-like, shape (n_samples,)
        The labels of the samples.
    
    Returns
    -------
    proportions : array, shape (n_bags,)
        The proportion of positive samples in each bag.
    """
    num_bags = int(max(bags)) + 1
    proportions = np.empty(num_bags, dtype=float)
    for i in range(num_bags):
        bag = np.where(bags == i)[0]
        if len(bag) == 0:
            proportions[i] = np.nan
        else:
            proportions[i] = np.count_nonzero(y[bag] == 1) / len(bag)
    return proportions

execution = 0
seed = [189395, 962432364, 832061813, 316313123, 1090792484,
        1041300646,  242592193,  634253792,  391077503, 2644570296, 
        1925621443, 3585833024,  530107055, 3338766924, 3029300153,
       2924454568, 1443523392, 2612919611, 2781981831, 3394369024,
        641017724,  626917272, 1164021890, 3439309091, 1066061666,
        411932339, 1446558659, 1448895932,  952198910, 3882231031]

if __name__ == "__main__":
    #base_dataset = "sample-datasets-ci/cifar-10-grey-animal-vehicle.parquet"
    base_dataset = "sample-datasets-ci/adult.parquet"
    #base_dataset = "sample-datasets-ci/cifar-10.parquet"

    # Reading X, y (base dataset) and bags (dataset)
    df = pd.read_parquet(base_dataset)
    X = df.drop(["y"], axis=1).values
    y = df["y"].values
    y = y.reshape(-1)

    #dataset = "sample-datasets-ci/cifar-10-grey-animal-vehicle-hard-large-equal-close-global-cluster-kmeans-5.parquet"
    dataset = "sample-datasets-ci/adult-hard-large-equal-close-global-cluster-kmeans-5.parquet"
    #dataset = "sample-datasets-ci/cifar-10-intermediate-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40.parquet"

    df = pd.read_parquet(dataset)
    bags = df["bag"].values
    bags = bags.reshape(-1)

    train_index, test_index = next(ShuffleSplit(n_splits=1, test_size=0.25, random_state=seed[execution]).split(X))

    X_train, y_train, bags_train = X[train_index], y[train_index], bags[train_index]
    X_test, y_test, bags_test = X[test_index], y[test_index], bags[test_index]
    proportions = compute_proportions(bags_train, y_train)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    if "cifar" in base_dataset:
        X_train = X_train.reshape(-1, 3, 32, 32).astype(np.float32)
        X_test = X_test.reshape(-1, 3, 32, 32).astype(np.float32)

    import torchvision.transforms as transforms
    import torch

    seed = 0

    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm
    #torch.use_deterministic_algorithms(True)

    # X_train = np.load("sample-mixbag-datasets/train_bags.npy")
    # y_train = np.load("sample-mixbag-datasets/train_labels.npy")
    # proportions = np.load("sample-mixbag-datasets/train_lps.npy")

    # X_test = np.load("sample-mixbag-datasets/test_data.npy")
    # y_test = np.load("sample-mixbag-datasets/test_label.npy")

    # X_test = X_test.transpose(0, 3, 1, 2)
    # X_test = torch.tensor(X_test).float()

    # transform = transforms.Compose(
    #             [
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(
    #                     (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    #                 ),
    #             ]
    #         )

    if "cifar" in base_dataset: 
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            
        X_train = X_train.transpose(0, 2, 3, 1)
        X_test = X_test.transpose(0, 2, 3, 1)

        # transform X_train
        new_X_train = []
        for elem in X_train:
            new_X_train.append([transform(elem)])

        X_train = torch.from_numpy(np.array(new_X_train)).float()
        X_train = X_train.reshape(-1, 3, 32, 32)

        # transform X_test
        new_X_test = []
        for elem in X_test:
            new_X_test.append([transform(elem)])
        
        X_test = torch.from_numpy(np.array(new_X_test)).float()
        X_test = X_test.reshape(-1, 3, 32, 32)

    # #Transform X_train and X_test using the same transform to a torch tensor
    # new_X_train = []
    # for bag in X_train:
    #     new_X_train.append([transform(x) for x in bag])
    # new_X_train = np.array(new_X_train)
    # X_train = new_X_train
    # X_train = torch.tensor(new_X_train)

    # # new_X_test = []
    # # for bag in X_test:
    # #     new_X_test.append([transform(x) for x in bag])
    # # X_test = np.array(new_X_test)
    # # X_test = X_test.reshape(-1, 3, 28, 28)
    # # y_test = y_test.reshape(-1)

    try:
        N_JOBS = eval(os.getenv('NSLOTS'))
    except:
        N_JOBS = -1
    print("Using {} cores".format(N_JOBS))


    from llp_learn.model_selection import gridSearchCV    
    mb = MixBag(lr=0.01, n_epochs=1, consistency="none", choice="uniform", confidence_interval=0.005, verbose=True, random_state=seed, device="mps", model_type="simple-mlp", hidden_layer_sizes=(100,), n_jobs=N_JOBS)
    mb.fit(X_train, bags_train, proportions)
    y_pred = mb.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(np.unique(y_pred, return_counts=True))
    exit()
    
    #params = {"lr": [0.1, 0.01], "consistency": ["none", "vat"]}
    # gs = gridSearchCV(mb, params, n_jobs=1, random_state=seed)
    # gs.fit(X_train, bags_train, proportions)
    # y_pred = gs.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # print(np.unique(y_pred, return_counts=True))
    # print()
    # print(gs.best_params_)