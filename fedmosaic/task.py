"""fedmosaic: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from datasets import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
import pandas as pd
from flwr.common import Context

FRACTION_MASKED = 1.0
MASK_TYPE = "img"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No sigmoid here
        return x.squeeze(1)  # Output shape: (batch_size,)

fds = None  # Cache FederatedDataset

def generate_synthetic_dataset(n_samples=1000):
    """
    Generates a DataFrame with two columns:
    - 'features': each row contains a NumPy array of shape (1024,)
    - 'label': binary class label (0 or 1)

    Classification signal is embedded in two distinct subregions per class,
    one in each half of the 1024-length feature vector.
    
    Returns:
        df: pandas DataFrame with columns ['features', 'label']
    """
    X = np.random.uniform(0, 1, (n_samples, 1024))
    y = np.random.randint(0, 2, size=(n_samples,))

    for i in range(n_samples):
        if y[i] == 0:
            bandA = (512,12)
            bandB = (1024,612)
            X[i, bandA[1]:bandA[0]] += np.random.normal(3.0, 1.5, size=bandA[0] - bandA[1])
            X[i, bandB[1]:bandB[0]] += np.random.normal(3.0, 1.5, size=bandB[0] - bandB[1])
        else:
            bandA = (300,0)
            bandB = (850,512)
            X[i, bandA[1]:bandA[0]] += np.random.normal(3.0, 1.5, size=bandA[0] - bandA[1])
            X[i, bandB[1]:bandB[0]] += np.random.normal(3.0, 2.3, size=bandB[0] - bandB[1])

        if np.random.rand() < FRACTION_MASKED:
            if MASK_TYPE == "img":
                X[i,:512] = 0
            else:
                X[i,512:] = 0

    df = pd.DataFrame({
        "feature": list(X),  # Each row holds a (1024,) array
        "label": torch.tensor(y, dtype=torch.float)
    })

    return Dataset.from_pandas(df)

def load_data(partition_id: int, num_partitions: int, synthetic_data:bool):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:

        if not synthetic_data:
            #Real dataset
            partitioner = IidPartitioner(num_partitions=num_partitions)
            partitioner.dataset = ... #Load real data
            partition = fds.load_partition(partition_id)
        else:
            #Todo: Add mask to randomly select both, some or none
            partitioner = IidPartitioner(num_partitions=num_partitions)
            partitioner.dataset = generate_synthetic_dataset(1000)
            partition = partitioner.load_partition(partition_id=partition_id)

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2)

    def apply_transforms(batch):
        batch["feature"] = [(torch.tensor(ft, dtype=torch.float32) - 0.5) / 0.5 for ft in batch["feature"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device, feature="img"):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch[feature]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device, feature="img"):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()    
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch[feature].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            predicted = (torch.sigmoid(outputs) >= 0.5).long()
            correct += (predicted == labels).sum().item()
            #correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
