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
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

DO_MASK = True
FRACTION_MASKED = 0.5 # All data in that client will be masked
DATA_PROPORTION = 1.0 # 100% of clients will have text and another will have img
PARTITION_CACHE = {}

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

class CrossPNet(nn.Module):
    def __init__(self):
        super(CrossPNet, self).__init__()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No sigmoid here
        return x.squeeze(1)  # Output shape: (batch_size,)



fds = None  # Cache FederatedDataset

def load_real_dataset():

    df = pd.read_csv("./embeddings/embeddings/fused_clip_embeddings.csv")
    df.drop("clip_textimg_512", axis=1, inplace=True)
    df.drop("filename", axis=1, inplace=True)
    
    y = df.iloc[:, 0].to_numpy(dtype=np.float32)
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    n_samples = len(y)

    data_dict = {
        'feature': [x.tolist() for x in X],
        'label': y.tolist()
    }

    hf_dataset = Dataset.from_dict(data_dict)

    return hf_dataset


def mask_partition(partition: Dataset, partition_num):
    # Usa o cache para obter ou definir o tipo de partição ('img' ou 'text')
    if partition_num not in PARTITION_CACHE:
        mask_type = 'img' if np.random.rand() < DATA_PROPORTION else 'text'
        PARTITION_CACHE[partition_num] = mask_type
    else:
        mask_type = PARTITION_CACHE[partition_num]

    def mask_example(example):
        x = example['feature']
        if np.random.rand() < FRACTION_MASKED and DO_MASK:
            if mask_type == 'img':
                x[:512] = [1] * 512
            else:
                x[512:] = [1] * 512

        # Produto vetorial para fusão
        x = [i * j for i, j in zip(x[:512], x[512:])]
        example['feature'] = x

        return example

    return partition.map(mask_example)

def load_data(partition_id: int, num_partitions: int, synthetic:bool):
    # Only initialize `FederatedDataset` once
    global fds
    partitioner = IidPartitioner(num_partitions=num_partitions)
    if fds is None:
        partitioner.dataset = load_real_dataset()
        fds = partitioner

    partition = fds.load_partition(partition_id=partition_id)
    partition = mask_partition(partition, partition_id)

    # Divide data on each node: 70% train, 30% test
    partition_train_test = partition.train_test_split(test_size=0.3)

    def apply_transforms(batch):
        #batch["feature"] = [(torch.tensor(ft, dtype=torch.float32) - 0.5) / 0.5 for ft in batch["feature"]]
        batch["feature"] = [torch.tensor(ft, dtype=torch.float32) for ft in batch["feature"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device, feature="img"):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
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
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
