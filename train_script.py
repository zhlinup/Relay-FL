# -*- coding: utf-8 -*-

import numpy as np

np.set_printoptions(precision=6, threshold=1e3)
import torch

from torchvision import datasets, transforms
import copy
import torch.nn as nn
from torch.utils.data import DataLoader


def mnist_iid(dataset, K, M):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(M):
        dict_users[i] = set(np.random.choice(all_idxs, int(K[i]), replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def load_fmnist_iid(K):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])
    dataset_train = datasets.FashionMNIST('./data/FASHION_MNIST/', download=True, train=True, transform=transform)
    dataset_test = datasets.FashionMNIST('./data/FASHION_MNIST/', download=True, train=False, transform=transform)

    loader = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)
    images, labels = next(enumerate(loader))[1]
    images, labels = images.numpy(), labels.numpy()
    D_k = int(len(labels) / K)

    train_images = []
    train_labels = []
    dict_users = {i: np.array([], dtype='int64') for i in range(K)}
    all_idxs = np.arange(len(labels))

    D = np.zeros(K)
    for i in range(K):
        dict_users[i] = set(np.random.choice(all_idxs, int(D_k), replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        train_images.append(images[list(dict_users[i])])
        train_labels.append(labels[list(dict_users[i])])
        D[i] = len(dict_users[i])

    test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=True)
    test_images, test_labels = next(enumerate(test_loader))[1]

    return train_images, train_labels, test_images.numpy(), test_labels.numpy(), D


def load_fmnist_noniid(K, NUM_SHARDS):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])
    dataset_train = datasets.FashionMNIST('./data/FASHION_MNIST/', download=True, train=True, transform=transform)
    dataset_test = datasets.FashionMNIST('./data/FASHION_MNIST/', download=True, train=False, transform=transform)

    loader = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)
    images, labels = next(enumerate(loader))[1]
    images, labels = images.numpy(), labels.numpy()

    train_images = []
    train_labels = []

    # PART = 10
    PART = 1

    num_shards = K * NUM_SHARDS * PART
    num_imgs = int(len(images) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(K)}
    all_idxs = np.arange(len(labels))

    # sort labels
    idxs_labels = np.vstack((all_idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    all_idxs = idxs_labels[0, :]

    idx_shard = idx_shard[::PART]

    D = np.zeros(K)
    for i in range(K):
        rand_set = set(np.random.choice(idx_shard, NUM_SHARDS, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], all_idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        train_images.append(images[dict_users[i]])
        train_labels.append(labels[dict_users[i]])
        D[i] = len(dict_users[i])

    test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=True)
    test_images, test_labels = next(enumerate(test_loader))[1]

    return train_images, train_labels, test_images.numpy(), test_labels.numpy(), D


def local_update(setup, d, model1, train_images, train_labels, idx, batch_size):
    initital_weight = copy.deepcopy(model1.state_dict())

    model = copy.deepcopy(model1)
    model.train()

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=setup.lr, momentum=setup.momentum)

    # optimizer = torch.optim.Adam(model.parameters(), lr=setup.lr)

    epoch_loss = []
    images = np.array_split(train_images[idx], len(train_images[idx]) // batch_size)
    labels = np.array_split(train_labels[idx], len(train_labels[idx]) // batch_size)

    for epoch in range(setup.local_ep):
        batch_loss = []
        for b_idx in range(len(images)):
            model.zero_grad()

            log_probs = model(torch.tensor(images[b_idx].copy(), device=setup.device))
            local_loss = loss_function(log_probs, torch.tensor(labels[b_idx].copy(), device=setup.device))

            local_loss.backward()
            optimizer.step()
            if setup.verbose == 2:
                print('User: {}, Epoch: {}, Batch No: {}/{} Loss: {:.6f}'.format(idx,
                                                                                 epoch, b_idx + 1, len(images),
                                                                                 local_loss.item()))
            batch_loss.append(local_loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    copyw = copy.deepcopy(model.state_dict())
    gradient2 = np.array([[]])
    w2 = np.array([[]])
    for item in copyw.keys():
        gradient2 = np.hstack((gradient2, np.reshape((initital_weight[item] - copyw[item]).cpu().numpy(),
                                                     [1, -1]) / setup.lr))

        w2 = np.hstack((w2, np.reshape((copyw[item] - initital_weight[item]).cpu().numpy(),
                                       [1, -1])))

    return w2, sum(epoch_loss) / len(epoch_loss), gradient2


def test_model(model, setup, test_images, test_labels):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    images = torch.tensor(test_images).to(setup.device)
    labels = torch.tensor(test_labels).to(setup.device)
    outputs = model(images).to(setup.device)
    loss_function = nn.CrossEntropyLoss()
    batch_loss = loss_function(outputs, labels)
    loss += batch_loss.item()
    _, pred_labels = torch.max(outputs, 1)
    pred_labels = pred_labels.view(-1)

    correct += torch.sum(torch.eq(pred_labels, labels)).item()
    total += len(labels)
    accuracy = correct / total

    if setup.verbose:
        print('Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            loss, int(correct), int(total), 100.0 * accuracy))
    return accuracy, loss
