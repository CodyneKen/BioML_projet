# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant les outils de Pytorch)
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

import gzip, numpy, torch

import gzip, numpy, torch
import time

import torch.nn as nn
import torch.nn.functional as F
from sympy.logic.inference import valid

platform = "cpu"

# CPU CONFIG
if platform == "cpu":
    device = torch.device("cpu")
    print("Device", device)

# GPU CONFIG
if platform == "gpu":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device)

# MPS CONFIG (MPS = Metal Performance Shaders) (Macos only)
if platform == "mac":
    device = torch.device("mps")
    print("Device", device)


start_time = time.time()

def displayTime(start_time):
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

class modele1couche(nn.Module):
    def __init__(self, inputX, outputY, hidden_layer):
        super(modele1couche, self).__init__()
        self.fc1 = nn.Linear(inputX, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, outputY)

    def forward(self, x):
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x
        # return x


# Single layer expermimentation
if __name__ != '__main__':
    # batch_size = 5  # nombre de données lues à chaque fois
    nb_epochs = 10  # nombre de fois que la base de données sera lue
    # eta = 0.0001  # taux d'apprentissage

    # on lit les données
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('mnist.pkl.gz'))
    # on crée les lecteurs de données
    train_dataset = torch.utils.data.TensorDataset(data_train, label_train)
    # on divise le dataset de train pour avoir un dataset de validation, avec un ratio de 80% pour le train et 20% pour la validation
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
    test_dataset = torch.utils.data.TensorDataset(data_test, label_test)

    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)



    # Hyperparamètres
    acc_max_hyper = 0.
    batch_size_max = 0
    eta_max = 0
    hlsize_max = 0
    print("TEST HYPERPARAMETRES")
    for batch_size in range(1, 11, 2):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for eta in [0.0001, 0.001, 0.01, 0.1]:
            for hlsize in range(10, 100, 10):

                model = modele1couche(data_train.shape[1], label_train.shape[1], hlsize)
                torch.nn.init.uniform_(model.fc1.weight, -0.001, 0.001)
                torch.nn.init.uniform_(model.fc2.weight, -0.001, 0.001)

                # on initialise le modèle et ses poids
                # on initiliase l'optimiseur
                loss_func = torch.nn.MSELoss(reduction='sum')
                optim = torch.optim.SGD(model.parameters(), lr=eta)

                # PARTIE CLASSIQUE DE L'APPRENTISSAGE
                for n in range(nb_epochs):
                    # on lit toutes les données d'apprentissage
                    for x, t in train_loader:
                        # on calcule la sortie du modèle
                        y = model(x)
                        # on met à jour les poids
                        loss = loss_func(t, y)
                        loss.backward()
                        optim.step()
                        optim.zero_grad()
                # test du modèle (on évalue la progression pendant l'apprentissage)
                acc = 0.
                # on lit toutes les donnéees de test
                for x, t in validation_loader:
                    # on calcule la sortie du modèle
                    y = model(x)
                    # on regarde si la sortie est correcte
                    acc += torch.argmax(y, 1) == torch.argmax(t, 1)

                # on affiche les hyperparamètres
                # on affiche le pourcentage de bonnes réponses
                print("batch_size: ", batch_size,"eta: ", eta,"hlsize: ", hlsize, "accuracy ==>", acc / len(validation_dataset))
                if acc > acc_max_hyper:
                    print("new best of hyperparameters !")
                    acc_max_hyper = acc
                    batch_size_max = batch_size
                    eta_max = eta
                    hlsize_max = hlsize

    # On réinjecte les meilleurs hyperparamètres
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_max, shuffle=True)

    model = modele1couche(data_train.shape[1], label_train.shape[1], hlsize_max)
    torch.nn.init.uniform_(model.fc1.weight, -0.001, 0.001)
    torch.nn.init.uniform_(model.fc2.weight, -0.001, 0.001)

    # on initialise le modèle et ses poids
    # on initiliase l'optimiseur
    loss_func = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.SGD(model.parameters(), lr=eta_max)

    # test du modèle (on évalue la progression pendant l'apprentissage)
    acc = 0.
    # on lit toutes les donnéees de test
    for x, t in test_loader:
        # on calcule la sortie du modèle
        y = model(x)
        # on regarde si la sortie est correcte
        acc += torch.argmax(y, 1) == torch.argmax(t, 1)
    # on affiche le pourcentage de bonnes réponses
    print(acc / data_test.shape[0])