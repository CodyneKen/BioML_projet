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

# Choose the platform
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

# For benchmarking
def displayTime(start_time):
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

class model3layers(nn.Module):
    #hl = hidden_layer
    def __init__(self, inputX, hl1, hl2,outputY):
        super(model3layers, self).__init__()

        #Linear regression layers
        self.fc1 = nn.Linear(inputX, hl1)
        self.fc2 = nn.Linear(hl1, hl2)
        self.fc3 = nn.Linear(hl2, outputY)

        #Convolutional layers
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
        # return x

# N regular layers model (same size layer n times)
class modelNreg(nn.Module):
    def __init__(self, inputX, nb_layer, layer_size, outputY):
        super(modelNreg, self).__init__()
        self.layers = nn.ModuleList()

        # Add the first layer
        self.layers.append(nn.Linear(inputX, layer_size))

        # Add the hidden layers
        for _ in range(nb_layer - 1):
            self.layers.append(nn.Linear(layer_size, layer_size))

        # Add the output layer
        self.layers.append(nn.Linear(layer_size, outputY))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def initialize_layers(self):
        # Will do nothing if no layer created
        for layer in self.layers:
            torch.nn.init.uniform_(layer.weight, -0.001, 0.001)


# Choose the experiment to run
experiment = 2

#Multi layer experimentation
# Juste de l'expérimentation (gardé le code pour référence)
if __name__ == '__main__' and experiment == 1:
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

    print("Shape of training data:", data_train.shape)
    print("Shape of training labels:", label_train.shape)

    print("Shape of test data:", data_test.shape)
    print("Shape of test labels:", label_test.shape)

    eta = 0.001
    # Hyperparamètres
    acc_max_hyper = 0.
    batch_size_max = 0
    hl1_max= 0
    hl2_max= 0
    print("TEST HYPERPARAMETRES")
    for batch_size in range(1, 11, 2):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for hl1 in [10, 30, 50, 70, 100]:
            for hl2 in [10, 30, 50, 70, 100]:
                # Might need to vary hlsize +10, -10 within model
                model = model3layers(data_train.shape[1],hl1, hl2,label_train.shape[1])
                torch.nn.init.uniform_(model.fc1.weight, -0.001, 0.001)
                torch.nn.init.uniform_(model.fc2.weight, -0.001, 0.001)
                torch.nn.init.uniform_(model.fc3.weight, -0.001, 0.001)

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
                print("batch_size: ", batch_size,"eta: ", eta,"hl1: ", hl1, "hl2:", hl2, "accuracy ==>", acc / len(validation_dataset))
                displayTime(start_time)
                if acc > acc_max_hyper:
                    print("new best of hyperparameters !")
                    acc_max_hyper = acc
                    batch_size_max = batch_size
                    eta_max = eta
                    hl1_max = hl1
                    hl2_max = hl2

    print("BEST HYPERPARAMETERS ==>", "batch_size: ", batch_size_max,"eta: ", eta_max,"hl1: ", hl1_max, "hl2:", hl2_max, "accuracy ==>", acc_max_hyper / len(validation_dataset))
    # On réinjecte les meilleurs hyperparamètres
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_max, shuffle=True)

    model = model3layers(data_train.shape[1], hl1_max, hl2_max, label_train.shape[1])
    torch.nn.init.uniform_(model.fc1.weight, -0.001, 0.001)
    torch.nn.init.uniform_(model.fc2.weight, -0.001, 0.001)
    torch.nn.init.uniform_(model.fc3.weight, -0.001, 0.001)

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

    # BENCHMARKING TIME
    displayTime(start_time)
    print("END OF PROGRAM")

# Correspond a la question 2 de la partie 3
if __name__ == '__main__' and experiment == 2:
    batch_size = 5  # nombre de données lues à chaque fois
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

    print("Shape of training data:", data_train.shape)
    print("Shape of training labels:", label_train.shape)

    print("Shape of test data:", data_test.shape)
    print("Shape of test labels:", label_test.shape)

    eta = 0.001
    # Hyperparamètres
    acc_max_hyper = 0.
    batch_size_max = 0
    ln_max= 0
    hls_max= 0
    print("TEST HYPERPARAMETRES")
    for batch_size in range(1, 11, 2):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for layer_number in range(1, 6):
            for hidden_layer_size in [10, 30, 50, 70, 100]:
                model = modelNreg(data_train.shape[1],layer_number, hidden_layer_size,label_train.shape[1])
                # on initialise le modèle et ses poids
                model.initialize_layers()

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
                print("batch_size: ", batch_size,"eta: ", eta,"layer_number: ", layer_number, "layer_sizes:", hidden_layer_size, "accuracy ==>", acc / len(validation_dataset))
                displayTime(start_time)
                if acc > acc_max_hyper:
                    print("new best of hyperparameters !")
                    acc_max_hyper = acc
                    batch_size_max = batch_size
                    eta_max = eta
                    ln_max = layer_number
                    hls_max = hidden_layer_size

    print("BEST HYPERPARAMETERS ==>", "batch_size: ", batch_size_max,"eta: ", eta_max,"layer_number: ", ln_max, "hidden_layer:", hls_max, "accuracy ==>", acc_max_hyper / len(validation_dataset))
    # On réinjecte les meilleurs hyperparamètres
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_max, shuffle=True)

    model = modelNreg(data_train.shape[1], ln_max, hls_max, label_train.shape[1])
    model.initialize_layers()

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

    # BENCHMARKING TIME
    displayTime(start_time)
    print("END OF PROGRAM")
