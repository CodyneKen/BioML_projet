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

import torch.nn as nn
import torch.nn.functional as F

class modele1couche(nn.Module):
    def __init__(self, inputX, outputY):
        super(modele1couche, self).__init__()
        hidden_layer = 50
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

if __name__ == '__main__':
    batch_size = 5  # nombre de données lues à chaque fois
    nb_epochs = 10  # nombre de fois que la base de données sera lue
    eta = 0.0001  # taux d'apprentissage

    # on lit les données
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('mnist.pkl.gz'))
    # on crée les lecteurs de données
    train_dataset = torch.utils.data.TensorDataset(data_train, label_train)
    test_dataset = torch.utils.data.TensorDataset(data_test, label_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # on initialise le modèle et ses poids
    model = modele1couche(data_train.shape[1], label_train.shape[1])
    torch.nn.init.uniform_(model.fc1.weight, -0.001, 0.001)
    torch.nn.init.uniform_(model.fc2.weight, -0.001, 0.001)
    # on initiliase l'optimiseur
    loss_func = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.SGD(model.parameters(), lr=eta)

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
        for x, t in test_loader:
            # on calcule la sortie du modèle
            y = model(x)
            # on regarde si la sortie est correcte
            acc += torch.argmax(y, 1) == torch.argmax(t, 1)
        # on affiche le pourcentage de bonnes réponses
        print(acc / data_test.shape[0])