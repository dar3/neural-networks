import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import numpy as np


# Data preparing

def get_data_loaders(batch_size=64, subset_fraction=1.0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # normalization from [-1 to 1]
    ])

    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    if subset_fraction < 1.0:
        subset_size = int(len(train_set) * subset_fraction)
        train_subset, _ = random_split(train_set, [subset_size, len(train_set) - subset_size])
        train_ds = train_subset
    else:
        train_ds = train_set

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Model and noise

def add_gauss_noise(tensor, std_dev):
    if std_dev <= 0:
        return tensor
    noise = torch.randn_like(tensor) * std_dev
    return tensor + noise


class FMLP(nn.Module):

    def __init__(self, num_layers, hidden_units):
        super(FMLP, self).__init__()
        self.flatten = nn.Flatten()  # Flatten image

        layers = []
        input_dim = 28 * 28  # 784

        # 1 hidden layer
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(nn.ReLU())

        # 2 hidden layer
        if num_layers == 2:
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())

        # Last (exit) layer
        layers.append(nn.Linear(hidden_units, 10))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


# def train_model()(model, train_loader, test_loader, epochs, lr, train_noise_std=0.0, test_noise_std=0.0):
#     model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#
#     history = {'train_loss': [], 'test_acc': []}
#
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#
#             # Adding noise to train data. Experiment with noise
#             if train_noise_std > 0:
#                 images = add_gauss_noise(images, train_noise_std)
#
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#         avg_loss = running_loss / len(train_loader)
#         history['train_loss'].append(avg_loss)
#
#         # Ewaluacja
#         acc = evaluate_model(model, test_loader, test_noise_std)
#         history['test_acc'].append(acc)
#
#     return history

def train_model(model, train_loader, test_loader, epochs, lr, train_noise_std=0.0, test_noise_std=0.0):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'test_acc': []}

    print(
        f"{'Epoch':^5} | {'Train Loss':^10} | {'Test Acc':^10} | {'Test F1':^10} | {'Precision':^10} | {'Recall':^10}")
    print("-" * 75)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if train_noise_std > 0:
                images = add_gauss_noise(images, train_noise_std)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        # Ewaluacja (zwraca teraz słownik metryk)
        metrics = evaluate_model(model, test_loader, test_noise_std)
        history['test_acc'].append(metrics['acc'])  # Zachowujemy zgodność z wykresami

        # Wypisywanie statystyk do konsoli
        print(
            f"{epoch + 1:^5} | {avg_loss:^10.4f} | {metrics['acc']:^10.4f} | {metrics['f1']:^10.4f} | {metrics['prec']:^10.4f} | {metrics['rec']:^10.4f}")

    return history

# def evaluate_model(model, loader, noise_std=0.0):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in loader:
#             images, labels = images.to(device), labels.to(device)
#
#             if noise_std > 0:
#                 images = add_gauss_noise(images, noise_std)
#
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     return correct / total

def evaluate_model(model, loader, noise_std=0.0):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Dodawanie szumu (jeśli dotyczy)
            if noise_std > 0:
                images = add_gauss_noise(images, noise_std)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Zbieramy wyniki do list (przenosimy na CPU)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())


    metrics = {
        'acc': accuracy_score(all_targets, all_preds),
        'prec': precision_score(all_targets, all_preds, average='macro', zero_division=0),
        'rec': recall_score(all_targets, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_targets, all_preds, average='macro', zero_division=0)
    }

    return metrics


# Creating plots

def plot_results(results, title, filename):
    plt.figure(figsize=(12, 5))

    # Train loss plot
    plt.subplot(1, 2, 1)
    for name, hist in results.items():
        plt.plot(hist['train_loss'], label=name)
    plt.title(f'{title} - Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for name, hist in results.items():
        plt.plot(hist['test_acc'], label=name)
    plt.title(f'{title} - Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot has been saved: {filename}")
    plt.show()


# Tests

def exp_1_arch():
    print(" \n Exp. 1: Architecture nad Batch Size")
    results = {}

    configs = [
        # (name, layers, neurons, batch)
        ("1-Layer, 64 Neurons, Batch 32", 1, 64, 32),
        ("2-Layer, 64 Neurons, Batch 32", 2, 64, 32),
        ("1-Layer, 256 Neurons, Batch 32", 1, 256, 32),
        ("1-Layer, 64 Neurons, Batch 256", 1, 64, 256),
    ]

    for name, layers, neurons, batch in configs:
        print(f"Training: {name}")
        train_loader, test_loader = get_data_loaders(batch_size=batch, subset_fraction=1.0)
        model = FMLP(num_layers=layers, hidden_units=neurons)
        hist = train_model(model, train_loader, test_loader, epochs=10, lr=0.001)
        results[name] = hist

    plot_results(results, "Architektura i Batch", "exp1_arch_batch.png")


def exp_2_datasize():
    print("\n--- Experiment 2: Size of training data 1/10/100% ---")
    results = {}

    fractions = [0.01, 0.1, 1.0]  # 1, 10 ,100 %

    for frac in fractions:
        name = f"Data: {int(frac * 100)}%"
        print(f"Training: {name}")
        train_loader, test_loader = get_data_loaders(batch_size=64, subset_fraction=frac)

        model = FMLP(num_layers=2, hidden_units=128)
        hist = train_model(model, train_loader, test_loader, epochs=15, lr=0.001)
        results[name] = hist

    plot_results(results, "Wpływ Ilości Danych", "exp2_datasize.png")


def exp_3_noise():
    print("\n Exp. 3 Noie tolerance")
    train_loader, test_loader = get_data_loaders(batch_size=64, subset_fraction=1.0)
    noise_level = 0.2

    results = {}

    print("Plan A: Train Clean -> Test Noisy")
    model_clean = FMLP(num_layers=2, hidden_units=128)
    hist_clean = train_model(model_clean, train_loader, test_loader, epochs=10, lr=0.001, train_noise_std=0.0,
                                test_noise_std=noise_level)
    results["Train Clean / Test Noisy"] = hist_clean

    print("Plan B: Train Noisy -> Test Noisy")
    model_noisy = FMLP(num_layers=2, hidden_units=128)
    hist_noisy = train_model(model_noisy, train_loader, test_loader, epochs=10, lr=0.001,
                                train_noise_std=noise_level, test_noise_std=noise_level)
    results["Train Noisy / Test Noisy"] = hist_noisy

    print("Referencja: Train Clean -> Test Clean")
    model_ref = FMLP(num_layers=2, hidden_units=128)
    hist_ref = train_model(model_ref, train_loader, test_loader, epochs=10, lr=0.001, train_noise_std=0.0,
                              test_noise_std=0.0)
    results["Reference (Clean/Clean)"] = hist_ref

    plot_results(results, "Wpływ Szumu (std=0.2)", "exp3_noise.png")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Used device CPU/Cuda (GPU): {device}")

if __name__ == "__main__":
    start_global = time.time()

    exp_1_arch()
    exp_2_datasize()
    exp_3_noise()

    print(f"\nProgam finally finished running in: {time.time() - start_global:.2f} seconds.")
