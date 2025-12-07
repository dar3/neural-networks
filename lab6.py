import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Used CPU or GPU (CUDA): {device}")



class FlexibleCNN(nn.Module):
    def __init__(self, out_channels, kernel_size, pool_size, num_classes=10):
        super(FlexibleCNN, self).__init__()


        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size
        )


        self.relu = nn.ReLU()

        # Max Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=pool_size)

        # Flattening data from (Batch, Channels, H, W) to (Batch, Features)
        self.flatten = nn.Flatten()

        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Going through convolusion - activation - pooling
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)


        x = self.flatten(x)

        # Classification
        x = self.fc(x)
        return x




def add_gaussian_noise(images, std_dev):

    if std_dev == 0.0:
        return images

    noise = torch.randn_like(images) * std_dev
    noisy_images = images + noise
    return noisy_images


def train_one_epoch(model, loader, optimizer, criterion, noise_std):
    model.train()
    total_loss = 0
    correct = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        # Applying noise to training data
        data = add_gaussian_noise(data, noise_std)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / len(loader.dataset)
    return avg_loss, accuracy


def evaluate(model, loader, criterion, noise_std):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            # Adding noised data
            data = add_gaussian_noise(data, noise_std)

            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / len(loader.dataset)
    return avg_loss, accuracy


# Preparing data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalization for MNIST
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


def run_experiment_with_history(out_channels, kernel_size, pool_size, train_noise, test_noise, label):
    print(f"Start: {label} (Ch={out_channels}, K={kernel_size}, P={pool_size})")
    model = FlexibleCNN(out_channels, kernel_size, pool_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    history = []

    epochs = 12
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, noise_std=train_noise)
        test_loss, test_acc = evaluate(model, test_loader, criterion, noise_std=test_noise)


        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })
        print(f"  Epoka {epoch}: Test Acc = {test_acc:.2f}%")

    return history


if __name__ == "__main__":

    noise_level = 0.5


    scenarios = [
        {
            "name": "Brak szumu (Referencja)",
            "params": (16, 3, 2, 0.0, 0.0)
        },
        {
            "name": "Szum tylko w tescie",
            "params": (16, 3, 2, 0.0, noise_level)
        },
        {
            "name": "Szum w treningu i tescie",
            "params": (16, 3, 2, noise_level, noise_level)
        },

        {
            "name": "Duzy Kernel (5x5)",
            "params": (16, 5, 2, 0.0, 0.0)
        },

        # 32 kanaly, filtr 3x3, standardowy pooling 2x2
        {
            "name": "32 Kanały, Pool 2x2",
            "params": (32, 3, 2, 0.0, 0.0)

        },

        # 32 kanaly, filtr 3x3, bardziej restrykcyjny pooling 3x3
        {
            "name": "32 Kanały, Pool 3x3",
            "params": (32, 3, 3, 0.0, 0.0)

        },


    ]


    csv_name = "eksperymenty_wyn.csv"

    with open(csv_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        header = ["Scenariusz", "Epoka", "Train Loss", "Train Acc (%)", "Test Loss", "Test Acc (%)",
                  "Out Channels", "Kernel Size", "Pool Size", "Train Noise", "Test Noise"]
        writer.writerow(header)

        plot_data = {}


        for sc in scenarios:
            name = sc["name"]

            ch, k, p, t_noise, te_noise = sc["params"]


            history = run_experiment_with_history(ch, k, p, t_noise, te_noise, label=name)


            accuracies = [entry['test_acc'] for entry in history]
            plot_data[name] = accuracies


            for entry in history:
                row = [
                    name,
                    entry['epoch'],
                    f"{entry['train_loss']:.4f}",
                    f"{entry['train_acc']:.2f}",
                    f"{entry['test_loss']:.4f}",
                    f"{entry['test_acc']:.2f}",
                    ch, k, p, t_noise, te_noise
                ]
                writer.writerow(row)

    print(f"\nWyniki zapisano: {csv_name}")


    plt.figure(figsize=(10, 6))
    for name, acc_list in plot_data.items():
        plt.plot(range(1, len(acc_list) + 1), acc_list, marker='o', label=name)

    plt.title('Porownanie dokladnosci modeli (Test Accuracy)')
    plt.xlabel('Epoka')
    plt.ylabel('Dokladnosc w %')
    plt.grid(True)
    plt.legend()
    plt.savefig("wykres_porownawczy_1.png")
    plt.show()


