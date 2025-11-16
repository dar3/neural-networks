import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def load_data(normalize=True, random_state=42):
    # Function loading and processing the data. Copied from previous lab

    hd = fetch_ucirepo(id=45)
    X = hd.data.features
    y = hd.data.targets['num']
    # binarization
    y_bin = (y > 0).astype(int)
    dataset = pd.concat([X, y_bin.rename('target')], axis=1)
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    dataset[categorical_cols] = dataset[categorical_cols].fillna('NaN').astype(str)

    dataset_enc = pd.get_dummies(dataset, columns=categorical_cols, drop_first=False)

    X_enc = dataset_enc.drop('target', axis=1)
    y_enc = dataset_enc['target'].values.astype(int)
    feature_names = X_enc.columns.tolist()
    X_vals = X_enc.values.astype(float)
    if normalize:
        scaler = StandardScaler()
        # data normalization
        X_scaled = scaler.fit_transform(X_vals)
        return X_scaled, y_enc, feature_names, scaler
    else:
        return X_vals, y_enc, feature_names, None


def calc_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = None
    return {'accuracy': acc, 'precision': precision, 'recall': rec, 'f1': f1, 'auc': auc}


# Implementation in PyTorch

# Defining neural network model
class MLPInTorch(nn.Module):

    # Defining MLP module inheriting after nn.Module.

    def __init__(self, input_dim, hidden_dim=32):
        # constructor of the higher class
        super(MLPInTorch, self).__init__()
        # Using nn.Sequential to define a network
        # 1 hidden layer, 32 neurons, ReLU activation
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Exit layer, returns 1 feature
            nn.Sigmoid()  # Sigmoid activation on exit
        )

    # going forward
    def forward(self, x):
        return self.network(x)



# Main experimental function
def pytorch_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uzywanie CPU czy GPU (cuda)?:  {device}")

    # Loading data
    seed = 42
    X, y, feat_names, scaler = load_data(normalize=True, random_state=seed)
    # 80% training data 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    input_dim = X_train.shape[1]

    # Convering data to PyTorch tensors
    X_train_t = torch.tensor(X_train.astype(np.float32))
    # labels need to be (N, 1) for BCELoss and be float
    y_train_t = torch.tensor(y_train.astype(np.float32)).view(-1, 1)
    X_test_t = torch.tensor(X_test.astype(np.float32))
    y_test_t = torch.tensor(y_test.astype(np.float32)).view(-1, 1)

    # declaring optimizers
    optimizers_to_test = ['SGD', 'Adam', 'RMSprop']
    batch_sizes_to_test = [16, 32, 64]

    lrs_to_test = {
        'SGD': [0.1, 0.05, 0.01],
        'Adam': [0.01, 0.001, 0.0001],
        'RMSprop': [0.01, 0.001, 0.0001]
    }

    results = []

    print("Uruchamianie eksperymentow PyTorch")

    # Experiments loop over optimizers and batches
    for optimizer_name in optimizers_to_test:
        for batch_size in batch_sizes_to_test:

            # Creating DataLoaders - new for every batch size (merging tensors and labels_
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            val_dataset = TensorDataset(X_test_t, y_test_t)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            for lr in lrs_to_test[optimizer_name]:
                print(f" Aktualnie testowane: Opt={optimizer_name}, Batch={batch_size}, LR={lr}")
                start_time = time.time()

                # Each time creating new model for the epoch to reset the weights
                # makes experiment independent of each other
                model = MLPInTorch(input_dim, hidden_dim=32)

                # Starting training
                model, history = train_and_return_results(model, train_loader, val_loader, optimizer_name, lr,
                                                          max_epochs=300, tol=1e-6, device=device)

                elapsed = time.time() - start_time

                # Evaluation on test dataset
                model.eval()
                with torch.no_grad():
                    probs_test_t = model(X_test_t.to(device))

                # Converting back to NumPy for calc_metrics function
                probs_test_np = probs_test_t.cpu().numpy().ravel()

                test_metrics = calc_metrics(y_test, probs_test_np)

                res = {
                    'optimizer': optimizer_name,
                    'batch_size': batch_size,
                    'lr': lr,
                    'epochs': history['epochs'],
                    'train_loss': history['train_loss'][-1],
                    'val_loss': history['val_loss'][-1],
                    'test_acc': test_metrics['accuracy'],
                    'test_auc': test_metrics['auc'],
                    'test_f1': test_metrics['f1'],
                    'time_s': elapsed
                }
                results.append(res)


    df_results = pd.DataFrame(results)
    return df_results


# Training function
def train_and_return_results(model, train_loader, val_loader, name_of_optimizer, lr, max_epochs, tol, device):
    model.to(device)

    if name_of_optimizer == 'SGD':
        # momentum takes weighted average of previous gradients
        # This allows for more stable learning.
        # 90% same direction, 10 % gradient from actual batch
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif name_of_optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif name_of_optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Brak danego optymalizatora w programie: {name_of_optimizer}")

    # loss function
    loss_fn = nn.BCELoss()

    train_loss_history = []
    val_loss_history = []

    # Training loop. Iterating through all epochs
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0

        # Iteration over batches
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # moving forward (predictions)
            outputs = model(inputs)

            # loss computing
            loss = loss_fn(outputs, labels)

            # back propagation
            loss.backward()

            # updating weights
            optimizer.step()

            # adding loss from batch to the sum
            running_loss += loss.item()

        # saving epochs loss
        epoch_train_loss = running_loss / len(train_loader)
        train_loss_history.append(epoch_train_loss)

        # Evaluation on validation set
        model.eval()
        running_val_loss = 0.0
        #  turning off calculating gradients. Saves time
        with torch.no_grad():
            for inputs_val, labels_val in val_loader:
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                outputs_val = model(inputs_val)
                val_loss = loss_fn(outputs_val, labels_val)
                running_val_loss += val_loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_loss_history.append(epoch_val_loss)

        # stop function
        if epoch > 1 and abs(val_loss_history[-2] - val_loss_history[-1]) < tol:
            print(f"Zbieganie po {epoch} epokach.")
            break

    return model, {'train_loss': train_loss_history, 'val_loss': val_loss_history, 'epochs': epoch}


if __name__ == "__main__":
    # running all experiments
    results_df = pytorch_experiments()

    print("\n" + "=" * 50)

    print("Wszystkie wyniki eksperymentow")

    print("=" * 50)
    # printing whole results table
    print(results_df.to_string())

    results_df.to_csv("pytorch_results.csv", index=False)
    print(f"\nWyniki zapisane do 'pytorch_results.csv'")

    # top 5 configurations
    print("\n" + "=" * 50)
    print("Najlepsze konfiguracje (sortowane po test_auc)")
    print("=" * 50)
    print(results_df.sort_values(by='test_auc', ascending=False).head(5).to_string())
