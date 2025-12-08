import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import tarfile
import re
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import matplotlib.pyplot as plt


FILENAME = 'data/imdb/aclImdb_v1.tar.gz'
VOCAB_SIZE = 10000  # limiting dictionary to 10K most frequent words
BATCH_SIZE = 64
EPOCHS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Used CPU or GPU (CUDA): {DEVICE}")


def raw_data_loader(filename):
    if not os.path.exists("aclImdb"):
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()

    data = []


    for split in ['train', 'test']:
        # adding binary names (positive - 1, negative - 0_
        for label_name, label_val in [('pos', 1), ('neg', 0)]:
            path = os.path.join('aclImdb', split, label_name)
            files = os.listdir(path)

            for file in files[:2500]:
                with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                    text = f.read()
                    data.append((text, label_val))
    return data



def build_vocab(data, vocab_size):
    # counting how many times particular word appeared
    counter = Counter()
    for text, _ in data:
        words = re.findall(r'\w+', text.lower())
        counter.update(words)

    most_common = counter.most_common(vocab_size - 2)
    #  PAD for filling empty spaces
    # UNK for words that doesn't appear in vocabulary
    vocabulary = {'<PAD>': 0, '<UNK>': 1}
    # assigning numbers to particular words
    for idx, (word, _) in enumerate(most_common, start=2):
        vocabulary[word] = idx

    return vocabulary

# converting sentence to numbers list
def convert_text_to_indices(text, vocab):
    words = re.findall(r'\w+', text.lower())
    return [vocab.get(w, 1) for w in words]


class RawIMDBDataset(Dataset):
    def __init__(self, raw_data, vocab, max_len=None):
        self.data = raw_data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        indices = convert_text_to_indices(text, self.vocab)

        # Truncation
        if self.max_len:
            indices = indices[:self.max_len]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float32)



def dataloader_creator(raw_data, vocab, max_len_truncate=None, batch_size=64):
    dataset = RawIMDBDataset(raw_data, vocab, max_len=max_len_truncate)

    def collate_fn(batch):
        texts, labels = zip(*batch)
        #  filling empty spaces in batches with zeros so GPU can use this matrix
        texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
        return texts_padded, labels

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# Defininf RNN and LSTM
class RecurrentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, rnn_type='RNN'):
        super(RecurrentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn_type = rnn_type

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)

        if self.rnn_type == 'LSTM':
            output, (hidden, cell) = self.rnn(embedded)
            # taking state from the last step. It should contain context of review in it
            final_hidden = hidden[-1]
        else:
            output, hidden = self.rnn(embedded)
            final_hidden = hidden[-1]

        return self.sigmoid(self.fc(final_hidden))


def experiments_runner(raw_data, vocab, rnn_type='RNN', hidden_dim=32, max_len=None, exp_name="Experiment"):
    print(f"\n--- Start: {exp_name} ---")
    print(f"Typ: {rnn_type}, Hidden Dim: {hidden_dim}, Max Len: {max_len if max_len else 'Dynamic (Full)'}")

    loader = dataloader_creator(raw_data, vocab, max_len_truncate=max_len, batch_size=BATCH_SIZE)

    model = RecurrentModel(VOCAB_SIZE, embed_dim=64, hidden_dim=hidden_dim, output_dim=1, rnn_type=rnn_type)
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()


    history = []

    model.train()
    for epoch in range(EPOCHS):
        total_acc = 0
        total_loss = 0
        count = 0

        for texts, labels in loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(texts).squeeze(1)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            rounded_preds = torch.round(predictions)
            correct = (rounded_preds == labels).float()
            total_acc += correct.sum().item()
            total_loss += loss.item()
            count += labels.size(0)

        avg_loss = total_loss / count
        avg_acc = total_acc / count

        print(f"Epoka {epoch + 1}: Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")


        history.append({
            'Experiment': exp_name,
            'Epoch': epoch + 1,
            'Loss': avg_loss,
            'Accuracy': avg_acc,
            'RNN_Type': rnn_type,
            'Hidden_Dim': hidden_dim,
            'Max_Len': str(max_len) if max_len else 'Full'
        })

    return history


def plot_results(df):
    plt.style.use('ggplot')

    experiments = df['Experiment'].unique()

    plt.figure(figsize=(12, 6))
    for exp in experiments:
        data = df[df['Experiment'] == exp]
        plt.plot(data['Epoch'], data['Accuracy'], marker='o', label=exp)

    plt.title('Porównanie Dokładności (Accuracy) wszystkich eksperymentów')
    plt.xlabel('Liczba Epok')
    plt.ylabel('Dokładność (Accuracy)')
    plt.legend()
    plt.grid(True)
    plt.savefig('wyniki_wszystkie_accuracy.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    target_exps = ["Proste RNN", "LSTM"]
    for exp in target_exps:
        if exp in experiments:
            data = df[df['Experiment'] == exp]
            plt.plot(data['Epoch'], data['Accuracy'], marker='o', label=exp)

    plt.title('Porównanie typu warstwy rekurencyjnej: RNN vs LSTM')
    plt.xlabel('Liczba Epok')
    plt.ylabel('Dokładność (Accuracy)')
    plt.legend()
    plt.grid(True)
    plt.savefig('wyniki_rnn_vs_lstm.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    target_exps = ["Pelna dlugosc (Dynamic Padding)", "Obciecie do 20 slow", "Obciecie do 50 slow"]
    for exp in target_exps:
        if exp in experiments:
            data = df[df['Experiment'] == exp]
            plt.plot(data['Epoch'], data['Accuracy'], marker='o', label=exp)

    plt.title('Wpływ przycinania sekwencji (Truncation)')
    plt.xlabel('Liczba Epok')
    plt.ylabel('Dokładność (Accuracy)')
    plt.legend()
    plt.grid(True)
    plt.savefig('wyniki_truncation.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    target_exps = ["LSTM Dim 16", "LSTM Dim 64"]
    for exp in target_exps:
        if exp in experiments:
            data = df[df['Experiment'] == exp]
            plt.plot(data['Epoch'], data['Accuracy'], marker='o', label=exp)

    plt.title('Porównanie wymiaru warstwy rekurencyjnej (Hidden Dimension)')
    plt.xlabel('Liczba Epok')
    plt.ylabel('Dokładność (Accuracy)')
    plt.legend()
    plt.grid(True)
    plt.savefig('wyniki_hidden_dim_comparison.png')
    plt.show()


if __name__ == "__main__":
    raw_data = raw_data_loader(FILENAME)
    vocab = build_vocab(raw_data, VOCAB_SIZE)
    print(f"Rozmiar danych: {len(raw_data)} recenzji")
    print(f"Rozmiar slownika: {len(vocab)} slow")

    all_results = []


    print("\n Experiment 1: RNN vs LSTM")
    all_results.extend(experiments_runner(raw_data, vocab, rnn_type='RNN', hidden_dim=32, max_len=None,
                                          exp_name="Proste RNN"))
    all_results.extend(experiments_runner(raw_data, vocab, rnn_type='LSTM', hidden_dim=32, max_len=None,
                                          exp_name="LSTM"))


    print("\nExperiment 2: Layer size")
    all_results.extend(experiments_runner(raw_data, vocab, rnn_type='LSTM', hidden_dim=16, max_len=None,
                                          exp_name="LSTM Dim 16"))
    all_results.extend(experiments_runner(raw_data, vocab, rnn_type='LSTM', hidden_dim=64, max_len=None,
                                          exp_name="LSTM Dim 64"))

    print("\nExperiment 3: Sequence length")
    all_results.extend(experiments_runner(raw_data, vocab, rnn_type='LSTM', hidden_dim=32, max_len=None,
                                          exp_name="Pelna dlugosc (Dynamic Padding)"))
    all_results.extend(
        experiments_runner(raw_data, vocab, rnn_type='LSTM', hidden_dim=32, max_len=20,
                           exp_name="Obciecie do 20 slow"))
    all_results.extend(
        experiments_runner(raw_data, vocab, rnn_type='LSTM', hidden_dim=32, max_len=50,
                           exp_name="Obciecie do 50 slow"))


    df = pd.DataFrame(all_results)

    csv_filename = 'wyniki_eksperymentow.csv'
    df.to_csv(csv_filename, index=False)


    plot_results(df)




