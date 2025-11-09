import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


#  funkcja sigma
# przeksztalcanie "n" do przedz. (0 - 1) jako prawdopod. klasy 1
#  predykcja prawdopod. klasy 1
def sigmoid(n):
    return 1.0 / (1.0 + np.exp(-n))


# binarna entropia krzyzowa
# mala liczba eps = 1e-15,aby nie bylo bledow numerycznych PPP
# aby nie bylo log(0)
def compute_loss_cross_entropy(y_true, y_pred, eps=1e-15):
    # np.clip ochrona przed zlymi wartosciami PPP
    y_pred = np.clip(y_pred, eps, 1 - eps)
    # entropia krzyżowa w klasyfikacji binarnej
    loss = - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


# liczenie pochodnych fun. kosztu po wagach i biasie .
def comp_grads(X_batch, y_batch, preds):
    m = X_batch.shape[0]
    # wektor roznic .
    error = preds - y_batch
    # wektor długości n (dla każdej cechy)
    grad_w = (X_batch.T @ error) / m
    # srednia bledu jako skalar
    grad_b = np.mean(error)
    return grad_w, grad_b


#  X macierz (m, n) przykladow razy cech
# y wektor z etykietami 0/1
# max_iters max. ilosc epok
# tol - tolerancja zmiany loss do uznania zbieznosci
def train_logis_model(X, y, alpha=0.1, max_iters=1000, tol=1e-6, batch_size=None, verbose=True):

    m, n = X.shape
    rng = np.random.default_rng(0)

    # inicjalizacja wagi wektor zero dl. n. bias zero
    W = np.zeros(n)
    b = 0.0

    loss_history = []
    prev_loss = None
    it = 0

    # jesli batch_size None, to ustawiamy m (full batch)
    if batch_size is None:
        batch_size = m

    # petla po epokach
    while it < max_iters:
        it += 1
        # mieszamy dane, aby byly podawane w innej kolejnosci SGD/mini-batch
        perm = rng.permutation(m)
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        # przechodzenie po mini-batch
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # forward dla kazdego batcha
            z = X_batch @ W + b
            preds = sigmoid(z)



            grad_w, grad_b = comp_grads(X_batch, y_batch, preds)

            # aktualizacja wag i b
            W -= alpha * grad_w
            b -= alpha * grad_b

        # obliczanie loss na calym zbiorze treningowym
        preds_all = sigmoid(X @ W + b)
        loss = compute_loss_cross_entropy(y, preds_all)
        loss_history.append(loss)

        # wypisywanie co 10 epok wraz z 1 epoka
        if verbose and (it % 10 == 0 or it == 1):
            print(f"Epoch {it:4d}: loss = {loss:.6f}")

        # model zbiega i zatrzymaj petle
        if prev_loss is not None and abs(prev_loss - loss) < tol:
            if verbose:
                print(f"Converged after {it} epochs: loss change {abs(prev_loss - loss):.2e} < tol {tol}")
            break
        prev_loss = loss

    return {"W": W, "b": b, "loss_history": loss_history, "epochs": it}



#  Ocena modelu
def eval_model(W, b, X_test, y_test):

    # prawdopod. klasy 1 dla zbioru testowego.
    probs = sigmoid(X_test @ W + b)
    preds = (probs >= 0.5).astype(int)

    # % popraw. predyk.
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = None
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "probs": probs, "preds": preds}



# ladowanie danych laczenie cech i target w dataset
def prepare_data_cats_to_cols_scaler():
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    dataset = pd.concat([X, y], axis=1)

    # BINARY dividing (0 health or 1) 1 is sum of 1-4 elements
    dataset['target_binary'] = dataset['num'].apply(lambda x: 1 if x > 0 else 0)


    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    # brakujace wartosci to Nan i rzutowanie na string
    dataset[categorical_cols] = dataset[categorical_cols].fillna('NaN').astype(str)

    # tworzenie kolumn 0/1 dla kazdej kateogrii
    # wszystkie kolumny w X teraz beda numeryczne
    dataset_encoded = pd.get_dummies(dataset, columns=categorical_cols, drop_first=False)

    # tworzenie macierzy cech  x_enc wszystko oprocz num i target_binary
    # y enc - wektor 0/1
    X_encoded = dataset_encoded.drop(['num', 'target_binary'], axis=1)
    y_encoded = dataset_encoded['target_binary'].astype(int)

    # standaryzacja cech numerycznych
    # Gradient descent uczy stabilniej gdy cechy maja porownywalne skale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded.values)


    # lista nazw, cech
    return X_scaled, y_encoded.values, X_encoded.columns.tolist(), scaler



if __name__ == "__main__":
    # przygotowanie danych
    X, y, feature_names, scaler = prepare_data_cats_to_cols_scaler()
    print("Macierz X shape:", X.shape, "y shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Train/test:", X_train.shape, X_test.shape)


    alpha = 0.5
    max_iters = 500
    tol = 1e-6  # kryterium zbieżności wg zmiany kosztu
    batch_size = 32

    # poczatek trenowania
    model = train_logis_model(X_train, y_train, alpha=alpha, max_iters=max_iters, tol=tol, batch_size=batch_size,
                              verbose=True)



    # ocena wynikow na zbiorze treningowym

    res_train = eval_model(model['W'], model['b'], X_train, y_train)
    print("")
    print("Ewaluacja na zbiorze treningowym:")
    print(f"Accuracy:  {res_train['accuracy']:.4f}")
    print(f"Precision: {res_train['precision']:.4f}")
    print(f"Recall:    {res_train['recall']:.4f}")
    print(f"F1:        {res_train['f1']:.4f}")
    if res_train['auc'] is not None:
        print(f"AUC:       {res_train['auc']:.4f}")



    # Ocena wynikow na zbiorze testowym
    res = eval_model(model['W'], model['b'], X_test, y_test)
    print("")
    print("Ewaluacja na zbiorze testowym:")
    print(f"Accuracy:  {res['accuracy']:.4f}")
    print(f"Precision: {res['precision']:.4f}")
    print(f"Recall:    {res['recall']:.4f}")
    print(f"F1:        {res['f1']:.4f}")
    if res['auc'] is not None:
        print(f"AUC:       {res['auc']:.4f}")

    # wykres funk. kosztu
    plt.figure(figsize=(6, 4))
    plt.plot(model['loss_history'], marker='o')
    plt.title('Loss (cross-entropy) w kolejnych epokach')
    plt.xlabel('epoka')
    plt.ylabel('loss')
    plt.grid(True)
    plt.show()
