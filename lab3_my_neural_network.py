import time
from typing import Union, Tuple, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


# Warstwy i aktywacje


class Linear:
    def __init__(self, in_dim, out_dim, weight_std=0.01, seed=None):
        rng = np.random.default_rng(seed)
        # waga wybierana losowo z rozkladu normalnego
        self.W = rng.normal(0.0, weight_std, size=(in_dim, out_dim))
        # bias inicjalizujemy zerami
        self.b = np.zeros(out_dim)
        # gradienty dw db do aktualizacji wag w backpropagation
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # zapisywanie cache dla backward
        self.cache_x = None


    # obliczanie wyjscia warstwy
    # macierz wej. x * wagi w + bias
    def forward(self, x):
        self.cache_x = x
        return x @ self.W + self.b

    def backward(self, d_out):
        x = self.cache_x
        m = x.shape[0]
        # obliczanie gradientow dw, db (gradient wag i biasow)
        self.dW = (x.T @ d_out) / m
        self.db = np.mean(d_out, axis=0)
        # przekazywanie gradientu do poprzedniej warstwy
        dx = d_out @ self.W.T
        return dx

    # aktualizacja wag i biasow za pomoca SGD
    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x
        # odcina wartosci ujemne
        return np.maximum(0, x)

    # przekazuje gradient tam gdzie x > 0
    # jesli neuron otrzymuje same wartości ≤ 0, przestaje się uczyć (gradient zawsze 0).
    def backward(self, d_out):
        x = self.cache
        dx = d_out * (x > 0).astype(float)
        return dx


class SigmoidAct:

    def __init__(self):
        self.cache = None

    def forward(self, x):
        # sigmoid zabezpieczony przed overflow
        z = np.clip(x, -500, 500)
        out = 1.0 / (1.0 + np.exp(-z))
        self.cache = out
        return out

    # mnozenie gradientu przez poch. sigmoidu
    def backward(self, d_out):
        s = self.cache
        return d_out * (s * (1 - s))


# jakakolwiek f.
# forward  liczy wyjscie
# backward liczy gradient
class CustActFunction:

    def __init__(self, func, func_grad):
        self.func = func
        self.func_grad = func_grad
        self.cache = None

    def forward(self, x):
        self.cache = x
        return self.func(x)

    def backward(self, d_out):
        x = self.cache
        dx = d_out * self.func_grad(x)
        return dx



# Loss (binary cross-entropy)

# obliczanie straty BCE dla klasyfikacji binarnej
# np.clip zab. przed log(0)
def bin_cross_entr_loss(y_true, y_prob, eps=1e-15):
    y_prob = np.clip(y_prob, eps, 1 - eps)
    loss = - np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return loss


# liczymy gradient straty wzgledem predykcji
def bin_cross_entr_grad(y_true, y_prob, eps=1e-15):
    y_prob = np.clip(y_prob, eps, 1 - eps)
    grad = (y_prob - y_true) / (y_prob * (1 - y_prob))
    return (y_prob - y_true)



# Model MLP (list of layers)


class MLP:
    def __init__(self, input_dim, layer_sizes, weight_std=0.01, seed=0, activation='relu'):

        # lista obiektow
        self.layers = []
        self.seed = seed
        rng = np.random.default_rng(seed)
        # lista rozmiarow warstw
        dims = [input_dim] + list(layer_sizes)
        for i in range(len(dims) - 1):
            # warstwa liniowa dla kazdej pary
            in_d, out_d = dims[i], dims[i + 1]
            layer = Linear(in_d, out_d, weight_std=weight_std, seed=int(rng.integers(1e9)))
            self.layers.append(layer)
            # dod. fun. aktywacji do wszystkich bez ostat. warstwy
            if i < len(dims) - 2:
                if isinstance(activation, tuple) and len(activation) == 2:
                    func, func_grad = activation
                    self.layers.append(CustActFunction(func, func_grad))
                elif isinstance(activation, str):
                    if activation == 'relu':
                        self.layers.append(ReLU())
                    elif activation == 'sigmoid':
                        self.layers.append(SigmoidAct())
                    else:
                        raise ValueError(f"Unsupported activation string: {activation}")
                else:
                    raise ValueError(f"Unsupported activation type: {type(activation)}")

        # fun. aktyw. ostatniej warstwy
        self.output_activation = SigmoidAct()

    # przeprowadza dane przez wszystkie warstwy
    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        out = self.output_activation.forward(out)
        return out.ravel()

    def backward(self, dLoss_dy):
        dout = dLoss_dy.reshape(-1, 1)
        dout = self.output_activation.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def step(self, lr):
        # Aktualizuje parametry (W, b) tylko w warstwach typu Linear
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.step(lr)

    def params_norm(self):
        s = 0.0
        for layer in self.layers:
            if isinstance(layer, Linear):
                s += np.sum(np.abs(layer.W)) + np.sum(np.abs(layer.b))
        return s


def data_loader(normalize=True, random_state=42):
    hd = fetch_ucirepo(id=45)
    X = hd.data.features
    y = hd.data.targets['num']
    # binaryzacja
    y_bin = (y > 0).astype(int)
    dataset = pd.concat([X, y_bin.rename('target')], axis=1)
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    dataset[categorical_cols] = dataset[categorical_cols].fillna('NaN').astype(str)
    # one-hot encoding
    dataset_enc = pd.get_dummies(dataset, columns=categorical_cols, drop_first=False)

    X_enc = dataset_enc.drop('target', axis=1)
    y_enc = dataset_enc['target'].values.astype(int)
    feature_names = X_enc.columns.tolist()
    X_vals = X_enc.values.astype(float)
    if normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_vals)
        return X_scaled, y_enc, feature_names, scaler
    else:
        return X_vals, y_enc, feature_names, None


def compute_metrics(y_true, probs, threshold=0.5):
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


def train_model(model, X_train, y_train, X_val=None, y_val=None,
                lr=0.01, batch_size=32, max_epochs=200, tol=1e-6, verbose=False):
    m, n = X_train.shape
    loss_history = []
    val_history = []
    rng = np.random.default_rng(0)


    for epoch in range(1, max_epochs + 1):
        # mieszanie danych w kaz. epoce na poczatku
        perm = rng.permutation(m)
        X_sh = X_train[perm]
        y_sh = y_train[perm]
        # iteracja po mini-batches
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            xb = X_sh[start:end]
            yb = y_sh[start:end]
            # forward
            probs = model.forward(xb)  # shape (batch,)
            # obliczanie gradientu bledu
            dLoss_dp = (probs - yb)
            # backward
            model.backward(dLoss_dp)
            # aktualizacja wag
            model.step(lr)
        # koniec epoki. Obliczanie i zapisywanie straty.
        probs_train = model.forward(X_train)
        loss = bin_cross_entr_loss(y_train, probs_train)
        loss_history.append(loss)
        if X_val is not None:
            probs_val = model.forward(X_val)
            val_loss = bin_cross_entr_loss(y_val, probs_val)
            val_history.append(val_loss)
        if verbose and (epoch == 1 or epoch % 10 == 0):
            msg = f"Epoch {epoch:4d}: train_loss={loss:.6f}"
            if X_val is not None:
                msg += f", val_loss={val_loss:.6f}"
            print(msg)
        # kryt. zatrzymania. Jesli zm. tolerancji mniejsza niż tolerancja to konczy trening
        if epoch > 1 and abs(loss_history[-2] - loss_history[-1]) < tol:
            if verbose:
                print(
                    f"Converged after {epoch} epochs: loss change {abs(loss_history[-2] - loss_history[-1]):.2e} < tol {tol}")
            break
    return {'model': model, 'loss_history': loss_history, 'val_history': val_history, 'epochs': epoch}


# tworzy model, trenuje go, liczy metr. na tren i test.
def run_experiment(hidden_sizes=None, n_layers=1, lr=0.01, weight_std=0.01,
                   normalize=True, activation: Union[str, Tuple[Callable, Callable]] = 'relu', batch_size=32,
                   max_epochs=200, tol=1e-6, seed=0):
    if hidden_sizes is None:
        hidden_sizes = [16]
    X, y, feat_names, scaler = data_loader(normalize=normalize, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    input_dim = X_train.shape[1]
    layer_sizes = []
    for _ in range(n_layers):
        layer_sizes.append(hidden_sizes[0])
    layer_sizes.append(1)
    model = MLP(input_dim, layer_sizes, weight_std=weight_std, seed=seed, activation=activation)
    start = time.time()
    history = train_model(model, X_train, y_train, X_val=X_test, y_val=y_test, lr=lr, batch_size=batch_size,
                          max_epochs=max_epochs, tol=tol, verbose=False)
    elapsed = time.time() - start

    probs_train = model.forward(X_train)
    probs_test = model.forward(X_test)
    train_metrics = compute_metrics(y_train, probs_train)
    test_metrics = compute_metrics(y_test, probs_test)
    result = {
        'hidden_size': hidden_sizes[0],
        'n_layers': n_layers,
        'lr': lr,
        'weight_std': weight_std,
        'normalize': normalize,
        'activation': activation,
        'batch_size': batch_size,
        'epochs': history['epochs'],
        'train_loss': history['loss_history'][-1] if history['loss_history'] else None,
        'val_loss': history['val_history'][-1] if history['val_history'] else None,
        'train_acc': train_metrics['accuracy'],
        'test_acc': test_metrics['accuracy'],
        'train_precision': train_metrics['precision'],
        'test_precision': test_metrics['precision'],
        'train_recall': train_metrics['recall'],
        'test_recall': test_metrics['recall'],
        'train_f1': train_metrics['f1'],
        'test_f1': test_metrics['f1'],
        'train_auc': train_metrics['auc'],
        'test_auc': test_metrics['auc'],
        'time_s': elapsed
    }
    print(result)
    return result, history, model, (X_train, X_test, y_train, y_test)


# A lot of experiments with different hiperparameters and
# activation functions


def multi_testing():
    hidden_sizes = [8, 16, 32]
    n_layers_list = [1, 2]
    lrs = [0.01, 0.05, 0.1]
    stds = [0.01, 0.1, 0.5]
    normalize = [True, False]
    activations = ['relu', 'sigmoid', 'custom']

    custom_func = lambda x: x / (1 + np.abs(x))
    custom_grad = lambda x: 1 / (1 + np.abs(x)) ** 2

    results = []

    for activation_name in activations:
        if activation_name == 'custom':
            activation = (custom_func, custom_grad)
        else:
            activation = activation_name

        for normalize in normalize:
            for std in stds:
                for n_layers in n_layers_list:
                    for h in hidden_sizes:
                        for lr in lrs:
                            res, hist, model, data_split = run_experiment(
                                hidden_sizes=[h],
                                n_layers=n_layers,
                                lr=lr,
                                weight_std=std,
                                normalize=normalize,
                                activation=activation,
                                batch_size=32,
                                max_epochs=300,
                                tol=1e-6,
                                seed=0
                            )

                            results.append(res)
                            print(
                                f"done: norm={normalize}, std={std}, layers={n_layers}, h={h}, lr={lr}, activation={activation_name}\n"
                                f" --> test_acc={res['test_acc']:.3f}, test_auc={res['test_auc']:.3f}"
                            )

    df = pd.DataFrame(results)
    df.to_csv("experiments_results.csv", index=False)
    print("Wyniki zapisane do experiments_results.csv")
    return df


if __name__ == "__main__":


    # MY OWN CUSTOM FUNCTION f = x/(1+|x|)
    #
    # custom_func = lambda x: x / (1 + np.abs(x))
    # custom_grad = lambda x: 1 / (1 + np.abs(x)) ** 2
    #
    # res, hist, model, splits = run_experiment(
    #     hidden_sizes=[32],
    #     n_layers=1,
    #     lr=0.05,
    #     weight_std=0.1,
    #     normalize=True,
    #     activation=(custom_func, custom_grad),
    #     batch_size=32,
    #     max_epochs=300,
    #     tol=1e-6,
    #     seed=0
    # )
    #
    # print("---------------- CUSTOM FUNCTION RESULTS ---------")
    # for k, v in res.items():
    #     if isinstance(v, float):
    #         print(f"{k}: {v:.4f}")
    #     else:
    #         print(f"{k}: {v}")


# running test with different learning rate, epochs etc.
# this function below works like big benchmark for testing
#     multi_testing()

    # RELU OR SIGMOID FUNCTION

    res, hist, model, splits = run_experiment(
        hidden_sizes=[32],
        n_layers=1,
        lr=0.05,
        weight_std=0.1,
        normalize=True,
        activation="relu",
        batch_size=32,
        max_epochs=300,
        tol=1e-6,
        seed=0
    )

    for k, v in res.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
