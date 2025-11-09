import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from ucimlrepo import fetch_ucirepo


heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
#  X cechy
#  y etykiety
X = heart_disease.data.features
y = heart_disease.data.targets


print(heart_disease.metadata)


print(heart_disease.variables)


dataset = pd.concat([X, y], axis=1)
# print(dataset.info())

# Class balancing
print(dataset['num'].value_counts())

# BINARY dividing (0 health or 1) 1 is sum of 1-4 elements
# if num > 0 przypisuje 1 (chora)
# if num == 0 przypisuje 0 (zdrowa)
dataset['target_binary'] = dataset['num'].apply(lambda x: 1 if x > 0 else 0)
print(dataset['target_binary'].value_counts())


# P1
# plot showing (binary)
sns.countplot(x='target_binary', data=dataset)
plt.title('Rozkład klas (0 = zdrowy, 1 = zsumowane stopnie choroby)')
plt.xlabel('Kolumna num 0 - zdrowy, 1 zsumowani chorzy')
plt.ylabel('Liczba próbek', fontsize=10)

for i, v in enumerate(dataset['num'].value_counts().sort_index()):
    plt.text(i, v + 1, str(v), ha='center', fontweight='bold')

plt.show()

# -----------------------------------------

# 0 - healthy 1-4 different diseases

# Konwersja 'num' na typ kategoryczny (string)
dataset['num'] = dataset['num'].astype(str)

plt.figure(figsize=(7, 5))
sns.countplot(x='num', data=dataset, order=['0', '1', '2', '3', '4'], palette='viridis')

plt.title('Rozkład klas (0–4) w zbiorze Heart Disease', fontsize=13)
plt.xlabel('Wartość kolumny num (0 = brak choroby, 1–4 = różne poziomy choroby)', fontsize=10)
plt.ylabel('Liczba próbek', fontsize=10)

# Dodanie wartosci nad slupkami
for i, v in enumerate(dataset['num'].value_counts().sort_index()):
    plt.text(i, v + 1, str(v), ha='center', fontweight='bold')

plt.show()


print(dataset['num'].value_counts().sort_index())

# P2
# srednia i odchylenia standardowe cech liczbowych
# mean - srednia kazdej cechy liczbowej
# std - odchylenie standardowe
# min, max - skrajne wartosci

numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
print("-----P2------")
print(dataset[numeric_cols].describe().T[['mean', 'std', 'min', 'max']])

# P3
#  czy rozklad  cech liczbowych jest zblizony do  normalnych (test shapiro - wilka)
print("-----P3-----")
for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
    stat, p = shapiro(dataset[col].dropna())
    print(f"{col}: p-value = {p:.4f}")

    # rysowanie wykresu (histogramu) dla każdej z cech i dodaje krzywa gestosci
    # (aby zobaczyć wizualnie czy jest to rozklad normalny)

    sns.histplot(dataset[col].dropna(), kde=True)
    plt.title(f'Histogram {col}')
    plt.show()


# P4
#  Analiza cech kategorycznych

categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

print("-----P4------")
for col in categorical_cols:
    # liczebnosc kazdej kategorii w okreslonej kolumnie
    print(f"\n{col}:\n{dataset[col].value_counts(dropna=False)}")
    dataset[col] = dataset[col].astype(str)
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=dataset, palette='crest',
                  order=sorted(dataset[col].unique()))

    plt.title(f'Rozkład kategorii dla {col}', fontsize=12)
    plt.xlabel(col, fontsize=10)
    plt.ylabel('Liczba próbek', fontsize=10)

    # Dodanie liczby probek nad slupkami
    counts = dataset[col].value_counts().sort_index()
    for i, v in enumerate(counts):
        plt.text(i, v + 1, str(v), ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

# Lista kolumn kategorycznych
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']


dataset_encoded = pd.get_dummies(dataset, columns=categorical_cols, drop_first=False)


print("Rozmiar macierzy po kodowaniu:", dataset_encoded.shape)

# Wydzielanie macierzy cech X i wektora celu y
X_encoded = dataset_encoded.drop(['num', 'target_binary'], axis=1)
y_encoded = dataset_encoded['num']
print("Przykład macierzy cech (pierwsze 5 wierszy):")
print(X_encoded.head())





