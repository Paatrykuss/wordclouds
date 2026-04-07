import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Wczytanie danych
data_train = pd.read_csv('grypa-train.csv')
data_test = pd.read_csv('grypa-test.csv')

# 2. Konwersja danych jakościowych na liczbowe
binary_mapping = {'tak': 1, 'nie': 0}

# Mapujemy kolumny dreszcze, katar, goraczka i grypa
cols_to_map = ['# dreszcze', 'katar', 'goraczka', 'grypa']
for col in cols_to_map:
    if col in data_train.columns:
        data_train[col] = data_train[col].replace(binary_mapping)

# To samo dla zbioru testowego (bez kolumny grypa)
cols_test = ['#dreszcze', 'katar', 'goraczka']
for col in cols_test:
    if col in data_test.columns:
        data_test[col] = data_test[col].replace(binary_mapping)

print("Krok 4 - Dane po częściowej konwersji (ból głowy nadal tekstem):")
print(data_train)

# 3. Podział na cechy i klasę oraz pełna konwersja na NumPy (KROK 5)
# Tutaj musimy już zamienić "bol_glowy" na liczby, żeby sklearn zadziałał
headache_mapping = {'nie': 0, 'sredni': 1, 'duzy': 2}
data_train['bol_glowy'] = data_train['bol_glowy'].replace(headache_mapping)
data_test['bol_glowy'] = data_test['bol_glowy'].replace(headache_mapping)

# Przekształcenie do macierzy numpy
X_train = data_train[['# dreszcze', 'katar', 'bol_glowy', 'goraczka']].values
y_train = data_train['grypa'].values
X_test = data_test[['#dreszcze', 'katar', 'bol_glowy', 'goraczka']].values

print("\nKrok 5 - Macierz X_train (teraz wszystko jest liczbą):")
print(X_train)

# 4. Trenowanie modelu
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train) #

# 5. Predykcja
ypred_test = dt.predict(X_test)
ypred_train = dt.predict(X_train)

print("\nPredykcja dla testowych:", ypred_test)
print("Predykcja dla treningowych:", ypred_train)
print("Dokładność:", accuracy_score(y_train, ypred_train))