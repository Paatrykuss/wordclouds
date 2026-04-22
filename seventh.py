import numpy as np
import sys
import os

# [ZACIĄGNIĘCIE ŚCIEŻKI] - Ustawienie ścieżek, aby Python widział folder 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# [ZACIĄGNIĘCIE Z PLIKU load.py i evaluate.py]
from utils.load import load_flu
from utils.evaluate import test_model

# =========================================================================
# KLASA PRZENIESIONA ZE SZKIELETU I UZUPEŁNIONA PRZEZE MNIE (Logika Bayesa)
# =========================================================================
class NaiveBayesNominal:
    def __init__(self):
        # [ORYGINAŁ ZE SZKIELETU]
        self.classes_ = None
        self.model = dict()
        self.y_prior = []

    def fit(self, X, y):
        # [WPISANE PRZEZE MNIE - LOGIKA UCZENIA]
        self.classes_ = np.unique(y)
        n_samples = len(y)
        self.y_prior = []

        for klasa in self.classes_:
            X_c = X[y == klasa]
            self.y_prior.append(len(X_c) / n_samples)

            for i in range(X.shape[1]):
                # Wyznaczenie unikalnych wartości dla wygładzania Laplace'a
                unikalne_wartosci = np.unique(X[:, i])
                for wartosc in unikalne_wartosci:
                    licznik = np.sum(X_c[:, i] == wartosc)
                    # MATEMATYKA Z PDF: Wygładzanie Laplace'a
                    p_warunkowe = (licznik + 1) / (len(X_c) + len(unikalne_wartosci))
                    self.model[(klasa, i, wartosc)] = p_warunkowe

    def predict_proba(self, X):
        # [WPISANE PRZEZE MNIE - LOGIKA OBLICZEŃ]
        wyniki = []
        for wiersz in X:
            szanse_klas = []
            for idx, klasa in enumerate(self.classes_):
                p = self.y_prior[idx]
                for i, wartosc in enumerate(wiersz):
                    p *= self.model.get((klasa, i, wartosc), 1e-6)
                szanse_klas.append(p)
            
            # Normalizacja do 1.0 (wymóg prawdopodobieństwa)
            suma = sum(szanse_klas)
            wyniki.append([s / suma for s in szanse_klas])
        return np.array(wyniki)

    def predict(self, X):
        # [WPISANE PRZEZE MNIE]
        proby = self.predict_proba(X)
        return self.classes_[np.argmax(proby, axis=1)]

# =========================================================================
# SEKCJA WYKONAWCZA (To co ma się zadziać po uruchomieniu)
# =========================================================================

# 1. [WYKORZYSTANIE FUNKCJI Z load.py]
# Ta funkcja wczytuje grypa1-train.csv, grypa1-test.csv i robi konwersję (tak/nie -> 1/0)
X_flu_train, y_flu_train, X_flu_test = load_flu()

# 2. Inicjalizacja i nauka modelu na zbiorze treningowym
nb = NaiveBayesNominal()
nb.fit(X_flu_train, y_flu_train)

# 3. [WYKORZYSTANIE FUNKCJI Z evaluate.py]
# Wypisanie wyników w formacie określonym w komentarzu na dole pliku
test_model(nb, X_flu_train, y_flu_train) # Wynik dla treningowego (Ground Truth)
test_model(nb, X_flu_test)               # Wynik dla testowego (Predicted)

# Ostateczne wypisanie wyniku (zgodnie z komentarzem w pliku):
# ground truth:
# [0 1 1 1 0 1 0 1]
# predicted:
# [1 1 1 1 0 1 0 1]
# accuracy:
# 0.875
# predicted:
# [1]