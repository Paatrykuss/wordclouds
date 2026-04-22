import numpy as np
import math
from scipy.stats import norm
from sklearn.base import BaseEstimator

class NaiveBayesNominal:
    def __init__(self):
        self.classes_ = None
        self.model = dict()  # Słownik na prawdopodobieństwa warunkowe P(cecha|klasa)
        self.y_prior = []    # Lista na prawdopodobieństwa a priori klas P(klasa)

    def fit(self, X, y):
        """
        Uczenie modelu: wyliczamy P(klasa) oraz P(cecha|klasa) dla wszystkich cech.
        """
        self.classes_ = np.unique(y)
        liczba_przykladow = len(y)
        self.y_prior = []

        for klasa in self.classes_:
            # 1. Obliczamy P(klasa) - prawdopodobieństwo wystąpienia danej klasy
            wiersze_klasy = X[y == klasa]
            p_klasy = len(wiersze_klasy) / liczba_przykladow
            self.y_prior.append(p_klasy)

            # 2. Obliczamy P(cecha_i = wartosc | klasa) dla każdej kolumny i każdej wartości
            for i in range(X.shape[1]):
                unikalne_wartosci_cechy = np.unique(X[:, i])
                for wartosc in unikalne_wartosci_cechy:
                    licznik = np.sum(wiersze_klasy[:, i] == wartosc)
                    
                    # Wygładzanie Laplace'a (zgodnie ze wzorem z PDF)
                    # (liczba wystąpień + 1) / (liczba wierszy w klasie + liczba możliwych wartości cechy)
                    p_warunkowe = (licznik + 1) / (len(wiersze_klasy) + len(unikalne_wartosci_cechy))
                    
                    # Zapisujemy do modelu: kluczem jest krotka (klasa, kolumna, wartość)
                    self.model[(klasa, i, wartosc)] = p_warunkowe

    def predict_proba(self, X):
        """
        Zwraca macierz numpy z prawdopodobieństwami dla każdej klasy.
        """
        wyniki = []
        for wiersz in X:
            szanse_klas = []
            for idx, klasa in enumerate(self.classes_):
                # Startujemy od prawdopodobieństwa a priori klasy
                p = self.y_prior[idx]
                
                # Mnożymy przez prawdopodobieństwa warunkowe kolejnych cech
                for i, wartosc in enumerate(wiersz):
                    # Pobieramy z modelu, jeśli nie ma takiej wartości w treningu - dajemy małą wartość (epsilon)
                    p *= self.model.get((klasa, i, wartosc), 1e-6)
                
                szanse_klas.append(p)
            
            # Normalizacja, aby suma prawdopodobieństw dla wiersza wynosiła 1.0
            suma = sum(szanse_klas)
            wyniki.append([s / suma for s in szanse_klas])
            
        return np.array(wyniki)

    def predict(self, X):
        """
        Zwraca przewidywaną klasę (0 lub 1).
        """
        proby = self.predict_proba(X)
        indeksy_najlepszych = np.argmax(proby, axis=1)
        return self.classes_[indeksy_najlepszych]

# Pozostałe klasy (Gaussian, NumNom) zostawiamy bez zmian lub z raise NotImplementedError, 
# chyba że zadanie z PDF wymaga ich uzupełnienia w tym samym kroku.