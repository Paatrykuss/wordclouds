import numpy as np
import math
from scipy.stats import norm
from sklearn.base import BaseEstimator

# ==========================================================
# [ORYGINAŁ] - Ta struktura i nazwa klasy była w Twoim pliku
# ==========================================================
class NaiveBayesNominal:
    def __init__(self):
        # Te pola były w szkielecie, przygotowane pod dane
        self.classes_ = None
        self.model = dict()
        self.y_prior = []

    # ==========================================================
    # [MOJE] - Poniżej znajduje się logika, którą dopisałem
    # ==========================================================
    def fit(self, X, y):
        """Uczenie modelu - wypełniamy pola self.model i self.y_prior"""
        self.classes_ = np.unique(y)
        liczba_przykladow = len(y)

        for klasa in self.classes_:
            # Wybieramy wiersze dla danej klasy (np. tylko te, gdzie grypa = tak)
            wiersze_klasy = X[y == klasa]
            
            # Liczymy P(klasa) i dodajemy do listy (zaciągnięte do self.y_prior)
            self.y_prior.append(len(wiersze_klasy) / liczba_przykladow)

            # Liczymy prawdopodobieństwa dla każdej cechy (kolumny)
            for i in range(X.shape[1]):
                unikalne_wartosci = np.unique(X[:, i])
                for wartosc in unikalne_wartosci:
                    # Licznik wystąpień danej wartości cechy w danej klasie
                    licznik = np.sum(wiersze_klasy[:, i] == wartosc)
                    
                    # LOGIKA: Wygładzanie Laplace'a (zgodnie z zadaniem z PDF)
                    p_warunkowe = (licznik + 1) / (len(wiersze_klasy) + len(unikalne_wartosci))
                    
                    # Zapisujemy do słownika 'model' (zaciągnięte do self.model)
                    self.model[(klasa, i, wartosc)] = p_warunkowe

    # ==========================================================
    # [MOJE] - Implementacja obliczeń prawdopodobieństwa
    # ==========================================
    def predict_proba(self, X):
        wyniki = []
        for wiersz in X:
            szanse_klas = []
            for idx, klasa in enumerate(self.classes_):
                # Startujemy od P(klasa)
                p = self.y_prior[idx]
                
                # Mnożymy przez P(cecha|klasa) dla każdego objawu
                for i, wartosc in enumerate(wiersz):
                    # Pobieramy wartość ze słownika, który wypełniliśmy w fit()
                    p *= self.model.get((klasa, i, wartosc), 1e-6)
                
                szanse_klas.append(p)
            
            # Normalizacja, żeby suma szans wynosiła 100% (1.0)
            suma = sum(szanse_klas)
            wyniki.append([s / suma for s in szanse_klas])
            
        return np.array(wyniki)

    # ==========================================================
    # [MOJE] - Wybór końcowej odpowiedzi
    # ==========================================================
    def predict(self, X):
        # Wykorzystujemy funkcję predict_proba, którą napisałem wyżej
        proby = self.predict_proba(X)
        
        # Wybieramy klasę z najwyższym wynikiem (np. jeśli 0.8 dla Grypy, to wybieramy Grypę)
        indeksy = np.argmax(proby, axis=1)
        return self.classes_[indeksy]

# ==========================================================
# [ORYGINAŁ] - Dalsze klasy (Gaussian/NumNom) zostawiam jako raise NotImplementedError
# ==========================================================
class NaiveBayesGaussian:
    def __init__(self):
        raise NotImplementedError # Oryginalny szkielet