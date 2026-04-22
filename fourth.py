import numpy as np
import math
from scipy.stats import norm
from sklearn.base import BaseEstimator


class NaiveBayesNominal:
    def __init__(self):
        self.classes_ = None
        self.model = dict()  # Tu będziemy trzymać P(cecha | klasa)
        self.y_prior = {}    # Tu będziemy trzymać P(klasa)

    def fit(self, X, y):
        # Ustalamy jakie mamy klasy (np. 0 i 1)
        self.classes_ = np.unique(y)
        liczba_wierszy = len(y)
        liczba_cech = X.shape[1]

        for klasa in self.classes_:
            # Wyciągamy tylko te wiersze, gdzie występuje dana klasa
            wiersze_z_klasa = X[y == klasa]
            
            # Liczymy prawdopodobieństwo a priori danej klasy: P(klasa)
            self.y_prior[klasa] = len(wiersze_z_klasa) / liczba_wierszy

            # Teraz liczymy prawdopodobieństwa warunkowe dla każdej kolumny (cechy)
            for i in range(liczba_cech):
                # Patrzymy jakie wartości może przyjąć ta cecha (np. 0, 1, 2)
                unikalne_wartosci_cechy = np.unique(X[:, i])
                
                for wartosc in unikalne_wartosci_cechy:
                    # Liczymy ile razy dana wartość wystąpiła dla tej konkretnej klasy
                    licznik = np.sum(wiersze_z_klasa[:, i] == wartosc)
                    
                    # Wygładzanie Laplace'a: (liczba wystąpień + 1) / (suma wszystkich + liczba wariantów)
                    # To zapobiega mnożeniu przez zero
                    prawdopodobienstwo = (licznik + 1) / (len(wiersze_z_klasa) + len(unikalne_wartosci_cechy))
                    
                    # Zapisujemy to w słowniku pod kluczem (klasa, kolumna, wartość)
                    self.model[(klasa, i, wartosc)] = prawdopodobienstwo

    def predict_proba(self, X):
        wszystkie_prawdopodobienstwa = []

        for wiersz in X:
            szanse_dla_klas = []
            
            for klasa in self.classes_:
                # Zaczynamy od prawdopodobieństwa samej klasy
                wynik_klasy = self.y_prior[klasa]
                
                # Mnożymy przez prawdopodobieństwa wszystkich cech tego pacjenta
                for i, wartosc in enumerate(wiersz):
                    # Pobieramy wartość z modelu (jeśli cecha nie wystąpiła, dajemy b. małą liczbę)
                    p_warunkowe = self.model.get((klasa, i, wartosc), 1e-6)
                    wynik_klasy *= p_warunkowe
                
                szanse_dla_klas.append(wynik_klasy)
            
            # Normalizacja: dzielimy przez sumę, żeby szanse sumowały się do 1 (100%)
            suma = sum(szanse_dla_klas)
            znormalizowane = [s / suma for s in szanse_dla_klas]
            wszystkie_prawdopodobienstwa.append(znormalizowane)
            
        return np.array(wszystkie_prawdopodobienstwa)

    def predict(self, X):
        # Pobieramy macierz prawdopodobieństw
        prawdopodobienstwa = self.predict_proba(X)
        
        # Wybieramy indeks tej klasy, która ma najwyższą wartość
        najlepsze_indeksy = np.argmax(prawdopodobienstwa, axis=1)
        
        # Zwracamy tablicę z nazwami/numerami tych klas
        return np.array([self.classes_[i] for i in najlepsze_indeksy])

# Resztę klas (Gaussian, NumNom) zostawiamy na razie z NotImplementedError 
# albo uzupełniamy analogicznie, jeśli będą potrzebne na kolejne zadania.