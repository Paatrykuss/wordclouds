import numpy as np
import math
from scipy.stats import norm
from sklearn.base import BaseEstimator

# =========================================================================
# SEKCJA: ORYGINALNY SZKIELET (z pliku naive_bayes.py.txt)
# =========================================================================
class NaiveBayesNominal:
    def __init__(self):
        # Te zmienne były zdefiniowane w Twoim pliku startowym jako pola klasy
        self.classes_ = None
        self.model = dict()  
        self.y_prior = []    

    # =====================================================================
    # SEKCJA: MOJA IMPLEMENTACJA (Logika wpisana przeze mnie)
    # =====================================================================
    def fit(self, X, y):
        """
        Uczenie modelu: wyliczamy P(klasa) oraz P(cecha|klasa).
        """
        # [ORYGINAŁ/NUMPY] Pobieramy unikalne klasy (np. 0 i 1)
        self.classes_ = np.unique(y)
        liczba_przykladow = len(y)
        self.y_prior = []

        for klasa in self.classes_:
            # [LOGIKA WPISANA PRZEZE MNIE]
            # 1. Obliczamy P(klasa) - Prawdopodobieństwo a priori
            wiersze_klasy = X[y == klasa]
            p_klasy = len(wiersze_klasy) / liczba_przykladow
            self.y_prior.append(p_klasy)

            # 2. Obliczamy P(cecha | klasa) dla każdej kolumny
            for i in range(X.shape[1]):
                # [ZACIĄGNIĘTE Z KONTEKSTU DANYCH] 
                # Musimy znać wszystkie możliwe wartości cechy, żeby Laplace działał poprawnie
                unikalne_wartosci_cechy = np.unique(X[:, i])
                
                for wartosc in unikalne_wartosci_cechy:
                    # [LOGIKA WPISANA PRZEZE MNIE]
                    licznik = np.sum(wiersze_klasy[:, i] == wartosc)
                    
                    # MATEMATYKA Z PDF (Zadanie 1): Wygładzanie Laplace'a
                    # Wzór: (wystąpienia + 1) / (liczba wierszy w klasie + liczba wariantów cechy)
                    p_warunkowe = (licznik + 1) / (len(wiersze_klasy) + len(unikalne_wartosci_cechy))
                    
                    # Zapisujemy do słownika (klasa, kolumna, wartość)
                    self.model[(klasa, i, wartosc)] = p_warunkowe

    # =====================================================================
    # SEKCJA: MOJA IMPLEMENTACJA (Obliczenia prawdopodobieństwa)
    # =====================================================================
    def predict_proba(self, X):
        """
        Zwraca macierz numpy z prawdopodobieństwami.
        Wymóg z PDF: "Funkcja powinna zwracać wektor (numpy)"
        """
        wyniki = []
        for wiersz in X:
            szanse_klas = []
            for idx, klasa in enumerate(self.classes_):
                # [LOGIKA WPISANA PRZEZE MNIE]
                # Startujemy od prawdopodobieństwa samej klasy P(Y)
                p = self.y_prior[idx]
                
                # Mnożymy przez P(Xi | Y) dla każdej cechy w wierszu
                for i, wartosc in enumerate(wiersz):
                    # Pobieramy z modelu. 1e-6 to "bezpiecznik" dla wartości, których nie było w treningu
                    p *= self.model.get((klasa, i, wartosc), 1e-6)
                
                szanse_klas.append(p)
            
            # [LOGIKA WPISANA PRZEZE MNIE]
            # Normalizacja: Bayes daje nam wartości proporcjonalne, 
            # musimy je podzielić przez sumę, żeby dostać % (np. 0.8 i 0.2)
            suma = sum(szanse_klas)
            wyniki.append([s / suma for s in szanse_klas])
            
        return np.array(wyniki)

    # =====================================================================
    # SEKCJA: MOJA IMPLEMENTACJA (Dostosowanie pod utils/evaluate.py)
    # =====================================================================
    def predict(self, X):
        """
        Zwraca przewidywaną klasę (0 lub 1).
        Wywoływane przez funkcję test_model() w pliku utils/evaluate.py
        """
        # Korzystamy z wyliczonych wcześniej prawdopodobieństw
        proby = self.predict_proba(X)
        
        # [ZACIĄGNIĘTE Z NUMPY] np.argmax wybiera indeks największej wartości
        indeksy_najlepszych = np.argmax(proby, axis=1)
        
        # Zwracamy rzeczywiste etykiety klas (zgodne z tym co wczytał load.py)
        return self.classes_[indeksy_najlepszych]