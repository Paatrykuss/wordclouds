import numpy as np


class NaiwnyKlasyfikatorBayesa:
    def __init__(self):
        # Słowniki na dane, których model "nauczy się" w funkcji fit
        self.szanse_klas = {}  # P(Klasa)
        self.szanse_warunkowe = {}  # P(Cecha | Klasa)
        self.lista_klas = []

    def fit(self, X, y):
        """Uczenie modelu na podstawie danych """
        liczba_przykladow = X.shape[0]
        liczba_cech = X.shape[1]
        self.lista_klas = np.unique(y)

        # 1. Liczymy prawdopodobieństwo a priori dla każdej klasy (np. Chory/Zdrowy)
        for klasa in self.lista_klas:
            ile_razy_klasa = np.sum(y == klasa)
            self.szanse_klas[klasa] = ile_razy_klasa / liczba_przykladow

        # Inicjalizujemy słownik dla cech
        for i in range(liczba_cech):
            self.szanse_warunkowe[i] = {}
            for klasa in self.lista_klas:
                self.szanse_warunkowe[i][klasa] = {}

        # 2. Liczymy prawdopodobieństwo każdej wartości cechy dla danej klasy
        for i in range(liczba_cech):
            for klasa in self.lista_klas:
                # Wybieramy tylko te wiersze z X, gdzie klasa w y się zgadza
                dane_dla_klasy = X[y == klasa]
                ile_wierszy_klasy = len(dane_dla_klasy)

                # Sprawdzamy jakie wartości przyjmuje ta cecha (np. Gorączka: Tak/Nie)
                wartosci, licznik = np.unique(dane_dla_klasy[:, i], return_counts=True)

                for w, l in zip(wartosci, licznik):
                    # Zapisujemy: P(Wartość | Klasa)
                    self.szanse_warunkowe[i][klasa][w] = l / ile_wierszy_klasy

        return self

    def predict_proba(self, X):
        """Zwraca prawdopodobieństwa dla każdej klasy """
        wyniki = []

        for wiersz in X:
            prawdopodobienstwa_wiersza = []

            for klasa in self.lista_klas:
                # Zaczynamy od szansy samej klasy
                wynik_klasy = self.szanse_klas[klasa]

                # Mnożymy przez szanse każdej cechy w tym wierszu
                for i in range(len(wiersz)):
                    wartosc_cechy = wiersz[i]

                    if wartosc_cechy in self.szanse_warunkowe[i][klasa]:
                        wynik_klasy *= self.szanse_warunkowe[i][klasa][wartosc_cechy]
                    else:
                        # Jeśli cecha nie wystąpiła w treningu, szansa zeruje się
                        wynik_klasy *= 0

                prawdopodobienstwa_wiersza.append(wynik_klasy)

            # Normalizacja, żeby suma prawdopodobieństw wynosiła 100% (1.0)
            suma = sum(prawdopodobienstwa_wiersza)
            if suma > 0:
                prawdopodobienstwa_wiersza = [p / suma for p in prawdopodobienstwa_wiersza]

            wyniki.append(prawdopodobienstwa_wiersza)

        return np.array(wyniki)

    def predict(self, X):
        """Zwraca najbardziej prawdopodobną klasę """
        szanse = self.predict_proba(X)

        # Wybieramy indeks tej klasy, która ma największą szansę
        indeksy_najlepszych = np.argmax(szanse, axis=1)

        # Zamieniamy indeksy na nazwy klas (np. 0 -> 'Zdrowy')
        return np.array([self.lista_klas[i] for i in indeksy_najlepszych])


# --- TESTOWANIE NA ZBIORZE "GRYPA" 

# Przykładowe dane nominalne jako liczby (np. Gorączka: 0-Nie, 1-Tak | Kaszel: 0-Nie, 1-Tak)
# X: [Gorączka, Kaszel, Ból mięśni]
X_treningowe = np.array([
    [1, 1, 1],  # Chory
    [1, 0, 1],  # Chory
    [0, 1, 0],  # Zdrowy
    [0, 0, 0],  # Zdrowy
    [1, 1, 0]  # Chory
])

# y: 1 - Chory na grypę, 0 - Zdrowy
y_treningowe = np.array([1, 1, 0, 0, 1])

# Tworzymy klasyfikator
model = NaiwnyKlasyfikatorBayesa()

# Uczymy model
model.fit(X_treningowe, y_treningowe)

# Testujemy na nowym pacjencie (ma gorączkę, nie ma kaszlu, ma ból mięśni)
nowy_pacjent = np.array([[1, 0, 1]])

procenty = model.predict_proba(nowy_pacjent)
wynik = model.predict(nowy_pacjent)

print(f"Prawdopodobieństwo (0-Zdrowy, 1-Chory): {procenty}")
print(f"Diagnoza: {'Chory' if wynik[0] == 1 else 'Zdrowy'}")