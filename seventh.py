import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.load import load_flu
from utils.evaluate import test_model

class NaiveBayesNominal:
    def __init__(self):
        self.classes_ = None
        self.model = dict()
        self.y_prior = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples = len(y)
        self.y_prior = []

        for klasa in self.classes_:
            X_c = X[y == klasa]
            self.y_prior.append(len(X_c) / n_samples)

            for i in range(X.shape[1]):
                unikalne_wartosci = np.unique(X[:, i])
                for wartosc in unikalne_wartosci:
                    licznik = np.sum(X_c[:, i] == wartosc)
                    p_warunkowe = (licznik + 1) / (len(X_c) + len(unikalne_wartosci))
                    self.model[(klasa, i, wartosc)] = p_warunkowe

    def predict_proba(self, X):
        wyniki = []
        for wiersz in X:
            szanse_klas = []
            for idx, klasa in enumerate(self.classes_):
                p = self.y_prior[idx]
                for i, wartosc in enumerate(wiersz):
                    p *= self.model.get((klasa, i, wartosc), 1e-6)
                szanse_klas.append(p)
            
            suma = sum(szanse_klas)
            if suma == 0:
                wyniki.append([1.0 / len(self.classes_)] * len(self.classes_))
            else:
                wyniki.append([s / suma for s in szanse_klas])
        return np.array(wyniki)

    def predict(self, X):
        proby = self.predict_proba(X)
        return self.classes_[np.argmax(proby, axis=1)]

X_flu_train, y_flu_train, X_flu_test = load_flu()

nb = NaiveBayesNominal()
nb.fit(X_flu_train, y_flu_train)

test_model(nb, X_flu_train, y_flu_train)
test_model(nb, X_flu_test)
