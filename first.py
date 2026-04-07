from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
import pickle

# 1. Wczytanie zbioru danych
digits = datasets.load_digits()

# 2. Wyświetlenie cech i etykiet (opcjonalne, do podglądu)
print("Dane (cechy):", digits.data)
print("Etykiety (target):", digits.target)

# 3. Wizualizacja pierwszej cyfry
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("Pierwsza cyfra w zbiorze")
plt.show()

# 4. Utworzenie i trenowanie modelu (wszystko poza ostatnią próbką)
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])

# 5. Przewidywanie dla ostatniego przykładu
prediction = clf.predict(digits.data[-1:])
print(f"Przewidziana cyfra: {prediction[0]}")
print(f"Rzeczywista cyfra: {digits.target[-1]}")

# Wizualizacja sprawdzająca
plt.figure(2, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title(f"Ostatnia cyfra (Predykcja: {prediction[0]})")
plt.show()

# 6. Serializacja modelu (zapis i odczyt)
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print("Wynik z wczytanego modelu (pickle):", clf2.predict(digits.data[-1:]))