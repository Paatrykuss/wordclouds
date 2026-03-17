import re
import nltk
from nltk.corpus import stopwords

# Pobieram bazę słów stop (uruchomić tylko raz)
nltk.download('stopwords')

# Otwieram plik z dramatem
plik = open('RomeoJuliet.txt', 'r', encoding='utf-8')
tekst = plik.read()
plik.close()

# ==========================================
# a) Oczyszczenie pliku
# ==========================================

# Usuwam didaskalia (to co jest w nawiasach kwadratowych)
tekst = re.sub(r'\[.*?\]', ' ', tekst)

# Usuwam nagłówki Act i Scene (prosta zamiana)
tekst = re.sub(r'ACT [IVXLC]+', ' ', tekst)
tekst = re.sub(r'Scene [IVXLC]+', ' ', tekst)
tekst = tekst.replace('THE PROLOGUE', ' ')

# Ręczne usuwanie znaków interpunkcyjnych
znaki_do_usuniecia = [',', '.', '!', '?', ';', ':', '-', "'", '"', '(', ')']
for znak in znaki_do_usuniecia:
    tekst = tekst.replace(znak, ' ')

# Zamiana na małe litery
tekst = tekst.lower()

# Tokenizacja (split sam dzieli po spacjach i usuwa podwójne spacje)
wszystkie_slowa = tekst.split()

# ==========================================
# b) Usuwanie stop words i krótkich słów
# ==========================================

angielskie_stop_words = stopwords.words('english')
# Dodaję ręcznie dziwne stare słowa z Szekspira, bo psuły wyniki
stare_slowa = ['thou', 'thy', 'thee', 'hath', 'doth', 'art', 'tis']
angielskie_stop_words.extend(stare_slowa)

dobre_slowa = []

# Przechodzę przez wszystkie słowa i sprawdzam warunki
for slowo in wszystkie_slowa:
    if slowo not in angielskie_stop_words:
        if len(slowo) > 2:
            dobre_slowa.append(slowo)

# ==========================================
# c) Przygotowanie pod chmurę wyrazów
# ==========================================

# Zapisuję przefiltrowane słowa do nowego pliku, żeby wkleić je na stronie
plik_chmura = open('dane_do_chmury.txt', 'w', encoding='utf-8')
for slowo in dobre_slowa:
    plik_chmura.write(slowo + " ")
plik_chmura.close()

print("Słowa do chmury zostały zapisane w pliku 'dane_do_chmury.txt'.")

# ==========================================
# d) Najczęściej powtarzane słowo
# ==========================================

# Ręczne liczenie słów za pomocą słownika
slownik_wystapien = {}

for slowo in dobre_slowa:
    if slowo in slownik_wystapien:
        slownik_wystapien[slowo] = slownik_wystapien[slowo] + 1
    else:
        slownik_wystapien[slowo] = 1

# Szukanie tego, które wystąpiło najwięcej razy
najczestsze_slowo = ""
najwieksza_liczba = 0

for slowo, liczba in slownik_wystapien.items():
    if liczba > najwieksza_liczba:
        najwieksza_liczba = liczba
        najczestsze_slowo = slowo

print("--- Wyniki ---")
print("Najczęściej powtarzane słowo to:", najczestsze_slowo)
print("Pojawiło się:", najwieksza_liczba, "razy w całym dramacie.")