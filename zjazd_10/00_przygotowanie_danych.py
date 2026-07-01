import pandas as pd  # Importuje bibliotekę pandas, która służy do pracy z danymi tabelarycznymi.
from sklearn.datasets import load_iris  # Importuje funkcję load_iris, która wczytuje gotowy zbiór danych Iris.
from sklearn.model_selection import train_test_split  # Importuje funkcję dzielącą dane na zbiór treningowy i testowy.

zbior = load_iris(as_frame=True)  # Wczytuje zbiór danych Iris; as_frame=True zwraca dane w formie tabel pandas.

dane = zbior.frame.copy()  # Tworzy kopię pełnej tabeli danych, zawierającej cechy oraz kolumnę target.

print("Pierwsze 10 wierszy danych:")
print(dane.head(10))  # Wyświetla pierwsze 10 wierszy tabeli danych.

print("Nazwy kolumn:")
print(dane.columns)  # Wyświetla nazwy wszystkich kolumn w tabeli.

print("Informacje o danych:")
print(dane.info())  # Wyświetla informacje o tabeli, takie jak liczba wierszy, typy danych i brakujące wartości.

print("Podstawowe statystki:")
print(dane.describe())  # Wyświetla podstawowe statystyki dla kolumn liczbowych, np. średnią, minimum, maksimum i kwartyle.

print("Liczba obserwacji w każdej klasie:")
print(dane["target"].value_counts().sort_index())  # Liczy liczbę obserwacji w każdej klasie i sortuje wynik według numeru klasy.

print("Nazwy klas:")
for numer, nazwa in enumerate(zbior.target_names):  # Przechodzi po nazwach klas i przypisuje im kolejne numery.
    print(numer, "=", nazwa)

X = zbior.data  # Zapisuje do zmiennej X dane wejściowe, czyli cechy opisujące kwiaty.
y = zbior.target  # Zapisuje do zmiennej y etykiety klas, czyli gatunki irysów.

print("Kształt X:", X.shape)  # Wyświetla kształt X, czyli liczbę wierszy i kolumn z cechami.
print("Kształt y:", y.shape)  # Wyświetla kształt y, czyli liczbę etykiet klas.

X_train, X_test, y_train, y_test = train_test_split(  # Dzieli dane na zbiór treningowy i testowy.
    X,  # Przekazuje cechy, czyli dane wejściowe.
    y,  # Przekazuje etykiety klas.
    test_size=0.25,  # Określa, że 25% danych zostanie przeznaczone na zbiór testowy.
    random_state=42,  # Ustawia stałą losowość, aby wynik podziału był powtarzalny.
    stratify=y,  # Zachowuje podobne proporcje klas w zbiorze treningowym i testowym.
)  # Kończy wywołanie funkcji train_test_split.

print("Rozmiar zbioru treningowego:", X_train.shape)  # Wyświetla rozmiar zbioru treningowego.
print("Rozmiar zbioru testowego:", X_test.shape)  # Wyświetla rozmiar zbioru testowego.

print("Wnioski:")
print("1. Każdy wiersz to jedna obserwacja, czyli jeden kwiat.")
print("2. Kolumny z pomiarami to cechy X.")
print("3. Kolumna target to etykieta y, czyli gatunek irysa.")
print("4. Dane dzielimy na treningowe i testowe, aby sprawdzić model na nowych przykładach.")