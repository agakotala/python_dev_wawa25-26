import numpy as np  # Importuje bibliotekę NumPy, używaną tutaj do generowania losowych danych.
import pandas as pd  # Importuje bibliotekę pandas, używaną do tworzenia i łączenia tabel z transakcjami.
import matplotlib.pyplot as plt  # Importuje moduł pyplot z biblioteki matplotlib do tworzenia wykresów.
from sklearn.ensemble import IsolationForest  # Importuje model Isolation Forest służący do wykrywania anomalii.

generator = np.random.default_rng(seed=42)  # Tworzy generator liczb losowych z ustalonym ziarnem, aby wyniki były powtarzalne.

normalne_transakcje = pd.DataFrame({  # Tworzy tabelę z przykładowymi normalnymi transakcjami.
    "kwota": generator.normal(loc=120, scale=35, size=300),  # Generuje 300 kwot transakcji wokół średniej 120 z odchyleniem 35.
    "liczba_transakcji_24h": generator.poisson(lam=3, size=300),  # Generuje liczbę transakcji w 24h według rozkładu Poissona ze średnią 3.
    "godzina": generator.integers(low=0, high=24, size=300),  # Generuje losowe godziny transakcji od 0 do 23.
})  # Kończy tworzenie tabeli normalnych transakcji.

podejrzane_transakcje = pd.DataFrame({  # Tworzy tabelę z ręcznie wpisanymi podejrzanymi transakcjami.
    "kwota": [950, 1200, 870, 1500, 30],  # Podaje nietypowe kwoty transakcji, zwykle dużo większe od normalnych.
    "liczba_transakcji_24h": [18, 22, 20, 25, 30],  # Podaje bardzo wysoką liczbę transakcji w ostatnich 24 godzinach.
    "godzina": [2, 3, 1, 4, 0]  # Podaje godziny transakcji, głównie nocne.
})  # Kończy tworzenie tabeli podejrzanych transakcji.

dane = pd.concat(  # Łączy normalne i podejrzane transakcje w jedną tabelę.
    [normalne_transakcje, podejrzane_transakcje],  # Przekazuje listę tabel, które mają zostać połączone.
    ignore_index=True,  # Tworzy nowy indeks od zera, zamiast zachowywać stare indeksy z osobnych tabel.
)  # Kończy łączenie tabel.

dane["kwota"] = dane["kwota"].clip(lower=1)  # Zamienia ewentualne kwoty mniejsze niż 1 na wartość 1.

model = IsolationForest(  # Tworzy model Isolation Forest do wykrywania nietypowych obserwacji.
    contamination=0.03,  # Zakłada, że około 3% danych może być anomaliami, czyli podejrzanymi przypadkami.
    random_state=42,  # Ustawia stałą losowość, aby wyniki modelu były powtarzalne.
)  # Kończy definicję modelu.

dane["wynik_modelu"] = model.fit_predict(dane)  # Uczy model na danych i przypisuje każdej transakcji wynik: 1 dla normalnej, -1 dla anomalii.

dane["status"] = dane["wynik_modelu"].map({  # Tworzy czytelną kolumnę status na podstawie wyniku modelu.
    1: "normalna",  # Zamienia wartość 1 na opis "normalna".
    -1: "podejrzana",  # Zamienia wartość -1 na opis "podejrzana".
})  # Kończy mapowanie wyników modelu na tekst.

print("Liczba transakcji według statusu:")

print(dane["status"].value_counts())  # Zlicza, ile transakcji zostało uznanych za normalne i podejrzane.

print("Transakcje podejrzane:")
print(dane[dane["status"] == "podejrzana"])  # Wyświetla tylko te transakcje, które model oznaczył jako podejrzane.

plt.figure(figsize=(8, 6))  # Tworzy nowy wykres o rozmiarze 8 na 6 cali.

plt.scatter(  # Tworzy wykres punktowy transakcji.
    dane["kwota"],  # Ustawia kwotę transakcji jako wartości na osi X.
    dane["liczba_transakcji_24h"],  # Ustawia liczbę transakcji w 24h jako wartości na osi Y.
    c=(dane["status"] == "podejrzana"),  # Koloruje punkty w zależności od tego, czy transakcja jest podejrzana.
    alpha=0.7,  # Ustawia częściową przezroczystość punktów.
)  # Kończy tworzenie wykresu punktowego.

plt.xlabel("Kwota transakcji")  # Dodaje opis osi X.
plt.ylabel("Liczba transakcji w ostatnich 24h")  # Dodaje opis osi Y.
plt.title("Wykrywanie podejrzanych transakcji")  # Dodaje tytuł wykresu.

plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.