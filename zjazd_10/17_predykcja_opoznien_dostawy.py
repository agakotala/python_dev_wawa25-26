import numpy as np  # Importuje bibliotekę NumPy, używaną tutaj do generowania losowych danych.
import pandas as pd  # Importuje bibliotekę pandas, używaną do tworzenia tabeli z danymi i ważnością cech.
from sklearn.ensemble import RandomForestClassifier  # Importuje model Random Forest do klasyfikacji.
from sklearn.metrics import classification_report  # Importuje funkcję tworzącą szczegółowy raport klasyfikacji.
from sklearn.model_selection import train_test_split  # Importuje funkcję dzielącą dane na zbiór treningowy i testowy.

generator = np.random.default_rng(seed=42)  # Tworzy generator liczb losowych z ustalonym ziarnem, aby wyniki były powtarzalne.
liczba_dostaw = 600  # Określa liczbę sztucznie wygenerowanych dostaw.

dane = pd.DataFrame({  # Tworzy tabelę z przykładowymi danymi o dostawach.
    "odleglosc_km": generator.normal(loc=80, scale=35, size=liczba_dostaw),  # Generuje odległości dostaw wokół średniej 80 km.
    "liczba_paczek": generator.poisson(lam=4, size=liczba_dostaw),  # Generuje liczbę paczek według rozkładu Poissona ze średnią 4.
    "czy_weekend": generator.integers(0, 2, size=liczba_dostaw),  # Losuje, czy dostawa była w weekend: 0 oznacza nie, 1 oznacza tak.
    "natezenie_ruchu": generator.integers(1, 6, size=liczba_dostaw),  # Losuje natężenie ruchu w skali od 1 do 5.
    "doswiadczenie_kuriera_lata": generator.normal(loc=4, scale=2, size=liczba_dostaw),  # Generuje doświadczenie kuriera wokół średniej 4 lata.
})  # Kończy tworzenie tabeli z danymi.

dane["odleglosc_km"] = dane["odleglosc_km"].clip(lower=5)  # Zamienia odległości mniejsze niż 5 km na wartość 5.
dane["doswiadczenie_kuriera_lata"] = dane["doswiadczenie_kuriera_lata"].clip(lower=0)  # Zamienia ujemne doświadczenie na 0 lat.

ryzyko = (  # Tworzy sztuczny wynik ryzyka opóźnienia dostawy.
    0.03 * dane["odleglosc_km"]  # Zwiększa ryzyko wraz ze wzrostem odległości dostawy.
    + 0.35 * dane["liczba_paczek"]  # Zwiększa ryzyko wraz z większą liczbą paczek.
    + 1.2 * dane["czy_weekend"]  # Zwiększa ryzyko, jeśli dostawa odbywa się w weekend.
    + 0.9 * dane["natezenie_ruchu"]  # Zwiększa ryzyko przy większym natężeniu ruchu.
    - 0.4 * dane["doswiadczenie_kuriera_lata"]  # Zmniejsza ryzyko przy większym doświadczeniu kuriera.
    + generator.normal(0, 2, size=liczba_dostaw)  # Dodaje losowy szum, żeby dane były bardziej realistyczne.
)  # Kończy obliczanie sztucznego ryzyka opóźnienia.

dane["czy_opozniona"] = (ryzyko > ryzyko.quantile(0.7)).astype(int)  # Tworzy etykietę: 1 oznacza dostawę opóźnioną, a 0 nieopóźnioną.

X = dane.drop(columns="czy_opozniona")  # Tworzy dane wejściowe X, usuwając kolumnę z odpowiedzią.
y = dane["czy_opozniona"]  # Tworzy zmienną y, czyli informację, czy dostawa była opóźniona.

X_train, X_test, y_train, y_test = train_test_split(  # Dzieli dane na zbiór treningowy i testowy.
    X,  # Przekazuje cechy opisujące dostawy.
    y,  # Przekazuje etykiety mówiące, czy dostawa była opóźniona.
    test_size=0.25,  # Określa, że 25% danych trafi do zbioru testowego.
    random_state=42,  # Ustawia stałą losowość, aby podział danych był powtarzalny.
    stratify=y  # Zachowuje podobne proporcje klas opóźnionych i nieopóźnionych w obu zbiorach.
)  # Kończy wywołanie funkcji train_test_split.

model = RandomForestClassifier(  # Tworzy model Random Forest do klasyfikacji dostaw.
    n_estimators=300,  # Ustawia liczbę drzew decyzyjnych w lesie losowym na 300.
    random_state=42,  # Ustawia stałą losowość, aby wyniki modelu były powtarzalne.
)  # Kończy definicję modelu.

model.fit(X_train, y_train)  # Uczy model na danych treningowych.
y_pred = model.predict(X_test)  # Przewiduje, które dostawy ze zbioru testowego będą opóźnione.

print("RAPORT KLASYFIKACJI:")
print(classification_report(y_test, y_pred))  # Wyświetla precision, recall, F1-score i support dla każdej klasy.

waznosc = pd.Series(  # Tworzy serię pandas z ważnością cech według modelu Random Forest.
    model.feature_importances_,  # Pobiera wartości ważności cech z wytrenowanego modelu.
    index=X.columns,  # Ustawia nazwy kolumn jako indeksy serii.
).sort_values(ascending=False)  # Sortuje cechy od najważniejszej do najmniej ważnej.

print("WAŻNOŚĆ CECH:")
print(waznosc)  # Wyświetla ważność poszczególnych cech.

przyklady = X_test.head(8).copy()  # Tworzy kopię pierwszych 8 przykładów ze zbioru testowego.

przyklady["prawdopodobienstwo_opoznienia"] = model.predict_proba(X_test.head(8))[:, 1]  # Dodaje prawdopodobieństwo, że dana dostawa będzie opóźniona.

print("Przykładowe dostawy:")
print(przyklady)  # Wyświetla przykładowe dostawy razem z przewidywanym prawdopodobieństwem opóźnienia.