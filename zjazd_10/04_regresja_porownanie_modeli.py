import numpy as np  # Importuje bibliotekę NumPy, używaną tutaj do obliczenia pierwiastka przy RMSE.
import pandas as pd  # Importuje bibliotekę pandas, używaną tutaj do utworzenia tabeli z wynikami modeli.
from sklearn.datasets import load_diabetes  # Importuje funkcję wczytującą gotowy zbiór danych Diabetes.
from sklearn.ensemble import RandomForestRegressor  # Importuje model Random Forest do zadania regresji.
from sklearn.linear_model import LinearRegression, Ridge  # Importuje modele regresji liniowej i Ridge.
from sklearn.metrics import mean_absolute_error  # Importuje funkcję obliczającą MAE, czyli średni błąd bezwzględny.
from sklearn.metrics import mean_squared_error  # Importuje funkcję obliczającą MSE, czyli średni błąd kwadratowy.
from sklearn.metrics import r2_score  # Importuje funkcję obliczającą współczynnik determinacji R2.
from sklearn.model_selection import train_test_split  # Importuje funkcję dzielącą dane na zbiór treningowy i testowy.

zbior = load_diabetes(as_frame=True)  # Wczytuje zbiór Diabetes w formie tabel pandas dzięki parametrowi as_frame=True.

X = zbior.data  # Zapisuje do zmiennej X dane wejściowe, czyli cechy opisujące obserwacje.
y = zbior.target  # Zapisuje do zmiennej y wartości docelowe, które modele mają przewidywać.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Dzieli dane na zbiór treningowy i testowy; 20% danych trafia do testu, a random_state zapewnia powtarzalność.

modele = {  # Tworzy słownik z nazwami modeli oraz odpowiadającymi im obiektami modeli.
    "Regresja liniowa": LinearRegression(),  # Dodaje model regresji liniowej.
    "Ridge": Ridge(alpha=1),  # Dodaje model Ridge z parametrem alpha równym 1, który określa siłę regularyzacji.
    "Random Forest": RandomForestRegressor(  # Dodaje model lasu losowego do regresji.
        n_estimators=300,  # Ustawia liczbę drzew w lesie losowym na 300.
        random_state=42,  # Ustawia stałą losowość, aby wyniki modelu były powtarzalne.
    ),  # Kończy definicję modelu Random Forest.
}  # Kończy tworzenie słownika z modelami.

wyniki = []  # Tworzy pustą listę, do której będą dodawane wyniki każdego modelu.

for nazwa, model in modele.items():  # Przechodzi po każdym modelu ze słownika, pobierając jego nazwę i obiekt modelu.
    model.fit(X_train, y_train)  # Uczy aktualny model na danych treningowych.
    y_pred = model.predict(X_test)  # Tworzy przewidywania dla danych testowych.
    mse = mean_squared_error(y_test, y_pred)  # Oblicza MSE dla aktualnego modelu.
    wyniki.append({  # Dodaje do listy słownik z nazwą modelu i jego metrykami.
        "model": nazwa,  # Zapisuje nazwę aktualnego modelu.
        "MAE": round(mean_absolute_error(y_test, y_pred), 3),  # Oblicza MAE i zaokrągla wynik do trzech miejsc po przecinku.
        "RMSE": round(np.sqrt(mse), 3),  # Oblicza RMSE jako pierwiastek z MSE i zaokrągla wynik do trzech miejsc po przecinku.
        "R2": round(r2_score(y_test, y_pred), 3),  # Oblicza R2 i zaokrągla wynik do trzech miejsc po przecinku.
    })  # Kończy dodawanie wyników aktualnego modelu do listy.

tabela = pd.DataFrame(wyniki).sort_values(by="RMSE")  # Tworzy tabelę pandas z wynikami i sortuje modele według RMSE rosnąco.

print("Porównanie modeli:")
print(tabela)  # Wyświetla tabelę z porównaniem modeli.

print("Interpretacja:")
print("1. Niższe MAE i RMSE oznaczają mniejszy błąd.")
print("2. Wyższe R2 oznacza lepsze wyjaśnienie zmienności celu.")
print("3. Modele należy porównywać na tym samym zbiorze testowym.")