import matplotlib.pyplot as plt  # Importuje moduł pyplot z biblioteki matplotlib, używany do tworzenia wykresów.
import numpy as np  # Importuje bibliotekę NumPy, używaną tutaj do obliczenia pierwiastka przy RMSE.
import pandas as pd  # Importuje bibliotekę pandas, używaną tutaj do utworzenia Series i DataFrame.
from sklearn.datasets import load_diabetes  # Importuje funkcję wczytującą gotowy zbiór danych Diabetes.
from sklearn.linear_model import LinearRegression  # Importuje model regresji liniowej.
from sklearn.metrics import mean_absolute_error  # Importuje funkcję obliczającą MAE, czyli średni błąd bezwzględny.
from sklearn.metrics import mean_squared_error  # Importuje funkcję obliczającą MSE, czyli średni błąd kwadratowy.
from sklearn.metrics import r2_score  # Importuje funkcję obliczającą współczynnik determinacji R2.
from sklearn.model_selection import train_test_split  # Importuje funkcję dzielącą dane na zbiór treningowy i testowy.

zbior = load_diabetes(as_frame=True)  # Wczytuje zbiór Diabetes w formie tabel pandas dzięki parametrowi as_frame=True.

X = zbior.data  # Zapisuje do zmiennej X dane wejściowe, czyli cechy opisujące obserwacje.
y = zbior.target  # Zapisuje do zmiennej y wartości docelowe, które model ma przewidywać.

print("Pierwsze wiersze danych:")
print(zbior.frame.head(10))  # Wyświetla pierwsze 10 wierszy pełnej tabeli danych.

print("Kształt cech X:", X.shape)  # Wyświetla rozmiar danych wejściowych X, czyli liczbę wierszy i kolumn.
print("Kształt celu y:", y.shape)  # Wyświetla rozmiar zmiennej docelowej y, czyli liczbę wartości do przewidywania.

X_train, X_test, y_train, y_test = train_test_split(  # Dzieli dane na część treningową i testową.
    X,  # Przekazuje cechy, czyli dane wejściowe.
    y,  # Przekazuje wartości docelowe.
    test_size=0.2,  # Określa, że 20% danych zostanie przeznaczone na zbiór testowy.
    random_state=42,  # Ustawia stałą losowość, aby podział danych był powtarzalny.
)  # Kończy wywołanie funkcji train_test_split.

model = LinearRegression()  # Tworzy obiekt modelu regresji liniowej.

model.fit(X_train, y_train)  # Uczy model na danych treningowych.

y_pred = model.predict(X_test)  # Tworzy przewidywania dla danych testowych.

mae = mean_absolute_error(y_test, y_pred)  # Oblicza MAE, czyli średni bezwzględny błąd predykcji.
mse = mean_squared_error(y_test, y_pred)  # Oblicza MSE, czyli średni kwadratowy błąd predykcji.
rmse = np.sqrt(mse)  # Oblicza RMSE jako pierwiastek kwadratowy z MSE.
r2 = r2_score(y_test, y_pred)  # Oblicza współczynnik R2, który pokazuje, jak dobrze model wyjaśnia zmienność danych.

print("Współczynniki modelu:")
wspolczynniki = pd.Series(model.coef_, index=X.columns)  # Tworzy serię pandas z wartościami współczynników modelu i nazwami cech jako indeksami.
print(wspolczynniki.sort_values(ascending=False))  # Sortuje współczynniki malejąco i je wyświetla.

print("Wyraz wolny modelu:")
print(round(model.intercept_, 3))  # Wyświetla wyraz wolny modelu zaokrąglony do trzech miejsc po przecinku.

print("Metryki oceny:")
print("MAE:", round(mae, 3))  # Wyświetla MAE zaokrąglone do trzech miejsc po przecinku.
print("MSE:", round(mse, 3))  # Wyświetla MSE zaokrąglone do trzech miejsc po przecinku.
print("RMSE:", round(rmse, 3))  # Wyświetla RMSE zaokrąglone do trzech miejsc po przecinku.
print("R2: ", round(r2, 3))  # Wyświetla R2 zaokrąglone do trzech miejsc po przecinku.

wyniki = pd.DataFrame({  # Tworzy tabelę pandas z wynikami rzeczywistymi i przewidywanymi.
    "rzeczywiste": y_test.values,  # Dodaje kolumnę z rzeczywistymi wartościami ze zbioru testowego.
    "przewidywane": y_pred,  # Dodaje kolumnę z wartościami przewidzianymi przez model.
})  # Kończy tworzenie DataFrame.

wyniki["blad"] = wyniki["rzeczywiste"] - wyniki["przewidywane"]  # Dodaje kolumnę z błędem, czyli różnicą między wartością rzeczywistą a przewidywaną.

print("Pierwsze 10 wyników")
print(wyniki.head(10))  # Wyświetla pierwsze 10 wierszy tabeli z wynikami.

plt.figure(figsize=(8, 6))  # Tworzy nowy wykres o rozmiarze 8 na 6 cali.
plt.scatter(y_test, y_pred, alpha=0.7)  # Tworzy wykres punktowy porównujący wartości rzeczywiste z przewidywanymi.

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])  # Rysuje linię przekątną oznaczającą idealne przewidywania.
plt.xlabel("Wartości rzeczywiste")  # Dodaje opis osi X.
plt.ylabel("Wartości przewidywane")  # Dodaje opis osi Y.
plt.title("Regresja liniowa: rzeczywiste vs przewidywane")  # Dodaje tytuł wykresu.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.

print("Interpretacja:")
print("1. Im bliżej punkty są linii przekątnej, tym lepsze predykcje.")
print("2. MAE mówi o przeciętnym błędzie w jednostkach zmiennej docelowej.")
print("3. R2 pokazuje, jaka część zmienności została wyjaśniona przez model.")