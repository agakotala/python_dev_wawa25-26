import pandas as pd  # Importuje bibliotekę pandas, używaną tutaj do utworzenia tabeli z wynikami.
from sklearn.datasets import load_diabetes  # Importuje funkcję wczytującą gotowy zbiór danych Diabetes.
from sklearn.linear_model import Ridge  # Importuje model regresji Ridge, czyli regresji liniowej z regularyzacją L2.
from sklearn.metrics import r2_score  # Importuje funkcję obliczającą współczynnik determinacji R2.
from sklearn.model_selection import train_test_split  # Importuje funkcję dzielącą dane na zbiór treningowy i testowy.
import matplotlib.pyplot as plt  # Importuje moduł pyplot z biblioteki matplotlib do tworzenia wykresów.

zbior = load_diabetes(as_frame=True)  # Wczytuje zbiór Diabetes w formie tabel pandas dzięki parametrowi as_frame=True.
X = zbior.data  # Zapisuje do zmiennej X cechy, czyli dane wejściowe modelu.
y = zbior.target  # Zapisuje do zmiennej y wartości docelowe, które model ma przewidywać.

X_train, X_test, y_train, y_test = train_test_split(  # Dzieli dane na część treningową i testową.
    X,  # Przekazuje cechy wejściowe.
    y,  # Przekazuje wartości docelowe.
    test_size=0.2,  # Określa, że 20% danych trafi do zbioru testowego.
    random_state=42,  # Ustawia stałą losowość, aby podział był powtarzalny.
)  # Kończy wywołanie funkcji train_test_split.

model = Ridge(alpha=1.0)  # Tworzy model Ridge z parametrem alpha=1.0, który określa siłę regularyzacji.
model.fit(X_train, y_train)  # Uczy model na danych treningowych.
y_pred = model.predict(X_test)  # Wyznacza przewidywania modelu dla danych testowych.

reszty = y_test - y_pred  # Oblicza reszty, czyli różnice między wartościami rzeczywistymi a przewidywanymi.

print("R2:", round(r2_score(y_test, y_pred), 3))  # Oblicza współczynnik R2 i wyświetla go po zaokrągleniu do trzech miejsc po przecinku.

wyniki = pd.DataFrame({  # Tworzy tabelę pandas z wynikami rzeczywistymi, przewidywanymi i resztami.
    "rzeczywiste": y_test.values,  # Dodaje kolumnę z rzeczywistymi wartościami ze zbioru testowego.
    "przewidywane": y_pred,  # Dodaje kolumnę z wartościami przewidywanymi przez model.
    "reszty": reszty.values,  # Dodaje kolumnę z resztami modelu.
})  # Kończy tworzenie DataFrame.

print("Pierwsze 10 wierszy wyników:")
print(wyniki.head(10))  # Wyświetla pierwsze 10 wierszy tabeli z wynikami.

plt.figure(figsize=(8, 6))  # Tworzy nowy wykres o rozmiarze 8 na 6 cali.
plt.scatter(y_test, y_pred, alpha=0.7)  # Tworzy wykres punktowy porównujący wartości rzeczywiste i przewidywane.
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])  # Rysuje linię przekątną oznaczającą idealne przewidywania.
plt.xlabel("Wartości rzeczywiste")  # Dodaje opis osi X.
plt.ylabel("Wartości przewidywane")  # Dodaje opis osi Y.
plt.title("Rzeczywiste vs przewidywane")  # Dodaje tytuł wykresu.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.

plt.figure(figsize=(8, 6))  # Tworzy nowy wykres o rozmiarze 8 na 6 cali.
plt.scatter(y_pred, reszty, alpha=0.7)  # Tworzy wykres punktowy przedstawiający zależność między przewidywaniami a resztami.
plt.axhline(y=0, linewidth=2)  # Rysuje poziomą linię na poziomie 0, która ułatwia analizę reszt.
plt.xlabel("Wartości przewidywane")  # Dodaje opis osi X.
plt.ylabel("Reszty")  # Dodaje opis osi Y.
plt.title("Wykres reszt")  # Dodaje tytuł wykresu.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.

plt.figure(figsize=(8, 6))  # Tworzy nowy wykres o rozmiarze 8 na 6 cali.
plt.hist(reszty, bins=15)  # Tworzy histogram rozkładu reszt z podziałem na 15 przedziałów.
plt.xlabel("Reszta")  # Dodaje opis osi X.
plt.ylabel("Liczba obserwacji")  # Dodaje opis osi Y.
plt.title("Rozkład reszt")  # Dodaje tytuł wykresu.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.

print("Interpretacja:")
print("1. Wykres rzeczywiste vs przewidywane pokazuje ogólną jakość predykcji.")
print("2. Wykres reszt pomaga sprawdzić, czy błędy wyglądają losowo.")
print("3. Jeśli reszty układają się we wzór, model może pomijać ważną zależność.")