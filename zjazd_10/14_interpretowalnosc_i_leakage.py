import matplotlib.pyplot as plt  # Importuje moduł pyplot z biblioteki matplotlib do tworzenia wykresów.
import numpy as np  # Importuje bibliotekę NumPy, używaną tutaj do losowego szumu przy tworzeniu podejrzanej cechy.
import pandas as pd  # Importuje bibliotekę pandas, używaną tutaj do tworzenia serii z ważnością cech i współczynnikami.
from sklearn.datasets import load_breast_cancer  # Importuje funkcję wczytującą gotowy zbiór danych Breast Cancer.
from sklearn.inspection import permutation_importance  # Importuje metodę sprawdzającą ważność cech przez ich losowe mieszanie.
from sklearn.linear_model import LogisticRegression  # Importuje regresję logistyczną do klasyfikacji.
from sklearn.model_selection import cross_val_score, train_test_split  # Importuje walidację krzyżową oraz podział na trening i test.
from sklearn.pipeline import Pipeline  # Importuje Pipeline, który łączy kilka kroków, np. skalowanie i model.
from sklearn.preprocessing import StandardScaler  # Importuje StandardScaler, który standaryzuje cechy liczbowe.

zbior = load_breast_cancer(as_frame=True)  # Wczytuje zbiór Breast Cancer w formie tabel pandas dzięki parametrowi as_frame=True.

X = zbior.data  # Zapisuje do zmiennej X cechy, czyli dane wejściowe modelu.
y = zbior.target  # Zapisuje do zmiennej y etykiety klas, czyli zmienną, którą model ma przewidywać.

X_train, X_test, y_train, y_test = train_test_split(  # Dzieli dane na część treningową i testową.
    X,  # Przekazuje cechy wejściowe.
    y,  # Przekazuje etykiety klas.
    test_size=0.2,  # Określa, że 20% danych zostanie przeznaczone na zbiór testowy.
    random_state=42,  # Ustawia stałą losowość, aby podział danych był powtarzalny.
    stratify=y  # Zachowuje podobne proporcje klas w zbiorze treningowym i testowym.
)  # Kończy wywołanie funkcji train_test_split.

model = Pipeline(steps=[  # Tworzy pipeline, czyli połączony proces skalowania danych i trenowania modelu.
    ("skalowanie", StandardScaler()),  # Standaryzuje cechy, aby miały średnią 0 i odchylenie standardowe 1.
    ("model", LogisticRegression(max_iter=3000, random_state=42))  # Dodaje model regresji logistycznej z większą liczbą iteracji i stałą losowością.
])  # Kończy definicję pipeline'u.

model.fit(X_train, y_train)  # Uczy cały pipeline na danych treningowych.

print("WYNIK NA TEST:")
print(round(model.score(X_test, y_test), 3))  # Oblicza accuracy na zbiorze testowym i zaokrągla wynik do trzech miejsc po przecinku.

regresja = model.named_steps["model"]  # Pobiera z pipeline'u sam model regresji logistycznej, bez kroku skalowania.

wspolczynniki = pd.Series(regresja.coef_[0], index=X.columns).sort_values()  # Tworzy serię pandas ze współczynnikami regresji i sortuje je rosnąco.

plt.figure(figsize=(8, 7))  # Tworzy nowy wykres o rozmiarze 8 na 7 cali.

wspolczynniki.tail(10).plot(kind="barh")  # Rysuje poziomy wykres słupkowy dla 10 największych dodatnich współczynników.

plt.title("Największe dodatnie współczynniki")  # Dodaje tytuł wykresu.
plt.xlabel("Wartość współczynnika")  # Dodaje opis osi X.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.

plt.figure(figsize=(8, 7))  # Tworzy nowy wykres o rozmiarze 8 na 7 cali.
wspolczynniki.head(10).plot(kind="barh")  # Rysuje poziomy wykres słupkowy dla 10 najbardziej ujemnych współczynników.

plt.title("Największe ujemne współczynniki")  # Dodaje tytuł wykresu.
plt.xlabel("Wartość współczynnika")  # Dodaje opis osi X.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.

wyniki_permutacji = permutation_importance(  # Oblicza ważność cech metodą permutacji.
    model,  # Przekazuje wytrenowany pipeline, który ma zostać oceniony.
    X_test,  # Przekazuje cechy ze zbioru testowego.
    y_test,  # Przekazuje prawdziwe etykiety ze zbioru testowego.
    n_repeats=20,  # Określa, że każda cecha będzie mieszana 20 razy, aby wynik był stabilniejszy.
    random_state=42,  # Ustawia stałą losowość, aby wyniki permutacji były powtarzalne.
    scoring="accuracy",  # Ustawia accuracy jako miarę, której spadek będzie analizowany.
)  # Kończy wywołanie permutation_importance.

waznosc = pd.Series(  # Tworzy serię pandas z ważnością cech obliczoną metodą permutacji.
    wyniki_permutacji.importances_mean,  # Pobiera średni spadek wyniku po losowym przemieszaniu każdej cechy.
    index=X.columns,  # Ustawia nazwy cech jako indeksy serii.
).sort_values()  # Sortuje ważności cech rosnąco.

plt.figure(figsize=(8, 7))  # Tworzy nowy wykres o rozmiarze 8 na 7 cali.

waznosc.tail(12).plot(kind="barh")  # Rysuje poziomy wykres słupkowy dla 12 najważniejszych cech według permutacji.

plt.title("Permutacja")  # Dodaje tytuł wykresu.
plt.xlabel("Średni spadek accuracy po przetestowaniu cechy")  # Dodaje opis osi X.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.

generator = np.random.default_rng(seed=42)  # Tworzy generator liczb losowych z ustalonym ziarnem, aby wynik był powtarzalny.

X_leakage = X.copy()  # Tworzy kopię danych wejściowych, do której zostanie dodana podejrzana cecha.

X_leakage["podejrzana_cecha_target"] = y + generator.normal(0, 0.01, size=len(y))  # Dodaje cechę prawie identyczną z targetem, co symuluje data leakage.

model_normalny = Pipeline(steps=[  # Tworzy pipeline dla modelu trenowanego na normalnych danych.
    ("skalowanie", StandardScaler()),  # Standaryzuje cechy liczbowe.
    ("model", LogisticRegression(max_iter=3000, random_state=42))  # Dodaje model regresji logistycznej.
])  # Kończy definicję normalnego modelu.

model_z_leakage = Pipeline(steps=[  # Tworzy pipeline dla modelu trenowanego na danych z podejrzaną cechą.
    ("skalowanie", StandardScaler()),  # Standaryzuje cechy liczbowe.
    ("model", LogisticRegression(max_iter=3000, random_state=42))  # Dodaje model regresji logistycznej.
])  # Kończy definicję modelu z data leakage.

wynik_normalny = cross_val_score(  # Ocenia normalny model za pomocą walidacji krzyżowej.
    model_normalny,  # Przekazuje pipeline bez podejrzanej cechy.
    X,  # Przekazuje normalne cechy.
    y,  # Przekazuje etykiety klas.
    cv=5,  # Dzieli dane na 5 części w walidacji krzyżowej.
    scoring="accuracy",  # Ustawia accuracy jako miarę oceny.
).mean()  # Oblicza średni wynik accuracy z 5 podziałów.

wynik_leakage = cross_val_score(  # Ocenia model z podejrzaną cechą za pomocą walidacji krzyżowej.
    model_z_leakage,  # Przekazuje pipeline dla danych z podejrzaną cechą.
    X_leakage,  # Przekazuje cechy zawierające informację podobną do targetu.
    y,  # Przekazuje etykiety klas.
    cv=5,  # Dzieli dane na 5 części w walidacji krzyżowej.
    scoring="accuracy",  # Ustawia accuracy jako miarę oceny.
).mean()  # Oblicza średni wynik accuracy z 5 podziałów.

print("Wynik bez podejrzanej cechy:")
print(round(wynik_normalny, 3))  # Wyświetla średni wynik accuracy dla modelu bez podejrzanej cechy.

print("Wynik z podejrzaną cechą:")
print(round(wynik_leakage, 3))  # Wyświetla średni wynik accuracy dla modelu z podejrzaną cechą.

print("Interpretacja:")
print("1. Współczynniki pokazują, które cechy wpływają na wynik modelu.")
print("2. Permutacja sprawdza, jak bardzo wynik spada po pomieszaniu cechy.")
print("3. Bardzo wysoki wynik po dodaniu jednej cechy może sugerować data leakage")
print("4. Data leakage oznacza, że model dostał informację, której nie powinien mieć w momencie predykcji")