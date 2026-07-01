import pandas as pd  # Importuje bibliotekę pandas, używaną tutaj do utworzenia tabeli z wynikami GridSearchCV.
from sklearn.datasets import load_breast_cancer  # Importuje funkcję wczytującą gotowy zbiór danych Breast Cancer.
from sklearn.linear_model import LogisticRegression  # Importuje model regresji logistycznej do klasyfikacji.
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split  # Importuje narzędzia do podziału danych, walidacji krzyżowej i szukania najlepszych parametrów.
from sklearn.pipeline import Pipeline  # Importuje Pipeline, który łączy kilka etapów, np. skalowanie i model.
from sklearn.preprocessing import StandardScaler  # Importuje StandardScaler, który standaryzuje cechy liczbowe.

zbior = load_breast_cancer()  # Wczytuje zbiór Breast Cancer jako dane numeryczne NumPy.

X = zbior.data  # Zapisuje do zmiennej X cechy, czyli dane wejściowe modelu.
y = zbior.target  # Zapisuje do zmiennej y etykiety klas, czyli informację o klasie każdej obserwacji.

X_train, X_test, y_train, y_test = train_test_split(  # Dzieli dane na zbiór treningowy i testowy.
    X,  # Przekazuje cechy wejściowe.
    y,  # Przekazuje etykiety klas.
    test_size=0.2,  # Określa, że 20% danych trafi do zbioru testowego.
    random_state=42,  # Ustawia stałą losowość, aby podział danych był powtarzalny.
    stratify=y  # Zachowuje podobne proporcje klas w zbiorze treningowym i testowym.
)  # Kończy wywołanie funkcji train_test_split.

model = Pipeline(steps=[  # Tworzy pipeline, czyli jeden wspólny proces przygotowania danych i uczenia modelu.
    ("skalowanie", StandardScaler()),  # Standaryzuje cechy, aby miały średnią 0 i odchylenie standardowe 1.
    ("model", LogisticRegression(max_iter=3000, random_state=42))  # Dodaje model regresji logistycznej z większą liczbą iteracji i stałą losowością.
])  # Kończy definicję pipeline'u.

wyniki_cv = cross_val_score(  # Wykonuje walidację krzyżową dla modelu na zbiorze treningowym.
    model,  # Przekazuje pipeline, który ma zostać oceniony.
    X_train,  # Przekazuje cechy ze zbioru treningowego.
    y_train,  # Przekazuje etykiety ze zbioru treningowego.
    cv=5,  # Dzieli dane treningowe na 5 części, aby model był oceniany na różnych podziałach.
    scoring="f1"  # Ustawia F1-score jako miarę oceny modelu.
)  # Kończy wywołanie cross_val_score.

print("Wyniki Cross-Validation:")
print(wyniki_cv)  # Wyświetla wyniki F1-score uzyskane w kolejnych podziałach walidacji krzyżowej.

print("Średni F1:", round(wyniki_cv.mean(), 3))  # Oblicza i wyświetla średni wynik F1 ze wszystkich podziałów walidacji krzyżowej.
print("Odchylenie:", round(wyniki_cv.std(), 3))  # Oblicza i wyświetla odchylenie standardowe wyników, czyli informację o stabilności modelu.

parametry = {  # Tworzy słownik parametrów, które będą sprawdzane przez GridSearchCV.
    "model__C": [0.01, 0.1, 1, 10, 100],  # Określa różne wartości parametru C dla regresji logistycznej.
}  # Kończy tworzenie słownika parametrów.

szukanie = GridSearchCV(  # Tworzy obiekt GridSearchCV, który szuka najlepszego ustawienia modelu.
    model,  # Przekazuje pipeline, którego parametry będą testowane.
    param_grid=parametry,  # Przekazuje listę parametrów do sprawdzenia.
    cv=5,  # Ustawia 5-krotną walidację krzyżową dla każdego zestawu parametrów.
    scoring="f1",  # Ustawia F1-score jako miarę wyboru najlepszego modelu.
    n_jobs=-1,  # Pozwala użyć wszystkich dostępnych rdzeni procesora, aby przyspieszyć obliczenia.
)  # Kończy tworzenie obiektu GridSearchCV.

szukanie.fit(X_train, y_train)  # Uruchamia GridSearchCV, czyli trenuje i ocenia model dla różnych wartości parametru C.

print("Najlepsze parametry:")
print(szukanie.best_params_)  # Wyświetla najlepszy zestaw parametrów znaleziony przez GridSearchCV.

print("Najlepszy wynik walidacji:")
print(round(szukanie.best_score_, 3))  # Wyświetla najlepszy średni wynik F1 uzyskany w walidacji krzyżowej.

print("Wynik na finalnym zbiorze testowym:")
print(round(szukanie.score(X_test, y_test), 3))  # Ocenia najlepszy znaleziony model na zbiorze testowym, który wcześniej nie był używany do wyboru parametrów.

tabela = pd.DataFrame(szukanie.cv_results_)  # Tworzy tabelę pandas ze wszystkimi wynikami działania GridSearchCV.

print("Wyniki dla poszczególnych parametrów:")
print(tabela[["param_model__C", "mean_test_score", "std_test_score"]])  # Wyświetla wartości parametru C oraz średnie i odchylenia wyników F1.

print("Interpretacja:")
print("1. Cross-validation pokazuje stabilność wyniku na różnych podziałach.")
print("2. GridSearchCV sprawdza kilka ustawień modelu.")
print("3. Test zostawiamy na koniec, żeby uczciwie ocenić model.")