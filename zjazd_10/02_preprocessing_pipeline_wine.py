import numpy as np  # Importuje bibliotekę NumPy, używaną tutaj do losowego wybierania indeksów i wstawiania wartości NaN.
from sklearn.compose import ColumnTransformer  # Importuje ColumnTransformer, który pozwala zastosować wybrane przekształcenia do konkretnych kolumn.
from sklearn.datasets import load_wine  # Importuje funkcję load_wine, która wczytuje gotowy zbiór danych o winach.
from sklearn.impute import SimpleImputer  # Importuje SimpleImputer, który służy do uzupełniania brakujących danych.
from sklearn.linear_model import LogisticRegression  # Importuje model regresji logistycznej używany do klasyfikacji.
from sklearn.metrics import accuracy_score  # Importuje funkcję accuracy_score, która oblicza dokładność modelu.
from sklearn.model_selection import train_test_split  # Importuje funkcję dzielącą dane na zbiór treningowy i testowy.
from sklearn.pipeline import Pipeline  # Importuje Pipeline, który łączy kilka kroków przetwarzania i modelowania w jedną całość.
from sklearn.preprocessing import StandardScaler  # Importuje StandardScaler, który standaryzuje cechy liczbowe.

zbior = load_wine(as_frame=True)  # Wczytuje zbiór Wine w formie tabel pandas dzięki parametrowi as_frame=True.
X = zbior.data.copy()  # Tworzy kopię danych wejściowych, czyli cech opisujących próbki win.
y = zbior.target  # Zapisuje etykiety klas, czyli rodzaje win.

generator = np.random.default_rng(seed=42)  # Tworzy generator liczb losowych z ustalonym ziarnem, aby wyniki były powtarzalne.

for kolumna in X.columns[:4]:  # Przechodzi przez pierwsze cztery kolumny zbioru X.
    indeksy = generator.choice(X.index, size=5, replace=False)  # Losuje 5 różnych indeksów wierszy bez powtórzeń.
    X.loc[indeksy, kolumna] = np.nan  # Wstawia braki danych NaN w wylosowanych wierszach aktualnej kolumny.

print("Pierwsze wiersze z brakami:")
print(X.head(20))  # Wyświetla pierwsze 20 wierszy danych, aby pokazać wstawione braki.

print("Liczba braków w każdej kolumnie:")
print(X.isna().sum())  # Sprawdza, gdzie występują wartości NaN, a następnie zlicza braki w każdej kolumnie.

X_train, X_test, y_train, y_test = train_test_split(  # Dzieli cechy i etykiety na dane treningowe oraz testowe.
    X,  # Przekazuje dane wejściowe, czyli cechy próbek win.
    y,  # Przekazuje etykiety klas, czyli poprawne rodzaje win.
    test_size=0.25,  # Określa, że 25% danych zostanie przeznaczone na zbiór testowy.
    random_state=42,  # Ustawia stałą losowość, aby podział danych był powtarzalny.
    stratify=y,  # Zachowuje podobne proporcje klas w zbiorze treningowym i testowym.
)  # Kończy wywołanie funkcji train_test_split.

kolumny_liczbowe = X.columns.tolist()  # Tworzy listę nazw wszystkich kolumn liczbowych, które będą przetwarzane.

preprocessor = ColumnTransformer(transformers=[  # Tworzy obiekt odpowiedzialny za przetwarzanie wybranych kolumn.
    ("num",  # Nadaje nazwę transformacji dla kolumn liczbowych.
     Pipeline(steps=[  # Tworzy pipeline z krokami przetwarzania danych liczbowych.
         ("imputer", SimpleImputer(strategy="median")),  # Uzupełnia brakujące wartości medianą obliczoną na danych treningowych.
         ("scaler", StandardScaler()),  # Standaryzuje cechy, czyli przekształca je tak, aby miały średnią 0 i odchylenie standardowe 1.
     ]),  # Kończy listę kroków pipeline'u dla kolumn liczbowych.
     kolumny_liczbowe)  # Wskazuje, do których kolumn należy zastosować ten pipeline.
])  # Kończy definicję ColumnTransformer.

model = Pipeline(steps=[  # Tworzy główny pipeline łączący preprocessing i model uczenia maszynowego.
    ("preprocessing", preprocessor),  # Dodaje etap przygotowania danych, czyli uzupełnianie braków i standaryzację.
    ("model", LogisticRegression(max_iter=2000, random_state=42))  # Dodaje model regresji logistycznej z większą liczbą iteracji i stałą losowością.
])  # Kończy definicję głównego pipeline'u.

model.fit(X_train, y_train)  # Uczy cały pipeline na danych treningowych, wykonując preprocessing i dopasowanie modelu.
y_pred = model.predict(X_test)  # Przewiduje klasy dla danych testowych.

accuracy = accuracy_score(y_test, y_pred)  # Oblicza dokładność modelu, porównując prawdziwe etykiety z przewidywaniami.

print("Accuracy:", round(accuracy, 3))  # Wyświetla dokładność modelu zaokrągloną do trzech miejsc po przecinku.

print("Nazwy cech po transformacji:")
print(model.named_steps["preprocessing"].get_feature_names_out())  # Pobiera i wyświetla nazwy cech po przejściu przez etap preprocessing.

print("Wnioski:")
print("1. Braki danych zostały uzupełnione medianą obliczoną tylko na treningu.")
print("2. Cechy liczbowe zostały wystandaryzowane.")