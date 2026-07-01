import pandas as pd  # Importuje bibliotekę pandas, używaną tutaj do utworzenia tabeli z wynikami modeli.
from sklearn.datasets import load_iris  # Importuje funkcję wczytującą gotowy zbiór danych Iris.
from sklearn.linear_model import LogisticRegression  # Importuje model regresji logistycznej do klasyfikacji.
from sklearn.metrics import f1_score, accuracy_score  # Importuje miary oceny klasyfikacji: accuracy oraz F1.
from sklearn.model_selection import train_test_split  # Importuje funkcję dzielącą dane na zbiór treningowy i testowy.
from sklearn.neighbors import KNeighborsClassifier  # Importuje klasyfikator KNN, który klasyfikuje na podstawie najbliższych sąsiadów.
from sklearn.pipeline import Pipeline  # Importuje Pipeline, który pozwala połączyć kilka kroków w jeden proces.
from sklearn.preprocessing import StandardScaler  # Importuje StandardScaler, który standaryzuje cechy liczbowe.
from sklearn.tree import DecisionTreeClassifier  # Importuje klasyfikator drzewa decyzyjnego.

zbior = load_iris(as_frame=True)  # Wczytuje zbiór Iris w formie tabel pandas dzięki parametrowi as_frame=True.

X = zbior.data  # Zapisuje do zmiennej X cechy, czyli dane wejściowe modelu.
y = zbior.target  # Zapisuje do zmiennej y etykiety klas, czyli poprawne gatunki irysów.

print("Pierwsze 10 wierszy:")
print(zbior.frame.head(10))  # Wyświetla pierwsze 10 wierszy pełnej tabeli danych.

X_train, X_test, y_train, y_test = train_test_split(  # Dzieli dane na część treningową i testową.
    X,  # Przekazuje cechy wejściowe.
    y,  # Przekazuje etykiety klas.
    test_size=0.25,  # Określa, że 25% danych zostanie przeznaczone na zbiór testowy.
    random_state=42,  # Ustawia stałą losowość, aby podział danych był powtarzalny.
    stratify=y,  # Zachowuje podobne proporcje klas w zbiorze treningowym i testowym.
)  # Kończy wywołanie funkcji train_test_split.

modele = {  # Tworzy słownik zawierający nazwy modeli oraz odpowiadające im obiekty modeli.
    "Regresja logistyczna": Pipeline(steps=[  # Tworzy pipeline dla regresji logistycznej.
        ("skalowanie", StandardScaler()),  # Standaryzuje dane, co pomaga modelom wrażliwym na skalę cech.
        ("model", LogisticRegression(max_iter=1000, random_state=42)),  # Dodaje model regresji logistycznej do klasyfikacji.
    ]),  # Kończy pipeline regresji logistycznej.
    "KNN": Pipeline(steps=[  # Tworzy pipeline dla klasyfikatora KNN.
        ("skalowanie", StandardScaler()),  # Standaryzuje dane, ponieważ KNN porównuje odległości między punktami.
        ("model", KNeighborsClassifier(n_neighbors=19)),  # Dodaje model KNN, który patrzy na 19 najbliższych sąsiadów.
    ]),  # Kończy pipeline KNN.
    "Drzewo decyzyjne": DecisionTreeClassifier(max_depth=2, random_state=42),  # Dodaje model drzewa decyzyjnego z maksymalną głębokością równą 2.
}  # Kończy tworzenie słownika modeli.

wyniki = []  # Tworzy pustą listę, do której będą zapisywane wyniki każdego modelu.

for nazwa, model in modele.items():  # Przechodzi po każdym modelu ze słownika.
    model.fit(X_train, y_train)  # Uczy aktualny model na danych treningowych.
    y_pred = model.predict(X_test)  # Przewiduje klasy dla danych testowych.
    wyniki.append({  # Dodaje wyniki aktualnego modelu do listy.
        "model": nazwa,  # Zapisuje nazwę modelu.
        "accuracy": round(accuracy_score(y_test, y_pred), 3),  # Oblicza dokładność, czyli procent poprawnych klasyfikacji, i zaokrągla wynik.
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 3),  # Oblicza F1 osobno dla każdej klasy, uśrednia wyniki i zaokrągla rezultat.
    })  # Kończy dodawanie wyników aktualnego modelu.

tabela = pd.DataFrame(wyniki).sort_values(by="f1_macro", ascending=False)  # Tworzy tabelę z wynikami i sortuje modele od najlepszego według F1 macro.

print("Porównanie modeli:")
print(tabela)  # Wyświetla tabelę z porównaniem modeli i ich wynikami.

print("Wnioski:")
print("1. KNN porównuje nowe obserwacje dla najbliższych przykładów treningowych.")
print("2. Drzewo decyzyjne tworzy reguły typu 'jeżeli'....to....'.")
print("3. Porównywanie kilku modeli na tych samych danych to dobra praktyka.")