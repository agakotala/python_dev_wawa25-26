import pandas as pd  # Importuje bibliotekę pandas, używaną tutaj do utworzenia tabeli z wynikami modeli.
from sklearn.datasets import load_wine  # Importuje funkcję wczytującą gotowy zbiór danych Wine.
from sklearn.ensemble import RandomForestClassifier  # Importuje model Random Forest do klasyfikacji.
from sklearn.ensemble import StackingClassifier  # Importuje StackingClassifier, który łączy modele bazowe za pomocą dodatkowego modelu końcowego.
from sklearn.ensemble import VotingClassifier  # Importuje VotingClassifier, który łączy decyzje kilku modeli bazowych.
from sklearn.linear_model import LogisticRegression  # Importuje regresję logistyczną używaną jako model klasyfikacyjny.
from sklearn.metrics import accuracy_score  # Importuje funkcję obliczającą accuracy, czyli dokładność klasyfikacji.
from sklearn.metrics import f1_score  # Importuje funkcję obliczającą F1-score.
from sklearn.model_selection import train_test_split  # Importuje funkcję dzielącą dane na zbiór treningowy i testowy.
from sklearn.pipeline import Pipeline  # Importuje Pipeline, który pozwala połączyć kilka kroków, np. skalowanie i model.
from sklearn.preprocessing import StandardScaler  # Importuje StandardScaler, który standaryzuje cechy liczbowe.
from sklearn.svm import SVC  # Importuje klasyfikator SVM.
from sklearn.calibration import CalibratedClassifierCV  # Importuje kalibrację modelu, dzięki której SVM może zwracać lepiej dopasowane prawdopodobieństwa.

zbior = load_wine(as_frame=True)  # Wczytuje zbiór Wine w formie tabel pandas dzięki parametrowi as_frame=True.
X = zbior.data  # Zapisuje do zmiennej X cechy, czyli dane wejściowe modelu.
y = zbior.target  # Zapisuje do zmiennej y etykiety klas, czyli rodzaje win.

X_train, X_test, y_train, y_test = train_test_split(  # Dzieli dane na część treningową i testową.
    X,  # Przekazuje cechy wejściowe.
    y,  # Przekazuje etykiety klas.
    test_size=0.25,  # Określa, że 25% danych zostanie przeznaczone na zbiór testowy.
    random_state=42,  # Ustawia stałą losowość, aby podział danych był powtarzalny.
    stratify=y,  # Zachowuje podobne proporcje klas w zbiorze treningowym i testowym.
)  # Kończy wywołanie funkcji train_test_split.

regresja_logistyczna = Pipeline(steps=[  # Tworzy pipeline dla regresji logistycznej.
    ("skalowanie", StandardScaler()),  # Standaryzuje cechy, co pomaga modelowi działać stabilniej.
    ("model", LogisticRegression(max_iter=3000, random_state=42))  # Dodaje model regresji logistycznej z większą liczbą iteracji i stałą losowością.
])  # Kończy definicję pipeline'u regresji logistycznej.

svm = Pipeline(steps=[  # Tworzy pipeline dla modelu SVM, ale w tym kodzie ten model nie jest później używany w porównaniu.
    ("skalowanie", StandardScaler()),  # Standaryzuje cechy, ponieważ SVM jest wrażliwy na skalę danych.
    ("model", CalibratedClassifierCV(  # Dodaje skalibrowany model SVM, który może zwracać prawdopodobieństwa klas.
        estimator=SVC(random_state=42),  # Ustawia SVC jako bazowy model do kalibracji.
        cv=5,  # Ustawia 5-krotną walidację krzyżową do kalibracji prawdopodobieństw.
        ensemble=False  # Ustawia pojedynczy skalibrowany model zamiast zespołu modeli kalibracyjnych.
    )),  # Kończy definicję kroku modelu SVM.
])  # Kończy definicję pipeline'u SVM.

las = RandomForestClassifier(  # Tworzy model Random Forest do klasyfikacji.
    n_estimators=300,  # Ustawia liczbę drzew w lesie losowym na 300.
    random_state=42,  # Ustawia stałą losowość, aby wyniki modelu były powtarzalne.
)  # Kończy definicję modelu Random Forest.

modele_bazowe = [  # Tworzy listę modeli bazowych używanych później w Voting i Stacking.
    ("lr", regresja_logistyczna),  # Dodaje regresję logistyczną jako pierwszy model bazowy.
    ("rf", las),  # Dodaje Random Forest jako drugi model bazowy.
]  # Kończy listę modeli bazowych.

voting = VotingClassifier(  # Tworzy model VotingClassifier łączący decyzje modeli bazowych.
    estimators=modele_bazowe,  # Przekazuje listę modeli, które będą brały udział w głosowaniu.
    voting="hard",  # Ustawia głosowanie większościowe, czyli wybór klasy wskazanej przez większą liczbę modeli.
)  # Kończy definicję modelu VotingClassifier.

stacking = StackingClassifier(  # Tworzy model StackingClassifier, który uczy dodatkowy model na predykcjach modeli bazowych.
    estimators=modele_bazowe,  # Przekazuje listę modeli bazowych.
    final_estimator=LogisticRegression(max_iter=3000),  # Ustawia regresję logistyczną jako model końcowy łączący wyniki modeli bazowych.
    cv=5,  # Ustawia 5-krotną walidację krzyżową do tworzenia danych dla modelu końcowego.
)  # Kończy definicję modelu StackingClassifier.

modele = {  # Tworzy słownik modeli, które będą porównywane.
    "Regresja logistyczna": regresja_logistyczna,  # Dodaje regresję logistyczną do porównania.
    "Random Forest": las,  # Dodaje Random Forest do porównania.
    "Voting": voting,  # Dodaje VotingClassifier do porównania.
    "Stacking": stacking,  # Dodaje StackingClassifier do porównania.
}  # Kończy tworzenie słownika modeli.

wyniki = []  # Tworzy pustą listę, do której będą zapisywane wyniki modeli.

for nazwa, model in modele.items():  # Przechodzi po każdym modelu ze słownika.
    model.fit(X_train, y_train)  # Uczy aktualny model na danych treningowych.
    y_pred = model.predict(X_test)  # Przewiduje klasy dla danych testowych.
    wyniki.append({  # Dodaje wyniki aktualnego modelu do listy wyników.
        "model": nazwa,  # Zapisuje nazwę aktualnego modelu.
        "accuracy": round(accuracy_score(y_test, y_pred), 3),  # Oblicza accuracy, czyli procent poprawnych klasyfikacji, i zaokrągla wynik.
        "F1 macro": round(f1_score(y_test, y_pred, average="macro"), 3),  # Oblicza F1 osobno dla każdej klasy, uśrednia wyniki i zaokrągla rezultat.
    })  # Kończy dodawanie wyników aktualnego modelu.

tabela = pd.DataFrame(wyniki).sort_values(by="F1 macro", ascending=False)  # Tworzy tabelę z wynikami i sortuje modele od najlepszego według F1 macro.

print("Porównanie modeli:")
print(tabela)  # Wyświetla tabelę z porównaniem modeli.

print("Interpretacja:")
print("1. Voting hard wybiera klasę przez głosowanie większościowe modeli bazowych.")
print("2. Stacking uczy dodatkowy model, który łączy predykcje modeli bazowych.")
print("3. Im bardziej różnorodne modele bazowe, tym większa szansa na poprawę wyniku.")