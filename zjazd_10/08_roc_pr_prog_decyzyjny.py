import matplotlib.pyplot as plt  # Importuje moduł pyplot z biblioteki matplotlib do tworzenia wykresów.
import pandas as pd  # Importuje bibliotekę pandas, używaną tutaj do utworzenia tabeli z wynikami dla różnych progów.
from sklearn.datasets import load_breast_cancer  # Importuje funkcję wczytującą gotowy zbiór danych Breast Cancer.
from sklearn.linear_model import LogisticRegression  # Importuje model regresji logistycznej do klasyfikacji.
from sklearn.metrics import PrecisionRecallDisplay  # Importuje narzędzie do rysowania krzywej Precision-Recall.
from sklearn.metrics import RocCurveDisplay  # Importuje narzędzie do rysowania krzywej ROC.
from sklearn.metrics import f1_score  # Importuje funkcję obliczającą F1-score, czyli miarę łączącą precision i recall.
from sklearn.metrics import precision_score  # Importuje funkcję obliczającą precision, czyli precyzję predykcji pozytywnych.
from sklearn.metrics import recall_score  # Importuje funkcję obliczającą recall, czyli czułość modelu.
from sklearn.model_selection import train_test_split  # Importuje funkcję dzielącą dane na zbiór treningowy i testowy.
from sklearn.pipeline import Pipeline  # Importuje Pipeline, który pozwala połączyć skalowanie i model w jeden proces.
from sklearn.preprocessing import StandardScaler  # Importuje StandardScaler, który standaryzuje cechy liczbowe.

zbior = load_breast_cancer()  # Wczytuje zbiór Breast Cancer jako dane numeryczne NumPy.

X = zbior.data  # Zapisuje do zmiennej X cechy, czyli dane wejściowe modelu.
y = zbior.target  # Zapisuje do zmiennej y etykiety klas, czyli informację o klasie każdej obserwacji.

X_train, X_test, y_train, y_test = train_test_split(  # Dzieli dane na część treningową i testową.
    X,  # Przekazuje cechy wejściowe.
    y,  # Przekazuje etykiety klas.
    test_size=0.2,  # Określa, że 20% danych trafi do zbioru testowego.
    random_state=42,  # Ustawia stałą losowość, aby podział danych był powtarzalny.
    stratify=y,  # Zachowuje podobne proporcje klas w zbiorze treningowym i testowym.
)  # Kończy wywołanie funkcji train_test_split.

model = Pipeline(steps=[  # Tworzy pipeline, czyli połączony proces przygotowania danych i trenowania modelu.
    ("skalowanie", StandardScaler()),  # Standaryzuje cechy, aby miały średnią 0 i odchylenie standardowe 1.
    ("model", LogisticRegression(max_iter=2000, random_state=42)),  # Dodaje model regresji logistycznej z większą liczbą iteracji i stałą losowością.
])  # Kończy definicję pipeline'u.

model.fit(X_train, y_train)  # Uczy pipeline na danych treningowych.

RocCurveDisplay.from_estimator(model, X_test, y_test)  # Rysuje krzywą ROC, która pokazuje relację między wykrywaniem pozytywów a liczbą fałszywych alarmów.
plt.title("Krzywa ROC")  # Dodaje tytuł wykresu.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres ROC.

PrecisionRecallDisplay.from_estimator(model, X_test, y_test)  # Rysuje krzywą Precision-Recall, która pokazuje zależność między precision i recall dla różnych progów.
plt.title("Krzywa Precision-Recall")  # Dodaje tytuł wykresu.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres Precision-Recall.

prawdopodobienstwa = model.predict_proba(X_test)[:, 1]  # Pobiera prawdopodobieństwa przynależności do klasy 1 dla obserwacji testowych.

progi = [0.2, 0.3, 0.5, 0.7, 0.8]  # Tworzy listę progów decyzyjnych, które będą porównywane.
wyniki = []  # Tworzy pustą listę, do której będą dodawane metryki dla każdego progu.

for prog in progi:  # Przechodzi po każdym progu decyzyjnym z listy.
    y_pred_prog = (prawdopodobienstwa >= prog).astype(int)  # Zamienia prawdopodobieństwa na klasy 0 lub 1 według aktualnego progu.
    wyniki.append({  # Dodaje do listy słownik z wynikami dla aktualnego progu.
        "prog": prog,  # Zapisuje aktualny próg decyzyjny.
        "precision": round(precision_score(y_test, y_pred_prog), 3),  # Oblicza precision, czyli jak często predykcja klasy 1 była poprawna.
        "recall": round(recall_score(y_test, y_pred_prog), 3),  # Oblicza recall, czyli ile rzeczywistych przypadków klasy 1 model wykrył.
        "f1": round(f1_score(y_test, y_pred_prog), 3),  # Oblicza F1, czyli jedną miarę łączącą precision i recall.
    })  # Kończy dodawanie wyników dla aktualnego progu.

tabela = pd.DataFrame(wyniki)  # Tworzy tabelę pandas z wynikami metryk dla różnych progów.

print("Wpływ progu na metryki:")
print(tabela)  # Wyświetla tabelę pokazującą, jak zmieniają się precision, recall i F1 przy różnych progach.

plt.figure(figsize=(8, 6))  # Tworzy nowy wykres o rozmiarze 8 na 6 cali.
plt.plot(tabela["prog"], tabela["precision"], marker="o", label="precision")  # Rysuje linię pokazującą zmianę precision w zależności od progu.
plt.plot(tabela["prog"], tabela["recall"], marker="o", label="recall")  # Rysuje linię pokazującą zmianę recall w zależności od progu.
plt.plot(tabela["prog"], tabela["f1"], marker="o", label="f1")  # Rysuje linię pokazującą zmianę F1 w zależności od progu.
plt.xlabel("Próg decyzyjny")  # Dodaje opis osi X.
plt.ylabel("Wartość metryki")  # Dodaje opis osi Y.
plt.title("Wpływ progu na precision, recall, F1")  # Dodaje tytuł wykresu.
plt.legend()  # Wyświetla legendę z nazwami metryk.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.

print("Interpretacja:")
print("1. Próg powinien zależeć od kosztu błędu.")