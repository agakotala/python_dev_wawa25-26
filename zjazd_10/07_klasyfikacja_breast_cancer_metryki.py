import pandas as pd  # Importuje bibliotekę pandas, używaną tutaj do ustawień wyświetlania tabeli.
import matplotlib.pyplot as plt  # Importuje moduł pyplot z biblioteki matplotlib do tworzenia wykresów.
from sklearn.datasets import load_breast_cancer  # Importuje funkcję wczytującą gotowy zbiór danych Breast Cancer.
from sklearn.linear_model import LogisticRegression  # Importuje model regresji logistycznej do klasyfikacji.
from sklearn.metrics import ConfusionMatrixDisplay  # Importuje narzędzie do wyświetlania macierzy pomyłek.
from sklearn.metrics import accuracy_score  # Importuje funkcję obliczającą accuracy, czyli dokładność klasyfikacji.
from sklearn.metrics import classification_report  # Importuje funkcję tworzącą szczegółowy raport klasyfikacji.
from sklearn.metrics import f1_score  # Importuje funkcję obliczającą F1-score, czyli miarę łączącą precision i recall.
from sklearn.metrics import precision_score  # Importuje funkcję obliczającą precision, czyli precyzję modelu.
from sklearn.metrics import recall_score  # Importuje funkcję obliczającą recall, czyli czułość modelu.
from sklearn.model_selection import train_test_split  # Importuje funkcję dzielącą dane na zbiór treningowy i testowy.
from sklearn.pipeline import Pipeline  # Importuje Pipeline, który łączy kilka kroków przetwarzania i modelowania.
from sklearn.preprocessing import StandardScaler  # Importuje StandardScaler, który standaryzuje cechy liczbowe.

zbior = load_breast_cancer(as_frame=True)  # Wczytuje zbiór Breast Cancer w formie tabel pandas dzięki parametrowi as_frame=True.

pd.set_option("display.max_columns", None)  # Ustawia wyświetlanie wszystkich kolumn tabeli.
pd.set_option("display.max_rows", None)  # Ustawia wyświetlanie wszystkich wierszy tabeli.
pd.set_option("display.width", None)  # Wyłącza ograniczenie szerokości wyświetlania tabeli.
pd.set_option("display.max_colwidth", None)  # Wyłącza ograniczenie szerokości zawartości pojedynczej kolumny.

print("Pierwsze wiersze danych:")
print(zbior.frame.head().to_string())  # Wyświetla pierwsze wiersze pełnej tabeli danych jako tekst, bez ucinania kolumn.

X = zbior.data  # Zapisuje do zmiennej X cechy, czyli dane wejściowe modelu.
y = zbior.target  # Zapisuje do zmiennej y etykiety klas, czyli informację o typie nowotworu.

print("Kształt X:", X.shape)  # Wyświetla rozmiar X, czyli liczbę obserwacji i liczbę cech.
print("Kształt y:", y.shape)  # Wyświetla rozmiar y, czyli liczbę etykiet klas.
print("Klasa:", zbior.target_names)  # Wyświetla nazwy klas: malignant i benign.

X_train, X_test, y_train, y_test = train_test_split(  # Dzieli dane na część treningową i testową.
    X,  # Przekazuje cechy wejściowe.
    y,  # Przekazuje etykiety klas.
    test_size=0.2,  # Określa, że 20% danych zostanie przeznaczone na zbiór testowy.
    random_state=42,  # Ustawia stałą losowość, aby podział danych był powtarzalny.
    stratify=y,  # Zachowuje podobne proporcje klas w zbiorze treningowym i testowym.
)  # Kończy wywołanie funkcji train_test_split.

model = Pipeline(steps=[  # Tworzy pipeline, czyli połączony proces skalowania danych i trenowania modelu.
    ("skalowanie", StandardScaler()),  # Standaryzuje cechy, aby miały średnią 0 i odchylenie standardowe 1.
    ("model", LogisticRegression(max_iter=2000, random_state=42)),  # Dodaje model regresji logistycznej z większą liczbą iteracji i stałą losowością.
])  # Kończy definicję pipeline'u.

model.fit(X_train, y_train)  # Uczy cały pipeline na danych treningowych.
y_pred = model.predict(X_test)  # Przewiduje klasy dla danych testowych.

print("Metryki:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))  # Oblicza accuracy, czyli odsetek wszystkich poprawnych predykcji.
print("Precision:", round(precision_score(y_test, y_pred), 3))  # Oblicza precision, czyli jak często predykcja klasy pozytywnej była poprawna.
print("Recall:", round(recall_score(y_test, y_pred), 3))  # Oblicza recall, czyli ile rzeczywistych przypadków klasy pozytywnej model poprawnie wykrył.
print("F1-score:", round(f1_score(y_test, y_pred), 3))  # Oblicza F1-score, czyli średnią harmoniczną precision i recall.

print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=zbior.target_names))  # Wyświetla raport z precision, recall, F1-score i support dla każdej klasy.

ConfusionMatrixDisplay.from_predictions(  # Tworzy macierz pomyłek na podstawie wartości rzeczywistych i przewidywanych.
    y_test,  # Przekazuje prawdziwe etykiety ze zbioru testowego.
    y_pred,  # Przekazuje etykiety przewidziane przez model.
    display_labels=zbior.target_names,  # Ustawia nazwy klas wyświetlane na macierzy pomyłek.
)  # Kończy tworzenie macierzy pomyłek.

plt.title("Macierz pomyłek")  # Dodaje tytuł wykresu.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres macierzy pomyłek.

print("Interpretacja:")
print("1. Accuracy mówi, jaki procent decyzji był poprawny.")
print("2. Precision mówi, jak często predykcja pozytywna była trafna.")
print("3. Recall mówi, ile prawdziwych pozytywów wykrył model.")
print("4. F1 łączy precision i recall.")