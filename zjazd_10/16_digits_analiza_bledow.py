import matplotlib.pyplot as plt  # Importuje moduł pyplot z biblioteki matplotlib do tworzenia wykresów.
import numpy as np  # Importuje bibliotekę NumPy, używaną tutaj do znalezienia błędnych predykcji.
from sklearn.datasets import load_digits  # Importuje funkcję wczytującą zbiór danych z cyframi pisanymi odręcznie.
from sklearn.metrics import accuracy_score  # Importuje funkcję obliczającą accuracy, czyli dokładność klasyfikacji.
from sklearn.metrics import classification_report  # Importuje funkcję tworzącą szczegółowy raport klasyfikacji.
from sklearn.model_selection import train_test_split  # Importuje funkcję dzielącą dane na zbiór treningowy i testowy.
from sklearn.neural_network import MLPClassifier  # Importuje klasyfikator sieci neuronowej MLP.
from sklearn.pipeline import Pipeline  # Importuje Pipeline, który łączy kilka kroków w jeden proces.
from sklearn.preprocessing import StandardScaler  # Importuje StandardScaler, który standaryzuje cechy liczbowe.

zbior = load_digits()  # Wczytuje zbiór digits, czyli obrazy cyfr od 0 do 9.

X = zbior.data  # Zapisuje do zmiennej X obrazy zapisane jako tablice cech.
y = zbior.target  # Zapisuje do zmiennej y etykiety, czyli prawdziwe cyfry.

X_train, X_test, y_train, y_test, obrazy_train, obrazy_test = train_test_split(  # Dzieli jednocześnie dane, etykiety i obrazy na część treningową oraz testową.
    X,  # Przekazuje dane wejściowe w formie spłaszczonych obrazów.
    y,  # Przekazuje etykiety klas, czyli cyfry od 0 do 9.
    zbior.images,  # Przekazuje obrazy w formacie 8x8, aby później można było wyświetlić błędne przykłady.
    test_size=0.2,  # Określa, że 20% danych zostanie przeznaczone na zbiór testowy.
    random_state=42,  # Ustawia stałą losowość, aby podział danych był powtarzalny.
    stratify=y  # Zachowuje podobne proporcje cyfr w zbiorze treningowym i testowym.
)  # Kończy wywołanie funkcji train_test_split.

model = Pipeline(steps=[  # Tworzy pipeline, czyli połączony proces skalowania danych i trenowania modelu.
    ("skalowanie", StandardScaler()),  # Standaryzuje cechy, aby sieć neuronowa uczyła się stabilniej.
    ("mlp", MLPClassifier(  # Dodaje model MLP, czyli prostą wielowarstwową sieć neuronową.
        hidden_layer_sizes=(64, 32),  # Ustawia dwie warstwy ukryte: pierwszą z 64 neuronami i drugą z 32 neuronami.
        max_iter=300,  # Ustawia maksymalną liczbę iteracji uczenia modelu.
        random_state=42  # Ustawia stałą losowość, aby wynik uczenia był bardziej powtarzalny.
    )),  # Kończy definicję modelu MLP.
])  # Kończy definicję pipeline'u.

model.fit(X_train, y_train)  # Uczy cały pipeline na danych treningowych.

y_pred = model.predict(X_test)  # Przewiduje cyfry dla danych testowych.

print("ACCURACY:")
print(round(accuracy_score(y_test, y_pred), 3))  # Oblicza accuracy, czyli odsetek poprawnie rozpoznanych cyfr, i zaokrągla wynik.

print("RAPORT KLASYFIKACJI:")
print(classification_report(y_test, y_pred))  # Wyświetla precision, recall, F1-score i support dla każdej cyfry.

bledy = np.where(y_test != y_pred)[0]  # Znajduje indeksy tych obserwacji, dla których predykcja modelu różni się od prawdziwej etykiety.

print("LICZBA BŁĘDNYCH PREDYKCJI:")
print(len(bledy))  # Wyświetla liczbę błędnie sklasyfikowanych obrazów.

print("PIERWSZE 10 BŁĘDÓW:")

for indeks in bledy[:10]:  # Przechodzi po maksymalnie pierwszych 10 błędnych predykcjach.
    print(  # Wyświetla prawdziwą cyfrę oraz cyfrę przewidzianą przez model.
        "Prawdziwa cyfra:",
        y_test[indeks],
        "| Predykcja modelu:",
        y_pred[indeks],
    )  # Kończy wyświetlanie informacji o jednym błędzie.

liczba_obrazow = min(10, len(bledy))  # Ustala, ile błędnych obrazów pokazać, maksymalnie 10.

plt.figure(figsize=(12, 5))  # Tworzy nowy wykres o rozmiarze 12 na 5 cali.

for i in range(liczba_obrazow):  # Przechodzi po kolejnych błędnie sklasyfikowanych obrazach do wyświetlenia.
    indeks = bledy[i]  # Pobiera indeks konkretnego błędnie sklasyfikowanego obrazu.
    plt.subplot(2, 5, i + 1)  # Tworzy miejsce na obraz w siatce 2 wiersze na 5 kolumn.
    plt.imshow(obrazy_test[indeks], cmap="gray")  # Wyświetla błędnie sklasyfikowany obraz w skali szarości.
    plt.title(f"Prawda: {y_test[indeks]}\nModel: {y_pred[indeks]}")  # Dodaje tytuł z prawdziwą cyfrą i predykcją modelu.
    plt.axis("off")  # Wyłącza osie, aby obraz był czytelniejszy.

plt.tight_layout()  # Dopasowuje rozmieszczenie obrazów i tytułów, aby nie nachodziły na siebie.
plt.show()  # Wyświetla wykres z błędnie sklasyfikowanymi obrazami.