import matplotlib.pyplot as plt  # Importuje moduł pyplot z biblioteki matplotlib do tworzenia wykresów.
from sklearn.datasets import load_digits  # Importuje funkcję wczytującą gotowy zbiór danych z cyframi pisanymi odręcznie.
from sklearn.metrics import accuracy_score  # Importuje funkcję obliczającą accuracy, czyli dokładność klasyfikacji.
from sklearn.metrics import classification_report  # Importuje funkcję tworzącą szczegółowy raport klasyfikacji.
from sklearn.model_selection import train_test_split  # Importuje funkcję dzielącą dane na zbiór treningowy i testowy.
from sklearn.neural_network import MLPClassifier  # Importuje klasyfikator sieci neuronowej typu MLP.
from sklearn.pipeline import Pipeline  # Importuje Pipeline, który łączy kilka kroków w jeden proces.
from sklearn.preprocessing import StandardScaler  # Importuje StandardScaler, który standaryzuje cechy liczbowe.

zbior = load_digits()  # Wczytuje zbiór digits, czyli obrazy cyfr zapisane jako dane liczbowe.

X = zbior.data  # Zapisuje do zmiennej X dane wejściowe, czyli spłaszczone obrazy cyfr.
y = zbior.target  # Zapisuje do zmiennej y etykiety, czyli prawdziwe cyfry od 0 do 9.

print("Kształt X:")
print(X.shape)  # Wyświetla rozmiar X, czyli liczbę obrazów oraz liczbę cech dla każdego obrazu.

print("Kształt y:")
print(y.shape)  # Wyświetla rozmiar y, czyli liczbę etykiet odpowiadających obrazom.

print("Dostępne klasy:")
print(sorted(set(y)))  # Wyświetla posortowaną listę unikalnych klas, czyli cyfr występujących w zbiorze.

print("Pierwszy rekord X:")
print(X[0])  # Wyświetla pierwszy obraz zapisany jako jednowymiarowa tablica 64 wartości.

print("Pierwszy obraz 8x8:")
print(zbior.images[0])  # Wyświetla pierwszy obraz w formie macierzy 8 na 8 pikseli.

print("Etykieta pierwszego obrazu:", y[0])  # Wyświetla prawdziwą etykietę pierwszego obrazu.

plt.figure(figsize=(4, 4))  # Tworzy nowy wykres o rozmiarze 4 na 4 cale.

plt.imshow(zbior.images[0], cmap="gray")  # Wyświetla pierwszy obraz jako obraz w skali szarości.

plt.title(f"Etykieta pierwszego obrazu: {y[0]}")  # Dodaje tytuł wykresu z etykietą pierwszego obrazu.
plt.axis("off")  # Wyłącza osie, aby obraz był czytelniejszy.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.

fig, axes = plt.subplots(2, 5, figsize=(10, 5))  # Tworzy siatkę 2 na 5 wykresów do pokazania kilku obrazów naraz.

for i, ax in enumerate(axes.ravel()):  # Przechodzi po kolejnych miejscach na wykresie i przypisuje im indeksy obrazów.
    ax.imshow(zbior.images[i], cmap="gray")  # Wyświetla kolejny obraz cyfry w skali szarości.
    ax.set_title(f"Etykieta {y[i]}")  # Ustawia tytuł pojedynczego obrazka z jego prawdziwą etykietą.
    ax.axis("off")  # Wyłącza osie dla pojedynczego obrazka.

plt.tight_layout()  # Dopasowuje rozmieszczenie obrazów, aby tytuły i wykresy nie nachodziły na siebie.
plt.show()  # Wyświetla zestaw pierwszych 10 obrazów.

X_train, X_test, y_train, y_test = train_test_split(  # Dzieli dane na zbiór treningowy i testowy.
    X,  # Przekazuje dane wejściowe, czyli obrazy zapisane jako cechy.
    y,  # Przekazuje etykiety klas, czyli cyfry od 0 do 9.
    test_size=0.2,  # Określa, że 20% danych trafi do zbioru testowego.
    random_state=42,  # Ustawia stałą losowość, aby podział danych był powtarzalny.
    stratify=y  # Zachowuje podobne proporcje cyfr w zbiorze treningowym i testowym.
)  # Kończy wywołanie funkcji train_test_split.

model = Pipeline(steps=[  # Tworzy pipeline, czyli połączony proces skalowania danych i trenowania modelu.
    ("skalowanie", StandardScaler()),  # Standaryzuje cechy, co pomaga sieci neuronowej szybciej i stabilniej się uczyć.
    ("mlp", MLPClassifier(  # Dodaje model MLP, czyli prostą wielowarstwową sieć neuronową.
        hidden_layer_sizes=(64, 32),  # Ustawia dwie warstwy ukryte: pierwszą z 64 neuronami i drugą z 32 neuronami.
        activation="relu",  # Ustawia funkcję aktywacji ReLU, która pomaga modelowi uczyć się zależności nieliniowych.
        solver="adam",  # Ustawia algorytm optymalizacji Adam, który aktualizuje wagi sieci podczas uczenia.
        alpha=0.0005,  # Ustawia siłę regularyzacji, która pomaga ograniczać przeuczenie modelu.
        max_iter=300,  # Ustawia maksymalną liczbę iteracji uczenia.
        random_state=42,  # Ustawia stałą losowość, aby wynik uczenia był bardziej powtarzalny.
    )),  # Kończy definicję modelu MLP.
])  # Kończy definicję pipeline'u.

model.fit(X_train, y_train)  # Uczy cały pipeline na danych treningowych.
y_pred = model.predict(X_test)  # Przewiduje cyfry dla danych testowych.

accuracy = accuracy_score(y_test, y_pred)  # Oblicza accuracy, czyli odsetek poprawnie rozpoznanych cyfr.

print("Accuracy:", accuracy)  # Wyświetla dokładność modelu.
print("Raport klasyfikacji")
print(classification_report(y_test, y_pred))  # Wyświetla precision, recall, F1-score i support dla każdej cyfry.

krzywa_straty = model.named_steps["mlp"].loss_curve_  # Pobiera historię wartości funkcji straty z etapu uczenia sieci neuronowej.

plt.figure(figsize=(10, 5))  # Tworzy nowy wykres o rozmiarze 10 na 5 cali.
plt.plot(krzywa_straty)  # Rysuje przebieg funkcji straty w kolejnych iteracjach uczenia.
plt.xlabel("Iteracja")  # Dodaje opis osi X.
plt.ylabel("Strata")  # Dodaje opis osi Y.
plt.title("Przebieg uczenia sieci neuronowej")  # Dodaje tytuł wykresu.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.