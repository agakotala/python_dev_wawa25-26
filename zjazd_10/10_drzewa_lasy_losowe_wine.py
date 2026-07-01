import matplotlib.pyplot as plt  # Importuje moduł pyplot z biblioteki matplotlib do tworzenia wykresów.
import pandas as pd  # Importuje bibliotekę pandas, używaną tutaj do utworzenia serii z ważnością cech.
from sklearn.datasets import load_wine  # Importuje funkcję wczytującą gotowy zbiór danych Wine.
from sklearn.ensemble import RandomForestClassifier  # Importuje model Random Forest do klasyfikacji.
from sklearn.metrics import accuracy_score  # Importuje funkcję obliczającą accuracy, czyli dokładność klasyfikacji.
from sklearn.metrics import classification_report  # Importuje funkcję tworzącą szczegółowy raport klasyfikacji.
from sklearn.model_selection import train_test_split  # Importuje funkcję dzielącą dane na zbiór treningowy i testowy.
from sklearn.tree import DecisionTreeClassifier  # Importuje klasyfikator drzewa decyzyjnego.
from sklearn.tree import plot_tree  # Importuje funkcję do rysowania drzewa decyzyjnego.


zbior = load_wine(as_frame=True)  # Wczytuje zbiór Wine w formie tabel pandas dzięki parametrowi as_frame=True.
X = zbior.data  # Zapisuje do zmiennej X cechy, czyli dane wejściowe modelu.
y = zbior.target  # Zapisuje do zmiennej y etykiety klas, czyli rodzaje win.

X_train, X_test, y_train, y_test = train_test_split(  # Dzieli dane na zbiór treningowy i testowy.
    X,  # Przekazuje cechy wejściowe.
    y,  # Przekazuje etykiety klas.
    test_size=0.25,  # Określa, że 25% danych zostanie przeznaczone na zbiór testowy.
    random_state=42,  # Ustawia stałą losowość, aby podział danych był powtarzalny.
    stratify=y  # Zachowuje podobne proporcje klas w zbiorze treningowym i testowym.
)  # Kończy wywołanie funkcji train_test_split.

drzewo = DecisionTreeClassifier(max_depth=5, random_state=42)  # Tworzy drzewo decyzyjne z maksymalną głębokością 5 i stałą losowością.

las = RandomForestClassifier(n_estimators=50, random_state=42)  # Tworzy las losowy składający się z 50 drzew decyzyjnych.

modele = {  # Tworzy słownik z nazwami modeli oraz odpowiadającymi im obiektami modeli.
    "Drzewo decyzyjne": drzewo,  # Dodaje pojedyncze drzewo decyzyjne do słownika modeli.
    "Random Forest": las,  # Dodaje model Random Forest do słownika modeli.
}  # Kończy tworzenie słownika modeli.

for nazwa, model in modele.items():  # Przechodzi po każdym modelu ze słownika, pobierając jego nazwę i obiekt.
    model.fit(X_train, y_train)  # Uczy aktualny model na danych treningowych.
    y_pred = model.predict(X_test)  # Przewiduje klasy dla danych testowych.
    print("Model:", nazwa)  # Wyświetla nazwę aktualnie ocenianego modelu.
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))  # Oblicza accuracy, czyli odsetek poprawnych klasyfikacji.
    print(classification_report(y_test, y_pred, target_names=zbior.target_names))  # Wyświetla precision, recall, F1-score i support dla każdej klasy.

plt.figure(figsize=(16, 8))  # Tworzy nowy wykres o rozmiarze 16 na 8 cali.

plot_tree(  # Rysuje strukturę drzewa decyzyjnego.
    drzewo,  # Przekazuje wytrenowane drzewo decyzyjne do narysowania.
    feature_names=X.columns,  # Ustawia nazwy cech, które będą widoczne w węzłach drzewa.
    class_names=zbior.target_names,  # Ustawia nazwy klas, które będą widoczne na wykresie drzewa.
    filled=True,  # Koloruje węzły drzewa, aby łatwiej było odczytać dominującą klasę.
    max_depth=3,  # Pokazuje tylko pierwsze 3 poziomy drzewa, aby wykres był czytelniejszy.
)  # Kończy rysowanie drzewa.

plt.title("Uproszczone drzewo decyzyjne")  # Dodaje tytuł wykresu drzewa decyzyjnego.

plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.

plt.show()  # Wyświetla gotowy wykres drzewa decyzyjnego.

waznosc = pd.Series(las.feature_importances_, index=X.columns).sort_values()  # Tworzy serię z ważnością cech Random Forest i sortuje ją rosnąco.

plt.figure(figsize=(8, 7))  # Tworzy nowy wykres o rozmiarze 8 na 7 cali.
waznosc.tail(10).plot(kind="barh")  # Rysuje poziomy wykres słupkowy dla 10 najważniejszych cech.
plt.title("Random Forest - 10 najważniejszych cech")  # Dodaje tytuł wykresu ważności cech.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.

print("Interpretacja")
print("1. Drzewo decyzyjne tworzy reguły typu 'jeśli.... to....'.")
print("2. Pojedyncze drzewo jest łatwiejsze w interpretacji, ale może się przeuczyć.")
print("3. Random Forest łączy wiele drzew, dzięki temu jest zwykle stabilniejszy.")
print("4. Ważność cech pokazuje, które zmienne był użyteczne dla modelu.")