import matplotlib.pyplot as plt  # Importuje moduł pyplot z biblioteki matplotlib do tworzenia wykresów.
import pandas as pd  # Importuje bibliotekę pandas, używaną tutaj do utworzenia tabeli z danymi i etykietami klastrów.
from sklearn.cluster import KMeans  # Importuje algorytm K-Means, który służy do grupowania danych.
from sklearn.datasets import make_blobs  # Importuje funkcję tworzącą sztuczne dane w postaci skupisk punktów.
from sklearn.metrics import silhouette_score  # Importuje metrykę silhouette_score do oceny jakości klasteryzacji.

X, y_prawdziwe = make_blobs(  # Tworzy sztuczny zbiór danych z kilkoma naturalnymi grupami punktów.
    n_samples=500,  # Określa liczbę wszystkich wygenerowanych obserwacji.
    centers=4,  # Określa liczbę rzeczywistych centrów, czyli naturalnych grup w danych.
    cluster_std=2,  # Określa rozproszenie punktów wokół centrów; większa wartość oznacza bardziej rozciągnięte klastry.
    random_state=42,  # Ustawia stałą losowość, aby dane były takie same przy każdym uruchomieniu.
)  # Kończy wywołanie funkcji make_blobs.

plt.figure(figsize=(8, 6))  # Tworzy nowy wykres o rozmiarze 8 na 6 cali.
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)  # Rysuje punkty danych przed klasteryzacją, używając pierwszej i drugiej cechy.

plt.xlabel("Cecha 1")  # Dodaje opis osi X.
plt.ylabel("Cecha 2")  # Dodaje opis osi Y.
plt.title("Dane przed grupowaniem")  # Dodaje tytuł wykresu.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.

model = KMeans(  # Tworzy model K-Means, który będzie dzielił dane na klastry.
    n_clusters=4,  # Ustawia liczbę klastrów, na które model ma podzielić dane.
    random_state=42,  # Ustawia stałą losowość, aby wynik grupowania był powtarzalny.
    n_init=10,  # Określa, ile razy algorytm ma wystartować z różnymi początkowymi centroidami.
)  # Kończy definicję modelu K-Means.

etykiety_klastow = model.fit_predict(X)  # Dopasowuje model do danych i przypisuje każdej obserwacji numer klastra.
centroidy = model.cluster_centers_  # Pobiera współrzędne środków klastrów, czyli centroidów.

silhouette = silhouette_score(X, etykiety_klastow)  # Oblicza silhouette score, czyli miarę jakości podziału na klastry.

print("Wynik:")
print(round(silhouette, 3))  # Wyświetla wartość silhouette score zaokrągloną do trzech miejsc po przecinku.

tabela = pd.DataFrame(X, columns=["Cecha 1", "Cecha 2"])  # Tworzy tabelę pandas z dwiema cechami wygenerowanych danych.
tabela["klaster"] = etykiety_klastow  # Dodaje do tabeli kolumnę z numerem klastra przypisanym przez K-Means.

print("Pierwsze 10 obserwacji")
print(tabela.head(10))  # Wyświetla pierwsze 10 obserwacji razem z przypisanym klastrem.

plt.figure(figsize=(8, 6))  # Tworzy nowy wykres o rozmiarze 8 na 6 cali.

plt.scatter(  # Rysuje punkty danych po klasteryzacji.
    X[:, 0],  # Ustawia pierwszą cechę jako wartości na osi X.
    X[:, 1],  # Ustawia drugą cechę jako wartości na osi Y.
    c=etykiety_klastow,  # Koloruje punkty według przypisanego klastra.
    alpha=0.6,  # Ustawia przezroczystość punktów, aby wykres był czytelniejszy.
)  # Kończy rysowanie punktów danych.

plt.scatter(  # Rysuje centroidy, czyli środki klastrów.
    centroidy[:, 0],  # Ustawia pierwszą współrzędną centroidów na osi X.
    centroidy[:, 1],  # Ustawia drugą współrzędną centroidów na osi Y.
    marker="x",  # Ustawia znacznik centroidów jako znak X.
    s=250,  # Ustawia rozmiar znaczników centroidów.
    linewidths=2,  # Ustawia grubość linii znacznika X.
)  # Kończy rysowanie centroidów.

plt.xlabel("Cecha 1")  # Dodaje opis osi X.
plt.ylabel("Cecha 2")  # Dodaje opis osi Y.
plt.title("K-Means - klastry i środki klastrów")  # Dodaje tytuł wykresu.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.

wyniki = []  # Tworzy pustą listę, do której będą zapisywane wyniki dla różnych wartości k.

for k in range(2, 8):  # Sprawdza różne liczby klastrów od 2 do 7.
    model_k = KMeans(  # Tworzy model K-Means dla aktualnej liczby klastrów.
        n_clusters=k,  # Ustawia aktualną liczbę klastrów.
        random_state=42,  # Ustawia stałą losowość, aby wyniki były powtarzalne.
        n_init=10,  # Uruchamia algorytm kilka razy z różnymi startowymi centroidami.
    )  # Kończy definicję modelu dla aktualnego k.

    etykiety_k = model_k.fit_predict(X)  # Dopasowuje model i przypisuje obserwacje do klastrów dla aktualnego k.
    wyniki.append({  # Dodaje wyniki aktualnego modelu do listy.
        "k": k,  # Zapisuje aktualną liczbę klastrów.
        "inertia": round(model_k.inertia_, 3),  # Zapisuje inertia, czyli sumę odległości punktów od ich centroidów.
        "silhouette": round(silhouette_score(X, etykiety_k), 3),  # Oblicza silhouette score dla aktualnego podziału.
    })  # Kończy dodawanie wyników dla aktualnego k.

tabela_k = pd.DataFrame(wyniki)  # Tworzy tabelę pandas z wynikami dla różnych wartości k.

print("Porównanie różnych k:")
print(tabela_k)  # Wyświetla tabelę porównującą inertia i silhouette dla różnych liczb klastrów.

plt.figure(figsize=(8, 6))  # Tworzy nowy wykres o rozmiarze 8 na 6 cali.
plt.plot(tabela_k["k"], tabela_k["silhouette"], marker="o")  # Rysuje wykres pokazujący zmianę silhouette score dla różnych wartości k.

plt.xlabel("Liczba klastrów k")  # Dodaje opis osi X.
plt.ylabel("Metryka oceny")  # Dodaje opis osi Y.
plt.title("Dobór liczby klastrów")  # Dodaje tytuł wykresu.

plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.

print("Interpretacja:")
print("1. K-Means nie potrzebuje etykiet klas.")
print("2. Algorytm sam szuka podobnych punktów w przestrzeni cech.")
print("3. Silhouette pomaga ocenić jakość grupowania.")