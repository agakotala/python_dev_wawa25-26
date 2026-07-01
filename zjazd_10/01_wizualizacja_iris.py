from sklearn.datasets import load_iris  # Importuje funkcję load_iris, która wczytuje gotowy zbiór danych Iris.
import matplotlib.pyplot as plt  # Importuje moduł pyplot z biblioteki matplotlib, który służy do tworzenia wykresów.

zbior = load_iris(as_frame=True)  # Wczytuje zbiór Iris w formie tabel pandas dzięki parametrowi as_frame=True.

dane = zbior.frame.copy()  # Tworzy kopię pełnej tabeli danych, zawierającej cechy oraz kolumnę target.

dane["gatunek"] = dane["target"].map(dict(enumerate(zbior.target_names)))  # Tworzy nową kolumnę gatunek, zamieniając numery klas na ich nazwy.

print("Pierwsze 5 wierszy:")
print(dane.head())  # Wyświetla pierwsze 5 wierszy tabeli danych.

print("Średnie wartości cech w grupach:")
print(dane.groupby("gatunek")[zbior.feature_names].mean())  # Grupuje dane według gatunku i oblicza średnie wartości cech dla każdej grupy.

plt.figure(figsize=(8, 6))  # Tworzy nowy wykres o rozmiarze 8 na 6 cali.
for gatunek in zbior.target_names:  # Przechodzi po wszystkich nazwach gatunków irysów.
    podzbior = dane[dane["gatunek"] == gatunek]  # Wybiera z tabeli tylko wiersze należące do aktualnego gatunku.
    plt.scatter(  # Tworzy wykres punktowy dla aktualnego gatunku.
        podzbior["petal length (cm)"],  # Ustawia długość płatka jako wartości na osi X.
        podzbior["petal width (cm)"],  # Ustawia szerokość płatka jako wartości na osi Y.
        label=gatunek,  # Ustawia nazwę gatunku jako etykietę widoczną w legendzie.
        alpha=0.75,  # Ustawia częściową przezroczystość punktów, aby nachodzące punkty były lepiej widoczne.
    )  # Kończy wywołanie funkcji scatter.

plt.xlabel("Długość płatka")  # Dodaje opis osi X.
plt.ylabel("Szerokość płatka")  # Dodaje opis osi Y.
plt.title("Iris - długość i szerokość płatka")  # Dodaje tytuł wykresu.
plt.legend()  # Wyświetla legendę z nazwami gatunków.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.

plt.figure(figsize=(8, 6))  # Tworzy nowy wykres o rozmiarze 8 na 6 cali.
for gatunek in zbior.target_names:  # Przechodzi po wszystkich nazwach gatunków irysów.
    podzbior = dane[dane["gatunek"] == gatunek]  # Wybiera z tabeli tylko wiersze należące do aktualnego gatunku.
    plt.scatter(  # Tworzy wykres punktowy dla aktualnego gatunku.
        podzbior["sepal length (cm)"],  # Ustawia długość działki kielicha jako wartości na osi X.
        podzbior["sepal width (cm)"],  # Ustawia szerokość działki kielicha jako wartości na osi Y.
        label=gatunek,  # Ustawia nazwę gatunku jako etykietę widoczną w legendzie.
        alpha=0.75,  # Ustawia częściową przezroczystość punktów, aby nachodzące punkty były lepiej widoczne.
    )  # Kończy wywołanie funkcji scatter.

plt.xlabel("Długość działki kielicha")  # Dodaje opis osi X.
plt.ylabel("Szerokość działki kielicha")  # Dodaje opis osi Y.
plt.title("Iris - długość i szerokość działki kielicha")  # Dodaje tytuł wykresu.
plt.legend()  # Wyświetla legendę z nazwami gatunków.
plt.tight_layout()  # Dopasowuje elementy wykresu, aby nie nachodziły na siebie.
plt.show()  # Wyświetla gotowy wykres.

print("Interpretacja:")
print("1. Wykresy rozrzutu pomagają zobaczyć, czy klasy da się rozdzielić")
print("2. Jeśli klasy są dobrze rozdzielone na wykresie, prosty model może działać dobrze.")