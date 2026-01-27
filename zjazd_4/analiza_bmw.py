import pandas as pd  # importuje pandas do pracy na danych tabelarycznych (DataFrame)
import numpy as np  # importuje numpy do obliczeń numerycznych i funkcji pomocniczych
import matplotlib.pyplot as plt  # importuje matplotlib do tworzenia wykresów
import seaborn as sns  # importuje seaborn do estetycznych wykresów statystycznych

# opcjonalnie ustawienia wyświetlania  # sekcja ustawień wyglądu wydruków w konsoli/notebooku
pd.set_option("display.width", 160)  # ustawia szerokość wydruku tabel, żeby mniej zawijało wiersze
pd.set_option("display.max_columns", 50)  # ustawia maksymalną liczbę widocznych kolumn przy wyświetlaniu DataFrame
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")  # formatuje liczby float do 2 miejsc po przecinku

sciezka = "bmw.csv"  # zapisuje ścieżkę/nazwę pliku wejściowego z danymi
dane = pd.read_csv(sciezka, encoding="utf-8-sig")  # wczytuje plik CSV do DataFrame; utf-8-sig pomaga przy BOM
dane.columns = [c.strip().lower().replace(" ", "_") for c in dane.columns]  # normalizuje nazwy kolumn: trim, lower, spacje -> podkreślenia
#print("Wymiary danych (wiersze, kolumny):", dane.shape)  # wypisuje rozmiar danych (kontrola liczby wierszy i kolumn)
#print("\nPodgląd danych (pierwsze 5 wierszy):")  # wypisuje nagłówek przed podglądem
#print(dane.head())  # pokazuje pierwsze 5 wierszy, żeby szybko sprawdzić strukturę danych

# podstawowe czyszczenie  # sekcja czyszczenia i rzutowania typów
dane = dane.drop_duplicates().copy()  # usuwa duplikaty wierszy i robi kopię, aby pracować na „czystym” obiekcie
for kol in ["year", "price", "mileage", "tax", "mpg", "enginesize"]:  # lista kolumn, które powinny być liczbami
    if kol in dane.columns:  # zabezpiecza przed błędem, gdyby kolumny nie było w pliku
        dane[kol] = pd.to_numeric(dane[kol], errors="coerce")  # konwertuje na liczbę; błędne wartości zamienia na NaN

braki = dane.isna().sum().sort_values(ascending=False)  # liczy braki w kolumnach i sortuje od największej liczby braków
#print("\nBraki danych (kolumny z największą liczbą braków):")  # wypisuje nagłówek sekcji braków danych
#print(braki.head(10))  # pokazuje 10 kolumn z największą liczbą braków

dane = dane.dropna(subset=["price", "year", "mileage"])  # usuwa wiersze bez kluczowych pól (cena/rok/przebieg), bo bez nich analiza traci sens

dane["wiek_auta"] = (dane["year"].max() - dane["year"]).astype(int)  # tworzy wiek auta jako różnicę od maksymalnego rocznika w zbiorze

dane["przebieg_na_rok"] = dane["mileage"] / np.maximum(dane["wiek_auta"], 1)  # liczy przebieg na rok; minimum 1 rok chroni przed dzieleniem przez 0

dane["automat"] = np.where(dane["transmission"].str.lower().eq("automatic"), 1, 0)  # koduje skrzynię automatyczną jako 1, resztę jako 0

dane["diesel"] = np.where(dane["fueltype"].str.lower().eq("diesel"), 1, 0)  # koduje diesel jako 1, inne paliwa jako 0

dane["cena_na_1000_km"] = dane["price"] / np.where(dane["mileage"] == 0, np.nan, dane["mileage"] / 1000)  # liczy cenę na 1000 km; przy 0 km ustawia NaN

dane["log_cena"] = np.log(dane["price"])  # tworzy logarytm ceny (ułatwia analizę rozkładów skośnych i modele statystyczne)
dane["log_przebieg"] = np.log1p(dane["mileage"])  # tworzy log(1+przebieg), bezpieczne dla 0 i stabilne numerycznie

dane["segment_silnika"] = pd.cut(  # tworzy zmienną kategoryczną segmentu pojemności silnika
    dane["enginesize"],  # wskazuje kolumnę z pojemnością silnika do pocięcia na przedziały
    bins=[0.0, 1.5, 2.0, 2.5, 3.0, 10.0],  # granice przedziałów pojemności (binów)
    labels=["<=1.5", "1.6-2.0", "2.1-2.5", "2.6-3.0", ">3.0"],  # etykiety dla przedziałów, żeby wyniki były czytelne
    include_lowest=True  # włącza dolną granicę pierwszego przedziału, żeby nie tracić minimalnych wartości
)  # kończy tworzenie segmentów silnika

kpi = {  # słownik z kluczowymi wskaźnikami podsumowującymi zbiór
    "liczba_ofert": int(len(dane)),  # liczba ofert (wierszy) po czyszczeniu
    "liczba_modeli": int(dane["model"].nunique()),  # liczba unikalnych modeli w danych
    "rok_min": int(dane["year"].min()),  # najstarszy rocznik w danych
    "rok_max": int(dane["year"].max()),  # najnowszy rocznik w danych
    "mediana_ceny": float(dane["price"].median()),  # mediana ceny (odporna na wartości odstające)
    "srednia_ceny": float(dane["price"].mean()),  # średnia cena (wrażliwa na wartości odstające)
    "mediana_przebiegu": float(dane["mileage"].median()),  # mediana przebiegu
    "udzial_automat": float(dane["automat"].mean()),  # udział automatów (średnia z 0/1 = odsetek)
    "udzial_diesel": float(dane["diesel"].mean()),  # udział diesli (średnia z 0/1 = odsetek)
}  # koniec definicji KPI

#print("\n=== KPI (podstawowe informacje o danych) ===")  # drukuje nagłówek sekcji KPI
#print(pd.DataFrame([kpi]))  # wyświetla KPI w formie tabeli (1 wiersz)

top_modele = dane["model"].value_counts().head(10).rename_axis("model").reset_index(name="liczba_ofert")  # wylicza TOP 10 modeli wg liczby ofert
#print("\n=== TOP 10 modeli (najwięcej ofert) ===")  # drukuje nagłówek sekcji TOP modeli
#print(top_modele)  # wyświetla tabelę TOP modeli

ceny_wg_roku = (dane.groupby("year", as_index=False).agg(  # agreguje dane po roczniku, żeby zobaczyć trendy cenowe
    liczba=("price", "size"),  # liczy liczbę ofert w danym roczniku
    mediana_ceny=("price", "median"),  # liczy medianę ceny w roczniku
    srednia_ceny=("price", "mean"),  # liczy średnią cenę w roczniku
    mediana_przebiegu=("mileage", "median")  # liczy medianę przebiegu w roczniku
).sort_values("year")  # sortuje po roczniku rosnąco, żeby wykres/tabela miały logiczny porządek
)  # kończy budowę tabeli cen wg roku

#print("\n=== Ceny wg roku (ostatnie 10 roczników w danych) ===")  # drukuje nagłówek sekcji cen wg roku
#print(ceny_wg_roku.tail(10))  # pokazuje 10 ostatnich roczników z tabeli (zwykle najnowsze lata)

pivot_paliwo_skrzynia = pd.pivot_table(  # tworzy tabelę przestawną do porównania median cen między paliwem i skrzynią
    dane,  # dane wejściowe
    values="price",  # agregowana wartość: cena
    index="fueltype",  # wiersze: typ paliwa
    columns="transmission",  # kolumny: typ skrzyni
    aggfunc="median"  # agregacja: mediana (stabilniejsza na odstające ceny)
)  # kończy tworzenie pivotu

#print("\n=== Pivot: mediana ceny wg paliwa i skrzyni ===")  # drukuje nagłówek pivotu
#print(pivot_paliwo_skrzynia)  # wyświetla tabelę przestawną

segmenty = (  # przygotowuje statystyki wg segmentu pojemności silnika
    dane.groupby("segment_silnika", as_index=False).agg(  # grupuje po segmencie silnika i liczy agregaty
        liczba=("price", "size"),  # liczba ofert w segmencie
        mediana_ceny=("price", "median"),  # mediana ceny w segmencie
        sredni_przebieg=("mileage", "mean"),  # średni przebieg w segmencie
        udzial_diesel=("diesel", "mean"),  # udział diesla w segmencie
        udzial_automat=("automat", "mean")  # udział automatu w segmencie
    ).sort_values("mediana_ceny", ascending=False)  # sortuje segmenty od najdroższych medianowo
)  # kończy tabelę segmentów

#print("\n=== Segmenty silnika: statystyki ===")  # drukuje nagłówek statystyk segmentów
#print(segmenty)  # wyświetla tabelę segmentów

#wykresy  # sekcja generowania wykresów (wyświetlanie sterowane flagą SHOW_PLOTS)

SHOW_PLOTS = False  # steruje wyświetlaniem wykresu: False nie pokazuje, True pokazuje
plt.figure()  # tworzy nową figurę dla wykresu liniowego
plt.plot(ceny_wg_roku["year"], ceny_wg_roku["mediana_ceny"], marker="o")  # rysuje medianę ceny w funkcji rocznika
plt.title("Mediana ceny BMW wg roku")  # ustawia tytuł wykresu
plt.xlabel("Rok")  # ustawia podpis osi X
plt.ylabel("Mediana ceny")  # ustawia podpis osi Y
plt.tight_layout()  # dopasowuje marginesy i rozmieszczenie elementów, żeby nic się nie nakładało
if SHOW_PLOTS:  # sprawdza, czy wykres ma zostać pokazany
    plt.show()  # wyświetla wykres
else:  # alternatywa, gdy nie chcemy okna/wyjścia z wykresem
    plt.close()  # zamyka figurę, żeby nie zajmowała pamięci i nie generowała wyjścia

SHOW_PLOTS = False  # osobna flaga dla kolejnego wykresu (możesz niezależnie ustawić True)
probka = dane.sample(n=min(1500, len(dane)), random_state=42)  # losuje próbkę do 1500 wierszy dla szybszego wykresu i mniejszego „szumu”
plt.figure()  # tworzy nową figurę dla wykresu punktowego
plt.scatter(probka["mileage"], probka["price"])  # rysuje zależność cena vs przebieg na próbce
plt.title("Cena vs przebieg (próbka 1500)")  # ustawia tytuł wykresu
plt.xlabel("Przebieg")  # podpis osi X
plt.ylabel("Cena")  # podpis osi Y
plt.tight_layout()  # poprawia układ wykresu
if SHOW_PLOTS:  # sprawdza, czy pokazać wykres
    plt.show()  # wyświetla wykres
else:  # gdy nie wyświetlamy
    plt.close()  # zamyka figurę

SHOW_PLOTS = False  # flaga sterująca dla boxplota
plt.figure()  # tworzy nową figurę
sns.boxplot(data=dane, x="transmission", y="price")  # rysuje boxplot cen w podziale na rodzaj skrzyni biegów
plt.title("Rozkład ceny wg skrzyni biegów")  # ustawia tytuł wykresu
plt.xlabel("Skrzynia biegów")  # podpis osi X
plt.ylabel("Cena")  # podpis osi Y
plt.tight_layout()  # dopasowuje elementy wykresu
if SHOW_PLOTS:  # sprawdza, czy pokazać
    plt.show()  # wyświetla
else:  # gdy nie pokazujemy
    plt.close()  # zamyka figurę

SHOW_PLOTS = False  # flaga sterująca dla histogramu
plt.figure()  # tworzy nową figurę
sns.histplot(data=dane, x="price", kde=True, bins=40)  # rysuje histogram cen i KDE, żeby zobaczyć kształt rozkładu
plt.title("Rozkład ceny (histogram + KDE)")  # ustawia tytuł wykresu
plt.xlabel("Cena")  # podpis osi X
plt.ylabel("Liczba ofert")  # podpis osi Y
plt.tight_layout()  # dopasowuje układ
if SHOW_PLOTS:  # sprawdza, czy pokazać wykres
    plt.show()  # wyświetla wykres
else:  # gdy nie pokazujemy
    plt.close()  # zamyka figurę

#grupujemy dane okazje: najtansze oferty wzgledem modelu i roku  # sekcja wyszukiwania ofert „poniżej typowej ceny” w grupie model+rok

grupy = dane.groupby(["model", "year"])  # grupuje dane po modelu i roczniku, aby porównywać auta o podobnej specyfikacji bazowej
mediany = grupy["price"].median().rename("mediana_grupy").reset_index()  # liczy medianę ceny w każdej grupie i zamienia wynik na tabelę
polaczone = dane.merge(mediany, on=["model", "year"], how="left")  # dołącza medianę grupy do każdego wiersza, żeby móc policzyć różnicę

polaczone["odchylenie_od_mediany"] = polaczone["price"] - polaczone["mediana_grupy"]  # oblicza różnicę ceny od mediany grupy (ujemne = potencjalna okazja)

okazje = polaczone.sort_values("odchylenie_od_mediany").head(20)  # wybiera 20 ofert najbardziej „poniżej mediany” (najniższe odchylenie)
#print("\n=== TOP 20 okazji (najtańsze względem model+rok) ===")  # drukuje nagłówek sekcji okazji
#print(okazje[["model", "year", "price", "mileage", "fueltype", "transmission", "odchylenie_od_mediany"]])  # wyświetla kluczowe kolumny dla okazji

#zapis wyników do plików csv  # sekcja eksportu danych do plików do dalszej analizy/raportowania
dane.to_csv("bmw_clean.csv", index=False, encoding="utf-8-sig")  # zapisuje oczyszczone dane do pliku CSV
okazje.to_csv("okazje_bmw_top_20.csv", index=False, encoding="utf-8-sig")  # zapisuje TOP 20 okazji do pliku CSV
ceny_wg_roku.to_csv("bmw_ceny_wg_roku.csv", index=False, encoding="utf-8-sig")  # zapisuje agregacje cen wg roku do pliku CSV
segmenty.to_csv("bmw_segmenty_silnika.csv", index=False, encoding="utf-8-sig")  # zapisuje statystyki wg segmentu silnika do pliku CSV

print("\nZapisano pliki: bmw_clean.csv, okazje_bmw_top_20.csv, bmw_ceny_wg_roku.csv, bmw_segmenty_silnika.csv")  # wypisuje komunikat potwierdzający zapis plików
