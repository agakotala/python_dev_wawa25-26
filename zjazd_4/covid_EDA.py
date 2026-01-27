import pandas as pd  # pandas: wczytywanie/obrabianie danych tabelarycznych (DataFrame)
import numpy as np  # numpy: obliczenia numeryczne, NaN, wektoryzacja
import matplotlib.pyplot as plt  # matplotlib: tworzenie wykresów
import matplotlib.dates as mdates  # narzędzia do formatowania osi czasu (daty) na wykresach
from pathlib import Path  # Path: wygodne operacje na ścieżkach do plików
from datetime import datetime  # datetime: czas analizy + obsługa dat w danych
from textwrap import fill  # fill: łamie długie teksty na kilka linii (np. legendy/etykiety)
from io import StringIO  # StringIO: bufor tekstowy w pamięci do przechwytywania printów
import base64  # base64: kodowanie obrazków do wklejenia w HTML jako data URI
from contextlib import redirect_stdout  # redirect_stdout: przekierowanie printów do bufora zamiast do konsoli

INPUT_PATH = Path("ewp_dsh_zakazenia_po_szczepieniu_202203020921.csv")  # ścieżka do pliku wejściowego CSV
OUTPUT_DIR = Path("eda_output")  # katalog wyjściowy, gdzie zapisze się raport
OUTPUT_DIR.mkdir(exist_ok=True)  # tworzy katalog jeśli nie istnieje; jeśli istnieje, nie zgłasza błędu

ANALYSIS_TIME = datetime.now()  # zapisuje moment uruchomienia analizy (timestamp do raportu)
STAMP = ANALYSIS_TIME.strftime("%Y-%m-%d_%H-%M-%S")  # formatuje czas do bezpiecznej nazwy pliku
REPORT_HTML = OUTPUT_DIR / f"EDA_report_{STAMP}.html"  # buduje ścieżkę do finalnego pliku HTML z raportem

W = "liczba_zaraportowanych_zakazonych"  # nazwa kolumny „waga”: wyniki liczymy jako sumę tej kolumny

plt.style.use("seaborn-v0_8-whitegrid")  # ustawia czytelny styl wykresów (siatka, sensowne domyślne odstępy)

PASTEL = plt.cm.Pastel1(np.linspace(0, 1, 9))  # tworzy paletę pastelowych kolorów (tablica kolorów) do wykresów
BRIGHT = plt.cm.tab10(np.linspace(0, 1, 10))  # tworzy paletę wyraźnych kolorów (tablica kolorów) do wykresów

# funkcje pomocnicze dla wykresow  # sekcja narzędzi do skracania etykiet i formatowania osi

def shorten_labels(labels, maxlen=26):  # funkcja skracająca długie etykiety do maksymalnej długości maxlen
    out = []  # lista na skrócone etykiety
    for x in labels:  # iteruje po wszystkich etykietach wejściowych
        s = str(x)  # zamienia etykietę na tekst (żeby działało dla różnych typów)
        if len(s) <= maxlen:  # jeśli etykieta jest krótka, nie trzeba skracać
            out.append(s)  # dodaje etykietę bez zmian
        else:  # jeśli etykieta za długa
            out.append(s[:maxlen - 1] + "...")  # przycina i dodaje wielokropek dla czytelności
    return out  # zwraca listę gotowych etykiet

def wrap_labels(labels, width=18):  # funkcja łamiąca tekst etykiet na kilka linii o szerokości width
    return [fill(str(x), width=width) for x in labels]  # zwraca listę etykiet po „zawinięciu” tekstu

def format_date_axis(ax):  # funkcja formatująca oś X jako daty w czytelny sposób
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)  # wybiera rozsądne rozmieszczenie ticków na osi czasu
    ax.xaxis.set_major_locator(locator)  # ustawia lokalizator ticków na osi X
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))  # ustawia zwięzły format dat dopasowany do skali

def fig_to_base64(fig):  # funkcja zamieniająca wykres (figure) na base64 do wklejenia w HTML
    """
    zapisuje figurę do bufora w pamięci jako PNG,
    koduje na base64 i zwraca tekst (string),
    który można wkleić w HTML w tagu <img>.
    """  # docstring: opisuje co robi funkcja i po co (do raportu HTML bez plików PNG osobno)
    from io import BytesIO  # importuje BytesIO lokalnie (bufor bajtowy do trzymania obrazka w pamięci)
    buf = BytesIO()  # tworzy bufor bajtowy w RAM
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")  # zapisuje figurę do bufora jako PNG w dobrej jakości
    plt.close(fig)  # zamyka figurę, żeby nie trzymać jej w pamięci i nie dublować wykresów
    return base64.b64encode(buf.getvalue()).decode("utf-8")  # koduje PNG do base64 i zwraca jako tekst

def barh_top(series, n=15, title="", xlabel="", color=None):  # tworzy wykres poziomych słupków dla TOP n wartości serii
    s = series.dropna()  # usuwa braki, żeby nie psuły sortowania i rysowania
    s = s.sort_values(ascending=False)  # sortuje malejąco (największe na górze)
    s = s.head(n)  # bierze TOP n kategorii
    s = s.sort_values()  # odwraca kolejność dla barh (żeby największe były na końcu i ładnie wyglądały)
    labels = shorten_labels(s.index, 30)  # skraca etykiety osi Y, żeby nie wychodziły poza wykres
    fig_h = max(4, 0.35 * len(s) + 1)  # dobiera wysokość figury do liczby słupków (czytelność)
    fig, ax = plt.subplots(figsize=(10, fig_h))  # tworzy figurę i osie o ustalonym rozmiarze
    ax.barh(labels, s.values, color=color)  # rysuje poziome słupki (etykiety po lewej, wartości po prawej)
    ax.set_title(title)  # ustawia tytuł wykresu
    ax.set_xlabel(xlabel)  # ustawia podpis osi X
    fig.tight_layout()  # dopasowuje marginesy, żeby nic się nie ucinało
    return fig  # zwraca gotową figurę (potem kodowana do base64)

def pie_with_legend(series, top_n=6, title="", donut=True, colors=None):  # tworzy wykres kołowy z legendą (opcjonalnie donut)
    s = series.dropna().sort_values(ascending=False)  # usuwa braki i sortuje malejąco, żeby „TOP” miało sens
    if len(s) == 0:  # zabezpieczenie: brak danych do narysowania
        return None  # zwraca None, żeby wyżej nie próbować kodować pustej figury
    head = s.head(top_n).copy()  # bierze top_n kategorii (kopiujemy, bo będziemy dopisywać „Inne”)
    rest = s.iloc[top_n:].sum()  # sumuje resztę kategorii jako „Inne”
    if rest > 0:  # jeśli faktycznie istnieje „ogon”
        head.loc["Inne"] = rest  # dopisuje kategorię „Inne”, żeby wykres był czytelny
    labels = wrap_labels(shorten_labels(head.index, 22), width=18)  # skraca i zawija etykiety do legendy
    fig, ax = plt.subplots(figsize=(9, 6))  # tworzy figurę dla wykresu kołowego
    wedged, _, _ = ax.pie(head.values, autopct="%1.1f%%", startangle=90, colors=colors)  # rysuje koło z procentami
    ax.legend(wedged, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)  # dodaje legendę po prawej
    if donut:  # jeśli ma być donut (pierścień)
        ax.add_artist(plt.Circle((0, 0), 0.62, fc="white"))  # dorysowuje białe koło w środku (efekt donut)
    ax.set_title(title)  # ustawia tytuł wykresu
    fig.tight_layout()  # dopasowuje układ wykresu i legendy
    return fig  # zwraca figurę

# główna funkcja eda - zwraca tekst i liste wykresow  # EDA: buduje raport tekstowy + obrazy do HTML

def run_eda():  # uruchamia całą analizę i zwraca (tekst, lista obrazów)
    text_buffer = StringIO()  # tworzy bufor tekstowy do przechwytywania wydruków print()
    images = []  # lista par (tytuł, base64) dla wykresów do raportu HTML

    with redirect_stdout(text_buffer):  # przekierowuje printy do bufora zamiast do konsoli (żeby wkleić je do HTML)
        print("=" * 80)  # separator wizualny w raporcie tekstowym
        print("RAPORT EDA - COVID: ZAKAŻENIA PO SZCZEPIENIU")  # tytuł raportu
        print("Plik wejściowy:", INPUT_PATH.resolve())  # wypisuje pełną ścieżkę pliku wejściowego
        print("Data i godzina analizy:", ANALYSIS_TIME.strftime("%Y-%m-%d_%H-%M-%S"))  # wypisuje timestamp analizy
        print("=" * 80)  # separator kończący nagłówek

        try:  # próbuje wczytać plik w popularnym kodowaniu Windows (częste w PL)
            df = pd.read_csv(INPUT_PATH, encoding="cp1250", low_memory=False)  # wczytuje CSV, low_memory=False poprawia typowanie kosztem RAM
            used_encoding = "cp1250"  # zapisuje informację jakiego kodowania użyto
        except UnicodeDecodeError:  # jeśli cp1250 nie pasuje
            df = pd.read_csv(INPUT_PATH, encoding="utf-8", low_memory=False)  # próbuje UTF-8 jako alternatywę
            used_encoding = "utf-8"  # zapisuje informację o kodowaniu

        print("\n=== WCZYTANIE ===")  # sekcja raportu: wczytanie danych
        print("Użyte kodowanie", used_encoding)  # pokazuje, jak wczytano plik
        print("Rozmiar (wiersz, kolumny):", df.shape)  # pokazuje rozmiar danych: (wiersze, kolumny)
        print("Kolumny:", df.columns.tolist())  # wypisuje listę kolumn, żeby wiedzieć co jest dostępne
        print("\nPierwsze 5 wierszy:\n", df.head())  # szybki podgląd danych (kontrola struktury)

        if "data_rap_zakazenia" in df.columns:  # sprawdza, czy jest kolumna z datą raportu zakażenia
            df["data_rap_zakazenia"] = pd.to_datetime(df["data_rap_zakazenia"], errors="coerce")  # konwertuje na datetime; błędy -> NaT
            df["miesiac"] = df["data_rap_zakazenia"].dt.to_period("M").dt.to_timestamp()  # tworzy miesiąc (pierwszy dzień miesiąca) do agregacji

        if W in df.columns:  # sprawdza, czy istnieje kolumna „waga/liczba przypadków”
            df[W] = pd.to_numeric(df[W], errors="coerce").fillna(0)  # konwertuje na liczbę; błędy -> NaN -> 0, żeby sumy działały
            total_cases = float(df[W].sum())  # liczy łączną liczbę zakażeń jako sumę wag (ważne: to nie liczba wierszy)
        else:  # gdy brakuje kolumny wagi
            total_cases = float(len(df))  # używa liczby wierszy jako zastępczej miary (mniej precyzyjne)

        if "producent" in df.columns:  # jeśli jest kolumna producent szczepionki
            df["producent2"] = df["producent"].fillna("brak informacji")  # uzupełnia braki do jednej kategorii, by nie gubić ich w agregacjach
        if "dawka_ost" in df.columns:  # jeśli jest kolumna dawka
            df["dawka2"] = df["dawka_ost"].fillna("brak informacji")  # uzupełnia braki do czytelnej kategorii

        woj_map = {  # mapowanie kodów TERYT województw na nazwy czytelne w raporcie
            2: "dolnośląskie",  # kod 2 -> dolnośląskie
            4: "kujawsko-pomorskie",  # kod 4 -> kujawsko-pomorskie
            6: "lubelskie",  # kod 6 -> lubelskie
            8: "lubuskie",  # kod 8 -> lubuskie
            10: "łódzkie",  # kod 10 -> łódzkie
            12: "małopolskie",  # kod 12 -> małopolskie
            14: "mazowieckie",  # kod 14 -> mazowieckie
            16: "opolskie",  # kod 16 -> opolskie
            18: "podkarpackie",  # kod 18 -> podkarpackie
            20: "podlaskie",  # kod 20 -> podlaskie
            22: "pomorskie",  # kod 22 -> pomorskie
            24: "śląskie",  # kod 24 -> śląskie
            26: "świętokrzyskie",  # kod 26 -> świętokrzyskie
            28: "warmińsko-mazurskie",  # kod 28 -> warmińsko-mazurskie
            30: "wielkopolskie",  # kod 30 -> wielkopolskie
            32: "zachodniopomorskie"  # kod 32 -> zachodniopomorskie
        }  # koniec słownika mapowania
        if "teryt_woj" in df.columns:  # sprawdza, czy jest kolumna z kodem województwa
            df["woj"] = df["teryt_woj"].apply(lambda x: int(x) if pd.notna(x) else np.nan)  # normalizuje kod do int, a brak zostawia jako NaN
            df["woj_nazwa"] = df["woj"].map(woj_map)  # mapuje kod na nazwę województwa

        print("\n=== PODSTAWOWE INFO ===")  # sekcja raportu: podstawowe metryki
        print("Łączna liczba zakażeń (suma wag):", int(total_cases))  # wypisuje sumę zakażeń policzoną jako suma W
        if "data_rap_zakazenia" in df.columns:  # jeśli mamy datę zakażenia
            print("Zakres dat:", df["data_rap_zakazenia"].min(), " -> ", df["data_rap_zakazenia"].max())  # wypisuje min i max dat

        missing = df.isna().sum()  # liczy liczbę braków w każdej kolumnie
        missing_pct = (missing / len(df) * 100).round(2)  # liczy % braków względem liczby wierszy, zaokrągla do 2 miejsc
        missing_table = (pd.DataFrame({"braki": missing, "braki_%": missing_pct}).sort_values("braki", ascending=False))  # tworzy tabelę braków i sortuje
        print("\n=== BRAKI DANYCH (TOP 20) ===")  # nagłówek sekcji braków
        print(missing_table.head(20))  # wypisuje 20 kolumn z największą liczbą braków

        top = missing_table.head(20).iloc[::-1]  # bierze TOP20 i odwraca kolejność do barh (żeby największe były na górze)
        fig, ax = plt.subplots(figsize=(10, 6))  # tworzy figurę do wykresu braków
        ax.barh(shorten_labels(top.index, 32), top["braki_%"].values, color=PASTEL[0])  # rysuje % braków jako słupki poziome
        ax.set_title("TOP 20 kolumn wg % braków")  # tytuł wykresu
        ax.set_xlabel("% braków")  # podpis osi X
        fig.tight_layout()  # dopasowuje marginesy
        images.append(("TOP 20 kolumn wg % braków", fig_to_base64(fig)))  # dodaje wykres do listy obrazów jako base64

        if "data_rap_zakazenia" in df.columns and W in df.columns:  # warunek: mamy datę i wagę, więc da się policzyć trend dzienny
            daily = df.groupby("data_rap_zakazenia")[W].sum().sort_index()  # sumuje zakażenia dziennie i sortuje po dacie
            roll7 = daily.rolling(7).mean()  # liczy 7-dniową średnią kroczącą, by wygładzić szum raportowania

            print("\n=== SZCZYT DZIENNY ===")  # nagłówek sekcji szczytu dziennego
            print("Data szczytu:", daily.idxmax(), "| Liczba:", int(daily.max()))  # wypisuje dzień z maksymalną liczbą zakażeń i wartość

            fig, ax = plt.subplots(figsize=(12, 5))  # tworzy figurę trendu dziennego
            ax.plot(daily.index, daily.values, linewidth=1.0, color=BRIGHT[3])  # rysuje serię dzienną (bardziej „poszarpaną”)
            ax.plot(roll7.index, roll7.values, linewidth=2.5, color=BRIGHT[0])  # rysuje średnią 7-dniową (wygładzoną)
            format_date_axis(ax)  # formatuje oś dat, żeby była czytelna
            ax.set_title("Zakażenia dzienne + 7-dniowa średnia")  # tytuł wykresu
            ax.set_xlabel("Data")  # podpis osi X
            ax.set_ylabel("Liczba zakażeń")  # podpis osi Y
            fig.tight_layout()  # dopasowanie układu
            images.append(("Zakażenia dzienne + 7-dniowa średnia", fig_to_base64(fig)))  # dodaje wykres do raportu

        if "miesiac" in df.columns and W in df.columns:  # warunek: mamy miesiąc i wagę, więc liczymy sumy miesięczne
            monthly = df.groupby("miesiac")[W].sum().sort_index()  # sumuje zakażenia per miesiąc i sortuje chronologicznie
            print("\n=== MIESIĘCZNE SUMY (TOP 10) ===")  # nagłówek sekcji miesięcznej
            print(monthly.sort_values(ascending=False).head(10))  # wypisuje TOP 10 miesięcy o największej liczbie zakażeń

            fig, ax = plt.subplots(figsize=(12, 5))  # tworzy figurę trendu miesięcznego
            ax.plot(monthly.index, monthly.values, linewidth=3.0, color=PASTEL[1])  # rysuje miesięczne sumy (zwykle mniej szumu)
            format_date_axis(ax)  # formatuje oś czasu
            ax.set_title("Zakażenia miesięczne")  # tytuł wykresu
            ax.set_ylabel("Liczba zakażeń")  # podpis osi Y
            ax.set_xlabel("Miesiąc")  # podpis osi X
            fig.tight_layout()  # dopasowanie układu
            images.append(("Zakażenia miesięczne", fig_to_base64(fig)))  # dodaje wykres do raportu

        if "plec" in df.columns and W in df.columns:  # warunek: mamy płeć i wagę, więc liczymy rozkład płci
            sex = df.groupby("plec")[W].sum().sort_values(ascending=False)  # sumuje zakażenia wg płci
            print("\n=== PŁEĆ (udział % ) ===")  # nagłówek sekcji płci
            print((sex / total_cases * 100).round(2).rename("procent_%"))  # pokazuje udział procentowy każdej płci

            fig = pie_with_legend(sex, top_n=5, title="Zakażenia wg płci (donut)", donut=True, colors=BRIGHT[:7])  # tworzy donut płci
            if fig is not None:  # zabezpiecza przed pustymi danymi
                images.append(("Zakażenia wg płci (donut)", fig_to_base64(fig)))  # dodaje wykres do raportu

            if "kat_wiek" in df.columns and W in df.columns:  # warunek: mamy kategorię wieku i wagę
                age_cat = df.groupby("kat_wiek")[W].sum().sort_values(ascending=False)  # sumuje zakażenia wg kategorii wieku
                print("\n=== KATEGORIA WIEKU (TOP 10) ===")  # nagłówek sekcji kategorii wieku
                print(age_cat.head(10).astype(int))  # wypisuje TOP10 kategorii wieku jako liczby całkowite

                fig = barh_top(age_cat, n=15, title="Zakażenia wg kategorii wieku (TOP 15)", xlabel="Liczba zakażeń", color=PASTEL[2])  # wykres TOP15
                images.append(("Zakażenia wg kategorii wieku (TOP 15)", fig_to_base64(fig)))  # dodaje wykres do raportu

            if "wiek" in df.columns:  # warunek: mamy liczbowy wiek (lub do konwersji)
                age_num = pd.to_numeric(df["wiek"], errors="coerce").dropna()  # konwertuje wiek na liczby; błędy -> NaN; dropna usuwa braki
                print("\n=== WIEK (statystyki) ===")  # nagłówek sekcji statystyk wieku
                if len(age_num) > 0:  # sprawdza, czy po czyszczeniu zostały jakiekolwiek wartości wieku
                    print(age_num.describe())  # describe liczy statystyki tylko dla poprawnych liczb (stąd count = liczba nie-NaN po to_numeric+dropna)
                    print("Ile <0 lub >110:", int(((age_num < 0) | (age_num > 110)).sum()))  # liczy podejrzane wartości wieku poza zakresem (kontrola jakości)
                    # Wyniki są prawidłowe, bo: 1) count ~3.635978e+06 to liczba rekordów z poprawnym wiekiem po konwersji, 2) zapis e+06/e+02 to standardowy zapis naukowy dla dużych liczb w pandas, 3) min=0 jest możliwe (np. wiek 0 dla niemowląt lub „0” jako wartość w danych), 4) percentyle 25/50/75 (29/43/60) to kwartyle policzone z rozkładu wieku w tej próbce, 5) max=117 oraz tylko 5 wartości <0 lub >110 pokazuje, że prawie wszystkie wartości są w rozsądnym zakresie, a pojedyncze odstające zostały policzone przez filtr.

                    fig, ax = plt.subplots(figsize=(10, 5))  # tworzy figurę histogramu wieku
                    ax.hist(age_num, bins=40, edgecolor="black", alpha=0.9, color=PASTEL[3])  # rysuje histogram rozkładu wieku
                    ax.set_title("Histogram wieku")  # tytuł histogramu
                    ax.set_xlabel("Wiek")  # podpis osi X
                    ax.set_ylabel("Liczba rekordów")  # podpis osi Y
                    fig.tight_layout()  # dopasowanie układu
                    images.append(("Histogram wieku", fig_to_base64(fig)))  # dodaje histogram do raportu

                    fig, ax = plt.subplots(figsize=(6, 4))  # tworzy figurę boxplota wieku
                    ax.boxplot(age_num.values, vert=True, patch_artist=True)  # rysuje boxplot (medianę, kwartyle i outliery)
                    ax.set_title("Boxplot wieku")  # tytuł boxplota
                    ax.set_ylabel("Wiek")  # podpis osi Y
                    fig.tight_layout()  # dopasowanie układu
                    images.append(("Boxplot wieku", fig_to_base64(fig)))  # dodaje boxplot do raportu
                else:  # przypadek, gdy po czyszczeniu nie zostało nic w wieku
                    print("[POMINIĘTO] brak danych w 'wiek'")  # informuje w raporcie tekstowym, że sekcja wieku została pominięta

            if "producent2" in df.columns and W in df.columns:  # warunek: mamy producenta po uzupełnieniu i wagę
                prod = df.groupby("producent2")[W].sum().sort_values(ascending=False)  # sumuje zakażenia wg producenta
                prod_known = prod.drop(index=["brak informacji"], errors="ignore")  # usuwa kategorię „brak informacji”, by zobaczyć znanych producentów
                print("\n=== PRODUCENT (ZNANI) TOP 10 ===")  # nagłówek sekcji producentów
                print(prod_known.head(10))  # wypisuje TOP 10 producentów (znanych)

                fig = barh_top(prod_known, n=10, title="Zakażenia wg producenta (bez brak informacji)", xlabel="Liczba zakażeń", color=PASTEL[2])  # wykres TOP10
                images.append(("Zakażenia wg producenta (bez brak informacji)", fig_to_base64(fig)))  # dodaje wykres do raportu

            if "dawka2" in df.columns and W in df.columns:  # warunek: mamy dawkę po uzupełnieniu i wagę
                dose = df.groupby("dawka2")[W].sum().sort_values(ascending=False)  # sumuje zakażenia wg dawki
                print("\n=== DAWKA (udział %) ===")  # nagłówek sekcji dawki
                print((dose / total_cases * 100).round(2).rename("procent_%"))  # pokazuje udział procentowy zakażeń wg dawki

                fig = pie_with_legend(dose, top_n=6, title="Zakażenia wg dawki (donut)", donut=True, colors=BRIGHT[:7])  # wykres donut dla dawek
                if fig is not None:  # zabezpieczenie przed pustymi danymi
                    images.append(("Zakażenia wg dawki (donut)", fig_to_base64(fig)))  # dodaje wykres do raportu

            if "woj_nazwa" in df.columns and W in df.columns:  # warunek: mamy nazwę województwa i wagę
                woj = df.groupby("woj_nazwa")[W].sum().sort_values(ascending=False)  # sumuje zakażenia wg województwa
                print("\n=== WOJEWÓDZTWA TOP 10 ===")  # nagłówek sekcji województw
                print(woj.head(10))  # wypisuje TOP10 województw

                fig = barh_top(woj, n=10, title="Top 10 województw wg zakażeń", xlabel="Liczba zakażeń", color=PASTEL[2])  # wykres TOP10 województw
                images.append(("Top 10 województw", fig_to_base64(fig)))  # dodaje wykres do raportu

            if "woj_nazwa" in df.columns and "miesiac" in df.columns and W in df.columns:  # warunek: mamy województwo, miesiąc i wagę
                woj_total = df.groupby("woj_nazwa")[W].sum().sort_values(ascending=False)  # liczy łączną liczbę zakażeń w województwach
                top5 = woj_total.head(5).index.tolist()  # wybiera 5 województw z największą liczbą zakażeń

                pivot = (df[df["woj_nazwa"].isin(top5)].groupby(["miesiac", "woj_nazwa"])[W].sum().unstack(fill_value=0).sort_index())  # pivot: miesiąc x województwo
                fig, ax = plt.subplots(figsize=(12, 6))  # tworzy figurę trendu dla top5 województw
                for i, col in enumerate(pivot.columns):  # iteruje po województwach (kolumnach pivota)
                    ax.plot(pivot.index, pivot[col].values, linewidth=2.5, label=str(col), color=BRIGHT[i % len(BRIGHT)])  # rysuje linię trendu dla województwa
                format_date_axis(ax)  # formatuje oś czasu
                ax.legend(loc="upper left", frameon=False)  # dodaje legendę z nazwami województw
                ax.set_title("Top 5 województw - trend miesięczny")  # tytuł wykresu
                ax.set_xlabel("Miesiąc")  # podpis osi X
                ax.set_ylabel("Liczba zakażeń")  # podpis osi Y
                fig.tight_layout()  # dopasowuje układ
                images.append(("Top 5 województw - trend miesięczny", fig_to_base64(fig)))  # dodaje wykres do raportu

            print("\n" + "=" * 80)  # separator przed podsumowaniem
            print("PODSUMOWANIE KLUCZOWYCH ELEMENTÓW")  # tytuł podsumowania
            print("1) Waga danych:", W, "-> wyniki liczymy jako sumę tej kolumny nie liczbę wierszy")  # wyjaśnia interpretację kolumny W
            print("2) Największe ograniczenie: braki danych w niektórych kolumnach (sprawdź tabelę braków)")  # wskazuje ograniczenie jakości danych
            print("3) Trend dzienny + 7 dniowa średnia pokazuje fale i redukuje szum raporotowania")  # wyjaśnia po co średnia 7-dniowa
            print("4) Rozkłady: płeć, wiek, dawki, geografia")  # wymienia analizowane przekroje
            print("\n" + "=" * 80)  # separator kończący raport

    return text_buffer.getvalue(), images  # zwraca cały tekst raportu i listę wykresów w base64

# składanie jednego pliku html  # sekcja budowy HTML w jednym pliku

text, images = run_eda()  # uruchamia EDA i pobiera tekst oraz obrazy do raportu

html_parts = []  # lista fragmentów HTML, które potem zostaną sklejone
html_parts.append("<html><head><meta charset='utf-8'><title>EDA report</title></head><body>")  # start dokumentu HTML + kodowanie UTF-8
html_parts.append("<h1>EDA report</h1>")  # nagłówek dokumentu
html_parts.append(f"<p><b>Data i godzina:<b> {ANALYSIS_TIME.strftime('%Y-%m-%d %H:%M:%S')}</b></p>")  # wstawia czas analizy do HTML
html_parts.append(f"<p><b>Plik: </b> {str(INPUT_PATH.resolve())}</p>")  # wstawia ścieżkę pliku wejściowego do HTML
html_parts.append("<hr>")  # pozioma linia oddzielająca sekcje
html_parts.append("<h2>Wyniki tekstowe</h2>")  # nagłówek sekcji tekstowej
html_parts.append("<pre style='white-space: pre-wrap;font-family:Consolas, monospace;'>")  # blok pre z zawijaniem i czcionką monospace
html_parts.append(text)  # wstawia przechwycony tekst z printów do HTML
html_parts.append("</pre>")  # zamyka blok pre
html_parts.append("<hr>")  # separator między sekcjami
html_parts.append("<h2>Wykresy</h2>")  # nagłówek sekcji wykresów

for title, b64 in images:  # iteruje po wszystkich zapisanych wykresach (tytuł, base64)
    html_parts.append(f"<h3>{title}</h3>")  # dodaje tytuł wykresu jako nagłówek
    html_parts.append(  # dodaje tag img z osadzonym obrazem base64
        f"<img src='data:image/png;base64,{b64}' "  # osadza obraz jako data URI, bez osobnych plików PNG
        f"style='max-width:100%; height:auto; border:1px solid #ddd;'>"  # ustawia responsywny rozmiar i delikatną ramkę
    )  # kończy dodawanie img
html_parts.append(f"<br><br>")  # dodaje odstęp na końcu sekcji wykresów

html_parts.append("</body></html>")  # zamyka body i html

REPORT_HTML.write_text("\n".join(html_parts), encoding="utf-8")  # zapisuje złączony HTML do pliku w UTF-8

print(f"Gotowe - plik z ANALIZĄ EDA COVID zapisany: {REPORT_HTML}")  # informuje użytkownika gdzie zapisano gotowy raport
