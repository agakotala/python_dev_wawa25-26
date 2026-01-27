from __future__ import annotations  # pozwala używać adnotacji typów jako napisów i wygodnie odnosić się do klas jeszcze przed ich definicją
from dataclasses import dataclass  # importuje dekorator dataclass do prostego tworzenia klas przechowujących dane
from datetime import datetime  # importuje typ datetime do pracy z datą i godziną (np. szczyt, formatowanie)

class Pojazd:  # klasa bazowa dla różnych typów pojazdów, zawiera wspólne zasady liczenia ceny
    def __init__(self, nazwa: str, oplata_startowa: float, za_km: float, min_oplata: float) -> None:  # konstruktor przyjmujący parametry cennika
        self.nazwa = nazwa  # zapisuje nazwę pojazdu (np. Auto, Hulajnoga) do opisu i raportu
        self._oplata_startowa = float(oplata_startowa)  # zapisuje opłatę startową jako float do obliczeń
        self._za_km = float(za_km)  # zapisuje stawkę za kilometr jako float do obliczeń
        self._min_oplata = float(min_oplata)  # zapisuje minimalną opłatę jako float, aby wymusić dolny limit ceny

    def oszacuj_koszt_bazowy(self, km: float) -> float:  # metoda licząca koszt bazowy przejazdu bez szczytu i bez rabatu
        if km < 0:  # waliduje dystans, bo ujemny dystans nie ma sensu
            raise ValueError('Dystans nie może być ujemny')  # zgłasza błąd, aby wymusić poprawne dane wejściowe
        koszt = self._oplata_startowa + km * self._za_km  # liczy koszt wg wzoru: start + km * stawka
        return max(koszt, self._min_oplata)  # zwraca koszt, ale nie mniejszy niż minimalna opłata

    def doplata_szczyt(self, dt: datetime) -> float:  # metoda zwracająca mnożnik ceny zależny od godzin szczytu
        godzina = dt.hour  # pobiera godzinę z datetime, żeby sprawdzić czy jest szczyt
        w_szczycie = (7 <= godzina < 10) or (16 <= godzina < 19)  # ustala, czy czas jest w porannym lub popołudniowym szczycie
        return 1.20 if w_szczycie else 1.00  # zwraca mnożnik 1.20 w szczycie, w przeciwnym razie 1.00

    def wylicz_cene(self, km: float, dt: datetime, rabat: float = 0.0) -> float:  # metoda licząca końcową cenę z uwzględnieniem szczytu i rabatu
        if not (0.0 <= rabat <= 1.0):  # waliduje rabat jako ułamek (0–1), żeby nie podać np. 20 albo -0.5
            raise ValueError("Rabat musi być w zakresie 0.0 - 1.0")  # zgłasza błąd przy niepoprawnym rabacie
        bazowa = self.oszacuj_koszt_bazowy(km)  # wylicza bazowy koszt przejazdu na podstawie dystansu i cennika
        mnoznik = self.doplata_szczyt(dt)  # pobiera mnożnik za godzinę szczytu na podstawie datetime
        cena_po_szczycie = bazowa * mnoznik  # uwzględnia dopłatę za szczyt przez przemnożenie bazowej ceny
        cena_po_rabacie = cena_po_szczycie * (1.0 - rabat)  # stosuje rabat jako procentową zniżkę od ceny po szczycie
        return round(cena_po_rabacie, 2)  # zaokrągla wynik do 2 miejsc (waluta) i zwraca cenę

class Auto(Pojazd):  # klasa Auto dziedziczy po Pojazd i ustawia własny cennik
    def __init__(self) -> None:  # konstruktor auta bez dodatkowych parametrów
        super().__init__(nazwa="Auto", oplata_startowa=8.0, za_km=3.2, min_oplata=10.0)  # inicjalizuje bazę z cennikiem auta

class Hulajnoga(Pojazd):  # klasa Hulajnoga dziedziczy po Pojazd i ustawia własny cennik
    def __init__(self) -> None:  # konstruktor hulajnogi bez dodatkowych parametrów
        super().__init__(nazwa="Hulajnoga", oplata_startowa=2.0, za_km=1.6, min_oplata=4.0)  # inicjalizuje bazę z cennikiem hulajnogi

class AutoPremium(Pojazd):  # klasa AutoPremium dziedziczy po Pojazd, ma wyższe stawki i dodatkową dopłatę serwisową
    def __init__(self) -> None:  # konstruktor auta premium bez dodatkowych parametrów
        super().__init__(nazwa="AutoPremium", oplata_startowa=15.0, za_km=5.0, min_oplata=18.0)  # inicjalizuje bazę z cennikiem premium

    def wylicz_cene(self, km: float, dt: datetime, rabat: float = 0.0) -> float:  # nadpisuje liczenie ceny, aby dodać dopłatę serwisową
        cena = super().wylicz_cene(km, dt, rabat=rabat)  # liczy cenę standardowym mechanizmem z klasy bazowej
        doplata_serwisowa = 1.10  # definiuje mnożnik dopłaty serwisowej (np. premium fee)
        return round(cena * doplata_serwisowa, 2)  # dodaje dopłatę serwisową przez mnożnik i zaokrągla do 2 miejsc

@dataclass  # oznacza, że klasa ma generowane automatycznie __init__, __repr__ itp. na podstawie pól
class Przejazd:  # klasa danych opisująca pojedynczy przejazd: kto, czym, ile km i jaki rabat
    pasazer: str  # imię/nazwa pasażera do raportu
    km: float  # dystans przejazdu w kilometrach do wyliczenia kosztu
    pojazd: Pojazd  # obiekt pojazdu (Auto/Hulajnoga/AutoPremium) określający stawki i zasady ceny
    rabat: float = 0.0  # rabat domyślnie 0%, podawany jako ułamek (np. 0.10)
    czas: datetime | None = None  # czas przejazdu; jeśli None, przyjmowany będzie bieżący czas

    def cena(self) -> float:  # metoda wyliczająca cenę przejazdu na podstawie danych obiektu
        df = self.czas or datetime.now()  # wybiera ustawiony czas przejazdu albo bierze aktualny czas, gdy brak
        return self.pojazd.wylicz_cene(self.km, df, rabat=self.rabat)  # wylicza cenę, delegując do pojazdu i przekazując rabat

    def opis(self) -> float:  # buduje opis tekstowy przejazdu do raportu (pasażer, pojazd, km, rabat, godzina)
        dt = self.czas or datetime.now()  # wybiera czas przejazdu albo bieżący czas, żeby mieć spójną godzinę w raporcie
        godzina = dt.strftime("%H:%M")  # formatuje godzinę jako HH:MM dla czytelności
        return f"{self.pasazer:10} | {self.pojazd.nazwa:11} | {self.km:5.1f} km | {self.rabat*100:5.0f}% | {godzina}"  # zwraca uformatowaną linię opisu

def raport_przejazdow(przejazdy: list[Przejazd]) -> tuple[str, list[float], float]:  # tworzy raport tekstowy, listę cen i sumę dla listy przejazdów
    naglowek = "PASAŻER    |  POJAZD     |  KM      | RABAT  |GODZINA|  CENA  "  # definiuje stały nagłówek tabeli raportu
    kreska = "-" * len(naglowek)  # tworzy linię separującą o tej samej długości co nagłówek
    linie = [naglowek, kreska]  # inicjalizuje listę linii raportu nagłówkiem i separatorem
    ceny: list[float] = []  # tworzy pustą listę cen, aby zebrać ceny z każdego przejazdu
    for p in przejazdy:  # iteruje po wszystkich przejazdach, aby policzyć cenę i dodać linię do raportu
        cena = p.cena()  # wylicza cenę dla danego przejazdu metodą obiektu
        ceny.append(cena)  # dopisuje cenę do listy cen, żeby później policzyć sumę
        linie.append(f"{p.opis():52} | {cena:6.2f}")  # dodaje do raportu linię opisu przejazdu oraz cenę w formacie walutowym

    suma = round(sum(ceny), 2)  # liczy sumę wszystkich cen i zaokrągla do 2 miejsc
    linie.append(kreska)  # dodaje separator na końcu tabeli
    linie.append(f"SUMA: {suma:.2f} PLN")  # dodaje wiersz podsumowania z sumą w złotówkach
    tekst = "\n".join(linie)  # skleja wszystkie linie raportu w jeden tekst z nowymi liniami
    print(tekst)  # drukuje raport na ekran, aby użytkownik od razu go zobaczył
    return tekst, ceny, suma  # zwraca raport jako tekst, listę cen i sumę (np. do testów lub dalszego użycia)

def demo_przejazdy() -> tuple[str, list[float], float]:  # funkcja demonstracyjna przygotowująca przykładowe przejazdy i generująca raport
    rano_szczyt = datetime(2026, 1, 22, 8, 15)  # definiuje czas w porannym szczycie (wpływa na mnożnik 1.20)
    poza_szczytem = datetime(2026, 1, 22, 11, 30)  # definiuje czas poza szczytem (mnożnik 1.00)
    przejazdy = [  # tworzy listę przykładowych przejazdów do raportu
        Przejazd(pasazer="Ala", km=2.0, pojazd=AutoPremium(), rabat=0.0, czas=rano_szczyt),  # przejazd premium w szczycie bez rabatu
        Przejazd(pasazer="Ola", km=1.2, pojazd=Hulajnoga(), rabat=0.10, czas=poza_szczytem),  # przejazd hulajnogą poza szczytem z 10% rabatem
        Przejazd(pasazer="Jan", km=5.0, pojazd=Auto(), rabat=0.0, czas=rano_szczyt),  # przejazd autem w szczycie bez rabatu
    ]  # kończy listę przejazdów
    return raport_przejazdow(przejazdy)  # generuje raport dla listy przejazdów i zwraca jego wyniki

if __name__ == "__main__":  # uruchamia poniższy kod tylko wtedy, gdy plik jest odpalony bezpośrednio
    demo_przejazdy()  # wywołuje demonstrację przejazdów, aby pokazać działanie programu
