from __future__ import annotations  # pozwala używać adnotacji typów jako napisów (np. list[Wpis]) bez problemów z kolejnością definicji
from dataclasses import dataclass  # importuje dekorator i mechanizm dataclass do wygodnego tworzenia klas danych
from datetime import datetime, timezone  # importuje typ daty/czasu oraz strefę UTC do znaczników czasu

@dataclass(frozen=True)  # tworzy dataclass; frozen=True blokuje modyfikacje pól po utworzeniu obiektu (wpis historii ma być niezmienny)
class Wpis:  # definiuje strukturę pojedynczego wpisu w historii operacji
    czas: datetime  # przechowuje moment wykonania operacji jako obiekt datetime
    typ: str  # przechowuje typ operacji (np. "wplata", "wyplata") jako tekst
    kwota: float  # przechowuje kwotę operacji jako liczba zmiennoprzecinkowa
    saldo_po: float  # przechowuje saldo po operacji jako liczba zmiennoprzecinkowa

class Historia:  # klasa zarządzająca listą wpisów operacji (historią konta)
    def __init__(self) -> None:  # konstruktor historii, uruchamia się przy tworzeniu obiektu Historia
        self._wpisy: list[Wpis] = []  # tworzy pustą listę wpisów; _ sugeruje, że to pole jest „wewnętrzne”

    def dodaj(self, typ: str, kwota: float, saldo_po: float) -> Wpis:  # metoda dodająca wpis o podanym typie/kwocie/saldzie i zwracająca utworzony wpis
        czas = datetime.now(timezone.utc)  # pobiera bieżący czas w UTC, aby zapisać kiedy wykonano operację
        wpis = Wpis(czas=czas, typ=typ, kwota=kwota, saldo_po=saldo_po)  # tworzy obiekt Wpis z kompletem informacji o operacji
        self._wpisy.append(wpis)  # dodaje nowy wpis do listy historii
        return wpis  # zwraca utworzony wpis (np. do wydruku lub dalszego użycia)

    def ostatnie(self, n: int = 5) -> list[Wpis]:  # metoda zwracająca ostatnie n wpisów historii (domyślnie 5)
        return self._wpisy[-n:]  # zwraca końcówkę listy; gdy wpisów jest mniej niż n, zwróci wszystkie

class KontoBankowe:  # klasa reprezentująca konto bankowe wraz z saldem i historią
    def __init__(self, wlasciciel: str, saldo_poczatkowe: float = 0.0) -> None:  # konstruktor konta z właścicielem i opcjonalnym saldem startowym
        if saldo_poczatkowe < 0:  # waliduje niezmiennik: saldo początkowe nie może być ujemne
            raise ValueError("Saldo początkowe nie może być ujemne.")  # przerywa tworzenie konta, jeśli saldo startowe jest niepoprawne
        self.wlasciciel = wlasciciel  # zapisuje nazwę właściciela jako publiczny atrybut obiektu
        self._saldo = float(saldo_poczatkowe)  # ustawia saldo wewnętrzne jako float (ujednolica typ)
        self._historia = Historia()  # tworzy obiekt historii, aby rejestrować operacje na koncie
        self._historia.dodaj(typ="otwarcie", kwota=saldo_poczatkowe, saldo_po=self._saldo)  # zapisuje w historii operację otwarcia konta

    @property  # dekorator robi z metody „atrybut tylko do odczytu” (konto.saldo zamiast konto.saldo())
    def saldo(self) -> float:  # getter salda, zwraca aktualne saldo jako float
        return self._saldo  # zwraca wewnętrzne saldo bez pozwalania na bezpośrednią zmianę z zewnątrz

    def wplata(self, kwota: float) -> float:  # metoda wpłaty pieniędzy na konto; przyjmuje kwotę i zwraca nowe saldo
        if kwota <= 0:  # sprawdza poprawność: nie można wpłacić zera ani wartości ujemnej
            raise ValueError("Wpłata musi być dodatnia")  # zgłasza błąd, aby wymusić poprawne dane wejściowe

        self._saldo += kwota  # zwiększa saldo o wpłacaną kwotę
        wpis = self._historia.dodaj(typ="wplata", kwota=kwota, saldo_po=self._saldo)  # dodaje wpis do historii i przechowuje go do użycia w komunikacie
        print(f"[OK] wpłata {wpis.kwota:.2f} -> saldo {wpis.saldo_po:.2f}")  # wypisuje potwierdzenie wpłaty i aktualne saldo w formacie 2 miejsc po przecinku
        return self._saldo  # zwraca aktualne saldo po wpłacie

    def wyplata(self, kwota: float) -> float:  # metoda wypłaty pieniędzy z konta; przyjmuje kwotę i zwraca nowe saldo
        if kwota <= 0:  # sprawdza poprawność: nie można wypłacić zera ani wartości ujemnej
            raise ValueError("Wypłata musi być dodatnia")  # zgłasza błąd dla niepoprawnej kwoty

        if kwota > self._saldo:  # walidacja biznesowa: nie można wypłacić więcej niż dostępne środki
            raise ValueError("Brak środków na koncie")  # zgłasza błąd, gdy brakuje środków
        self._saldo -= kwota  # zmniejsza saldo o wypłacaną kwotę
        wpis = self._historia.dodaj(typ="wyplata", kwota=kwota, saldo_po=self._saldo)  # rejestruje wypłatę w historii i pobiera wpis do wydruku
        print(f"[OK] Wypłata {wpis.kwota:.2f} -> saldo {wpis.saldo_po:.2f}")  # wypisuje potwierdzenie wypłaty i nowe saldo
        return self._saldo  # zwraca aktualne saldo po wypłacie

    def wyciag(self, n: int = 5) -> str:  # buduje tekstowy wyciąg z ostatnich n operacji (domyślnie 5) i zwraca go jako string
        linie = [f"Wyciąg: {self.wlasciciel} | saldo: {self._saldo:.2f}"]  # tworzy listę linii raportu, zaczynając od nagłówka z właścicielem i saldem
        for w in self._historia.ostatnie(n):  # przechodzi po ostatnich n wpisach historii, aby dodać je do wyciągu
            czas_txt = w.czas.strftime("%Y-%m-%d %H:%M:%S UTC")  # formatuje czas wpisu do czytelnego tekstu w stałym formacie
            linie.append(f"{czas_txt} | {w.typ:8} {w.kwota:8.2f} -> {w.saldo_po:8.2f}")  # dodaje linię z danymi operacji (typ/kwota/saldo) w wyrównanym układzie
        return "\n".join(linie)  # łączy wszystkie linie w jeden tekst oddzielony znakami nowej linii

def demo_konto() -> str:  # funkcja demonstracyjna pokazująca użycie klasy KontoBankowe i zwracająca wygenerowany raport
    konto = KontoBankowe("Ala", 100.0)  # tworzy konto dla "Ala" z saldem początkowym 100.0
    konto.wplata(50.0)  # wykonuje wpłatę 50.0 na konto
    konto.wyplata(30.0)  # wykonuje wypłatę 30.0 z konta
    raport = konto.wyciag(10)  # generuje wyciąg z ostatnich 10 wpisów i zapisuje go do zmiennej
    print(raport)  # drukuje wyciąg na ekran
    return raport  # zwraca tekst wyciągu (np. do testów lub dalszego przetwarzania)

if __name__ == "__main__":  # sprawdza, czy plik został uruchomiony bezpośrednio (a nie zaimportowany jako moduł)
    demo_konto()  # uruchamia funkcję demo, aby pokazać działanie programu
