class Silnik:  # definiuje klasę Silnik opisującą parametry i zachowanie silnika
    def __init__(self, moc):  # konstruktor klasy Silnik, uruchamia się przy tworzeniu obiektu
        self.moc = moc  # atrybut obiektu: moc silnika w KM (konieczna do obliczeń i opisu)
        self.uruchomiony = False  # atrybut stanu: informuje czy silnik aktualnie pracuje (startowo jest wyłączony)

    def uruchom(self):  # metoda uruchamiająca silnik
        if self.uruchomiony:  # sprawdza, czy silnik już jest włączony, aby nie zmieniać stanu bez potrzeby
            print("Silnik jest już uruchomiony.")  # informuje użytkownika, że nie ma czego uruchamiać
        else:  # przypadek, gdy silnik jest wyłączony
            self.uruchomiony = True  # zmienia stan silnika na uruchomiony, aby inne metody mogły działać (np. przyspieszanie)
            print("Silnik uruchomiony.")  # informuje o poprawnym uruchomieniu

    def zgas(self):  # metoda gasząca silnik
        if not self.uruchomiony:  # sprawdza, czy silnik już jest zgaszony, aby nie wykonywać operacji drugi raz
            print("Silnik już jest zgaszony.")  # komunikat, gdy silnik jest już wyłączony
        else:  # przypadek, gdy silnik jest włączony
            self.uruchomiony = False  # zmienia stan silnika na zgaszony, aby auto nie mogło dalej przyspieszać
            print("Silnik zgaszony.")  # informuje o poprawnym zgaszeniu

    def zmien_moc(self, nowa_moc):  # metoda zmieniająca moc silnika, aby wpłynąć na osiągi auta
        if nowa_moc <= 0:  # walidacja: moc nie może być zerowa ani ujemna, bo to nielogiczne
            print("Moc musi być dodatnia.")  # informuje o błędnych danych
            return  # kończy metodę bez wprowadzania zmian
        self.moc = nowa_moc  # zapisuje nową wartość mocy w atrybucie obiektu
        print(f"Zmieniono moc silnika na {self.moc} KM")  # potwierdza zmianę mocy w czytelnej formie

    def __str__(self):  # metoda specjalna do zwracania tekstowego opisu obiektu Silnik (np. w print)
        stan = "uruchomiony" if self.uruchomiony else "zgaszony"  # wybiera opis stanu zależnie od flagi uruchomiony
        return f"Silnik: {self.moc} KM ({stan})"  # zwraca opis zawierający moc i stan silnika

class Samochod:  # definiuje klasę Samochod, która używa obiektu Silnik (kompozycja)
    def __init__(self, marka, silnik):  # konstruktor tworzy samochód z marką i przypisanym silnikiem
        self.marka = marka  # atrybut: marka samochodu do identyfikacji/wyświetlania
        self.silnik = silnik  # atrybut: obiekt Silnik w środku samochodu (kompozycja) do sterowania stanem i mocą
        self.predkosc = 0  # atrybut: aktualna prędkość samochodu, startowo 0

    def opis(self):  # metoda wypisująca podstawowe informacje o aucie (stan „na teraz”)
        print(f"Samochód marki {self.marka}, moc silnika: {self.silnik.moc} KM, prędkość: {self.predkosc} km/h.")  # drukuje opis auta

    def przyspiesz(self, ile):  # metoda zwiększająca prędkość o podaną wartość
        if not self.silnik.uruchomiony:  # sprawdza, czy silnik pracuje, bo bez tego nie da się przyspieszać
            print("Nie można przyspieszyć - silnik jest zgaszony.")  # informuje o przyczynie odmowy
            return  # kończy metodę bez zmian prędkości
        if ile <= 0:  # waliduje wartość przyspieszenia, żeby nie przyjmować 0 lub ujemnych wartości
            print("Wartość przyspieszenie musi być dodatnia.")  # komunikat o błędnych danych
            return  # kończy metodę bez zmian

        self.predkosc += ile  # zwiększa prędkość o podaną wartość, bo „przyspieszamy”

        max_predkosc = self.silnik.moc * 2  # wyznacza maksymalną prędkość jako prostą funkcję mocy silnika (limit osiągów)
        if self.predkosc > max_predkosc:  # sprawdza, czy przekroczono limit maksymalnej prędkości
            self.predkosc = max_predkosc  # obcina prędkość do maksymalnej, aby nie przekraczać możliwości auta
            print(f"Osiągnieto maksymalną prędkość: {self.predkosc} km/h.")  # informuje, że osiągnięto limit
        else:  # przypadek, gdy limit nie został przekroczony
            print(f"Przyspieszono. Aktualna prędkość: {self.predkosc} km/h.")  # potwierdza przyspieszenie i pokazuje nową prędkość

    def hamuj(self, ile):  # metoda zmniejszająca prędkość o podaną wartość
        if ile <= 0:  # waliduje wartość hamowania, bo musi być dodatnia
            print("Wartość hamowania musi być dodatnia.")  # informuje o błędnych danych
            return  # kończy metodę bez zmian
        self.predkosc -= ile  # zmniejsza prędkość o podaną wartość, bo „hamujemy”
        if self.predkosc < 0:  # zabezpiecza przed prędkością ujemną, która nie ma sensu w tym modelu
            self.predkosc = 0  # ustawia minimalną prędkość na 0
        print(f"Zwolniono. Aktualna prędkość: {self.predkosc} km/h.")  # informuje o nowej prędkości po hamowaniu

    def stop(self):  # metoda zatrzymująca samochód i gasząca silnik
        self.predkosc = 0  # ustawia prędkość na 0, aby auto było zatrzymane
        self.silnik.zgas()  # wywołuje metodę gaszenia silnika, aby zakończyć „pracę” auta
        print("Samochód zatrzymany.")  # informuje, że auto zostało zatrzymane

    def __str__(self):  # metoda specjalna zwracająca czytelny opis obiektu Samochod do print()
        return f"{self.marka} | {self.silnik} | prędkość: {self.predkosc} km/h"  # składa opis z marki, opisu silnika i prędkości


silnik = Silnik(150)  # tworzy obiekt Silnik o mocy 150 KM, który będzie używany przez samochód
auto = Samochod("Toyota", silnik)  # tworzy obiekt Samochod marki Toyota, przekazując mu wcześniej utworzony silnik

auto.opis()  # wypisuje opis auta (marka, moc, prędkość) na start
print(auto)  # drukuje tekstową reprezentację auta (wykorzystuje __str__)

auto.silnik.uruchom()  # uruchamia silnik, aby umożliwić przyspieszanie
auto.przyspiesz(50)  # zwiększa prędkość o 50 km/h (o ile silnik jest uruchomiony)
auto.przyspiesz(300)  # próbuje mocno przyspieszyć; może dojść do obcięcia do maksymalnej prędkości
auto.hamuj(30)  # zmniejsza prędkość o 30 km/h

auto.silnik.zmien_moc(200)  # zmienia moc silnika na 200 KM, co zwiększa możliwą prędkość maksymalną
auto.przyspiesz(80)  # przyspiesza ponownie, korzystając z nowej mocy (i potencjalnie wyższego limitu)

print(auto)  # drukuje aktualny stan auta po operacjach (marka, silnik, prędkość)
auto.stop()  # zatrzymuje auto i gasi silnik
print(auto)  # drukuje stan auta po zatrzymaniu (powinno być 0 km/h i silnik zgaszony)
