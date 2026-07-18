print("=" * 70)
print("  PYTHON 1. DERS — String İşlemleri, Sözlükler, Sınıflar")
print("=" * 70)

print(f"\n[STRING İŞLEMLERİ]")
metin = "  Python Programlama Dili  "
print(f"  Orijinal    : '{metin}'")
print(f"  upper()     : '{metin.upper()}'")
print(f"  lower()     : '{metin.lower()}'")
print(f"  strip()     : '{metin.strip()}'")
print(f"  split()     : {metin.strip().split()}")
print(f"  replace()   : '{metin.strip().replace('Python', 'Java')}'")
print(f"  find()      : {metin.find('Dili')}")
print(f"  count()     : {metin.count('a')}")
print(f"  startswith  : {metin.strip().startswith('Python')}")
print(f"  endswith    : {metin.strip().endswith('Dili')}")

print(f"\n  f-string formatlama:")
isim, yas, boy = "Ayşe", 28, 1.65
print(f"  {isim=}, {yas=}, {boy=:.2f}")
print(f"  {255:08b} (binary)")
print(f"  {3.14159:.3f}")

print(f"\n[SLİCE DETAYLI]")
kelime = "Python"
for i in range(len(kelime)):
    print(f"  [{i}] → {kelime[i]}")

print(f"\n[SÖZLÜKLER (DICTIONARY])")
ogrenci = {
    "isim": "Mehmet",
    "yas": 22,
    "bolum": "Bilgisayar Müh.",
    "notlar": [85, 92, 78, 95]
}
print(f"  Öğrenci     : {ogrenci}")
print(f"  isim        : {ogrenci['isim']}")
print(f"  get()       : {ogrenci.get('yas', 'Yok')}")
print(f"  keys()      : {list(ogrenci.keys())}")
print(f"  values()    : {list(ogrenci.values())}")
print(f"  items()     : {list(ogrenci.items())}")

ogrenci["sehir"] = "Ankara"
ogrenci["notlar"].append(88)
print(f"  Ekleme sonrası: {ogrenci}")

print(f"\n  Dict Comprehension:")
kareler = {x: x**2 for x in range(1, 6)}
print(f"  {kareler}")

print(f"\n[SET]")
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7, 8}
print(f"  A = {a}")
print(f"  B = {b}")
print(f"  Birleşim  : {a | b}")
print(f"  Kesişim   : {a & b}")
print(f"  Fark      : {a - b}")

print(f"\n[SINIFLARA GİRİŞ]")
class Hayvan:
    tur_sayisi = 0

    def __init__(self, isim, tur):
        self.isim = isim
        self.tur = tur
        self.enerji = 100
        Hayvan.tur_sayisi += 1

    def ses_cikar(self):
        return f"{self.isim} ses çıkarıyor"

    def besle(self, miktar):
        self.enerji = min(100, self.enerji + miktar)
        return f"{self.isim} beslendi. Enerji: {self.enerji}"

kedi = Hayvan("Boncuk", "Kedi")
köpek = Hayvan("Karabaş", "Köpek")

print(f"  {kedi.ses_cikar()}")
print(f"  {köpek.ses_cikar()}")
print(f"  {kedi.besle(30)}")
print(f"  Toplam Hayvan: {Hayvan.tur_sayisi}")

print(f"\n[KALITIM]")
class Kedi(Hayvan):
    def __init__(self, isim, disli_mi=False):
        super().__init__(isim, "Kedi")
        self.disli_mi = disli_mi

    def ses_cikar(self):
        return f"{self.isim} miyavlıyor"

    def tırmala(self):
        return f"{self.isim} tırmalıyor!"

meyve = Kedi("Meyve", disli_mi=True)
print(f"  {meyve.ses_cikar()}")
print(f"  {meyve.tırmala()}")
print(f"  Kedi mi? {isinstance(meyve, Kedi)}")
print(f"  Hayvan mi? {isinstance(meyve, Hayvan)}")

print(f"\n[KAPSÜLLEME]")
class BankaHesap:
    def __init__(self, sahip, bakiye=0):
        self.sahip = sahip
        self.__bakiye = bakiye

    @property
    def bakiye(self):
        return self.__bakiye

    def yatir(self, miktar):
        if miktar > 0:
            self.__bakiye += miktar
            return f"+{miktar} TL yatırıldı. Bakiye: {self.__bakiye} TL"
        return "Geçersiz miktar!"

    def cek(self, miktar):
        if 0 < miktar <= self.__bakiye:
            self.__bakiye -= miktar
            return f"-{miktar} TL çekildi. Bakiye: {self.__bakiye} TL"
        return "Yetersiz bakiye!"

hesap = BankaHesap("Ali", 1000)
print(f"  {hesap.yatir(500)}")
print(f"  {hesap.cek(200)}")
print(f"  Bakiye (property): {hesap.bakiye} TL")

print(f"\n[DUNDER METODLAR]")
class Vekil:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vekil({self.x}, {self.y})"

    def __add__(self, other):
        return Vekil(self.x + other.x, self.y + other.y)

    def __len__(self):
        return int((self.x**2 + self.y**2)**0.5)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

v1 = Vekil(3, 4)
v2 = Vekil(1, 2)
print(f"  v1 = {v1}")
print(f"  v2 = {v2}")
print(f"  v1 + v2 = {v1 + v2}")
print(f"  |v1| = {len(v1)}")
print(f"  v1 == v2: {v1 == v2}")

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — String, Sözlük, Sınıf, Kalıtım, Kapsülleme.")
print(f"{'='*70}")
