print("=" * 70)
print("  PYTHON CLASS SORULARI — İleri Seviye Sınıf Konuları")
print("=" * 70)

print(f"\n[1] CLASS DEĞİŞKENLERİ vs INSTANCE DEĞİŞKENLERİ]")
class Calisan:
    departman = "Genel"
    _sayac = 0

    def __init__(self, isim, maas):
        self.isim = isim
        self.maas = maas
        Calisan._sayac += 1
        self._id = Calisan._sayac

c1 = Calisan("Ali", 15000)
c2 = Calisan("Veli", 18000)
print(f"  {c1.isim} (ID:{c1._id}) — Departman: {c1.departman}")
print(f"  {c2.isim} (ID:{c2._id}) — Departman: {c2.departman}")
print(f"  Toplam çalışan: {Calisan._sayac}")

Calisan.departman = "Mühendislik"
print(f"  Departman değişikliği sonrası: {c1.departman}, {c2.departman}")

print(f"\n[@property — Getter/Setter]")
class Hesap:
    def __init__(self, sahip, bakiye=0):
        self.sahip = sahip
        self._bakiye = bakiye

    @property
    def bakiye(self):
        return self._bakiye

    @bakiye.setter
    def bakiye(self, deger):
        if deger < 0:
            raise ValueError("Bakiye negatif olamaz!")
        self._bakiye = deger

    @property
    def durum(self):
        return "Zengin" if self._bakiye > 100000 else "Normal" if self._bakiye > 0 else "Sifir"

hesap = Hesap("Ayşe", 50000)
print(f"  {hesap.sahip}: {hesap.bakiye} TL ({hesap.durum})")
hesap.bakiye = 75000
print(f"  Güncelle: {hesap.bakiye} TL ({hesap.durum})")

print(f"\n[@classmethod VE @staticmethod]")
class Matematik:
    PI = 3.14159

    @classmethod
    def daire_alan(cls, yari_cap):
        return cls.PI * yari_cap ** 2

    @staticmethod
    def mutlak_deger(x):
        return x if x >= 0 else -x

    @classmethod
    def bilgi(cls):
        return f"Matematik sınıfı — PI = {cls.PI}"

print(f"  Daire alanı (r=5): {Matematik.daire_alan(5):.2f}")
print(f"  |−7| = {Matematik.mutlak_deger(-7)}")
print(f"  {Matematik.bilgi()}")

print(f"\n[ITERATOR CLASS]")
class SayiUreteci:
    def __init__(self, baslangic, bitis, adim=1):
        self.baslangic = baslangic
        self.bitis = bitis
        self.adim = adim
        self.mevcut = baslangic

    def __iter__(self):
        return self

    def __next__(self):
        if self.mevcut >= self.bitis:
            raise StopIteration
        deger = self.mevcut
        self.mevcut += self.adim
        return deger

print(f"  SayiUreteci(1, 6):", list(SayiUreteci(1, 6)))
print(f"  SayiUreteci(0, 10, 2):", list(SayiUreteci(0, 10, 2)))

print(f"\n[MIXIN PATTERN]")
class LogMixin:
    def log(self, mesaj):
        print(f"  [LOG] {self.__class__.__name__}: {mesaj}")

class JSONMixin:
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

class Kullanici(LogMixin, JSONMixin):
    def __init__(self, isim, yas):
        self.isim = isim
        self.yas = yas
        self._gizli = "sır"

k = Kullanici("Ali", 25)
k.log("Kullanici oluşturuldu")
print(f"  to_dict: {k.to_dict()}")

print(f"\n[DUNDER METODLAR]")
class Vekil2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vekil2D({self.x}, {self.y})"

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __add__(self, other):
        return Vekil2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vekil2D(self.x - other.x, self.y - other.y)

    def __mul__(self, skaler):
        return Vekil2D(self.x * skaler, self.y * skaler)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return (self.x**2 + self.y**2) < (other.x**2 + other.y**2)

    def __len__(self):
        return int((self.x**2 + self.y**2)**0.5)

    def __contains__(self, deger):
        return deger in [self.x, self.y]

v1 = Vekil2D(3, 4)
v2 = Vekil2D(1, 2)
print(f"  v1 = {repr(v1)}")
print(f"  v1 + v2 = {v1 + v2}")
print(f"  v1 - v2 = {v1 - v2}")
print(f"  v1 * 3 = {v1 * 3}")
print(f"  |v1| = {len(v1)}")
print(f"  v1 == v2: {v1 == v2}")
print(f"  v1 < v2: {v1 < v2}")
print(f"  3 in v1: {3 in v1}")

print(f"\n[CONTEXT MANAGER]")
class DosyaYonetici:
    def __init__(self, dosya_adi, mod="r"):
        self.dosya_adi = dosya_adi
        self.mod = mod
        self.dosya = None

    def __enter__(self):
        self.dosya = open(self.dosya_adi, self.mod, encoding="utf-8")
        return self.dosya

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.dosya:
            self.dosya.close()

import tempfile, os
dosya_yolu = os.path.join(tempfile.gettempdir(), "context_ornek.txt")
with DosyaYonetici(dosya_yolu, "w") as f:
    f.write("Context Manager ile yazıldı!")

with DosyaYonetici(dosya_yolu, "r") as f:
    print(f"  Okunan: {f.read()}")

os.remove(dosya_yolu)

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — İleri seviye sınıf konuları.")
print(f"{'='*70}")
