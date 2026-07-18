print("=" * 70)
print("  TEMEL PYTHON — Operatörler, Hata Yönetimi, Kapsamlı Uygulama")
print("=" * 70)

print(f"\n[OPERATÖRLER]")
x, y = 10, 3
print(f"  {x} + {y} = {x + y}")
print(f"  {x} > {y} : {x > y}")
print(f"  {x} and {y} : {x and y}")
print(f"  {x} or {y}  : {x or y}")
print(f"  not {x}    : {not x}")

print(f"\n[F-STRING DETAYLI]")
isim, yas = "Zeynep", 24
print(f"  {isim=}, {yas=}")
print(f"  {3.14159265:.4f}")
print(f"  {255:010b}")
print(f"  {1234567:,}")

print(f"\n[LIST COMPREHENSION]")
kareler = [x**2 for x in range(1, 11)]
cift_kareler = [x**2 for x in range(1, 11) if x % 2 == 0]
print(f"  Kareler        : {kareler}")
print(f"  Çift kareler   : {cift_kareler}")

print(f"\n[DICTIONARY]")
sozluk = {"apple": "elma", "banana": "muz", "cherry": "kiraz"}
for k, v in sozluk.items():
    print(f"  {k:10s} → {v}")

print(f"\n[HATA YÖNETİMİ]")
def bolme(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Sıfıra bölünemez!"
    except TypeError:
        return "Geçersiz tipler!"
    finally:
        pass

print(f"  10 / 2 = {bolme(10, 2)}")
print(f"  10 / 0 = {bolme(10, 0)}")
print(f"  'a' / 2 = {bolme('a', 2)}")

print(f"\n[DOSYA İŞLEMLERİ]")
import tempfile, os
dosya_yolu = os.path.join(tempfile.gettempdir(), "ornek.txt")

with open(dosya_yolu, "w", encoding="utf-8") as f:
    f.write("Satır 1: Merhaba Dünya\n")
    f.write("Satır 2: Python öğreniyoruz\n")
    f.write("Satır 3: Dosya işlemleri kolay\n")

with open(dosya_yolu, "r", encoding="utf-8") as f:
    icerik = f.read()
    print(f"  Dosya içeriği:")
    for satir in icerik.strip().split("\n"):
        print(f"    {satir}")

os.remove(dosya_yolu)

print(f"\n[LAMBDA + MAP + FILTER]")
kareler = list(map(lambda x: x**2, range(1, 11)))
ciftler = list(filter(lambda x: x % 2 == 0, range(1, 21)))
print(f"  map(kare)     : {kareler}")
print(f"  filter(cift)  : {ciftler}")

from functools import reduce
toplam = reduce(lambda a, b: a + b, range(1, 11))
print(f"  reduce(topla) : {toplam}")

print(f"\n[ZIP + ENUMERATE]")
isimler = ["Ali", "Veli", "Ayşe"]
yaslar = [25, 30, 28]
for i, (isim, yas) in enumerate(zip(isimler, yaslar)):
    print(f"  {i}. {isim} → {yas} yaşında")

print(f"\n[TUPLE VE SET]")
t = (1, 2, 3, 4, 5)
print(f"  Tuple  : {t}")
print(f"  Index  : {t[2]}")

s = {1, 2, 3, 4, 5}
s.add(6)
print(f"  Set    : {s}")
print(f"  Union  : {s | {4, 5, 6, 7, 8}}")

print(f"\n[RECURSIVE FONKSİYON]")
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(f"  Fibonacci(10) = {fibonacci(10)}")
print(f"  İlk 10 terim:", [fibonacci(i) for i in range(10)])

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — Temel Python kavramları tamamlandı.")
print(f"{'='*70}")
