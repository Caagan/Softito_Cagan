print("=" * 70)
print("  TEMEL PYTHON 2 — Tuple, Set, Lambda, Recursive, Dosya İşlemleri")
print("=" * 70)

print(f"\n[TUPLE]")
t = (1, 2, 3, 4, 5)
print(f"  Tuple     : {t}")
print(f"  Tip       : {type(t)}")
print(f"  Index     : {t[2]}")
print(f"  Slice     : {t[1:4]}")
print(f"  Uzunluk   : {len(t)}")

punktlu = (1, [2, 3], "a")
punktlu[1].append(4)
print(f"  Değişken tuple: {punktlu}")

print(f"\n[SET]")
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7, 8}
print(f"  A             : {a}")
print(f"  B             : {b}")
print(f"  Birleşim      : {a | b}")
print(f"  Kesişim       : {a & b}")
print(f"  Fark (A-B)    : {a - b}")
print(f"  Simetrik fark : {a ^ b}")
a.add(10)
a.discard(1)
print(f"  add/discard   : {a}")

print(f"\n[ENUMERATE]")
meyveler = ["elma", "armut", "muz", "çilek"]
for i, meyve in enumerate(meyveler, start=1):
    print(f"  {i}. {meyve}")

print(f"\n[ZIP]")
isimler = ["Ali", "Veli", "Ayşe"]
notlar = [85, 92, 78]
sehirler = ["İstanbul", "Ankara", "İzmir"]
for isim, notu, sehir in zip(isimler, notlar, sehirler):
    print(f"  {isim:8s} → Not: {notu}, Şehir: {sehir}")

print(f"\n[LAMBDA]")
kare = lambda x: x ** 2
topla = lambda a, b: a + b
print(f"  lambda x: x**2 → {kare(5)}")
print(f"  lambda a,b: a+b → {topla(3, 7)}")

print(f"\n  Lambda + sorted:")
ogrenciler = [("Ali", 85), ("Veli", 92), ("Ayşe", 78), ("Mehmet", 95)]
nota_gore = sorted(ogrenciler, key=lambda x: x[1], reverse=True)
for i, (isim, notu) in enumerate(nota_gore, 1):
    print(f"  {i}. {isim:8s} → {notu}")

print(f"\n[RECURSIVE FONKSİYONLAR]")
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(f"  Fibonacci(10) = {fibonacci(10)}")
print(f"  İlk 10: {[fibonacci(i) for i in range(10)]}")

def terscevir(metin):
    if len(metin) <= 1:
        return metin
    return terscevir(metin[1:]) + metin[0]

print(f"  terscevir('Python') = {terscevir('Python')}")

def powering(taban, us):
    if us == 0:
        return 1
    return taban * powering(taban, us - 1)

print(f"  powering(2, 10) = {powering(2, 10)}")

print(f"\n[DOSYA İŞLEMLERİ]")
import tempfile, os, json, csv

temp_dir = tempfile.gettempdir()
dosya = os.path.join(temp_dir, "ornek.json")
csv_dosya = os.path.join(temp_dir, "ornek.csv")

with open(dosya, "w", encoding="utf-8") as f:
    json.dump({"isim": "Ali", "yas": 25, "sehirler": ["İstanbul", "Ankara"]}, f, ensure_ascii=False, indent=2)

with open(dosya, "r", encoding="utf-8") as f:
    veri = json.load(f)
    print(f"  JSON okundu: {veri}")

with open(csv_dosya, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["isim", "yas", "not"])
    writer.writerow(["Ali", 25, 85])
    writer.writerow(["Veli", 30, 92])
    writer.writerow(["Ayşe", 28, 78])

with open(csv_dosya, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for satir in reader:
        print(f"  CSV: {satir}")

os.remove(dosya)
os.remove(csv_dosya)

print(f"\n[DİCT + SET COMPREHENSION]")
kareler_dict = {x: x**2 for x in range(1, 6)}
print(f"  Dict comp: {kareler_dict}")

cift_set = {x for x in range(1, 21) if x % 2 == 0}
print(f"  Set comp : {cift_set}")

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — Tuple, Set, Lambda, Recursive, Dosya İşlemleri.")
print(f"{'='*70}")
