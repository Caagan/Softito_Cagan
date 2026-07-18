print("=" * 70)
print("  PYTHON BAŞLANGIÇ — Değişkenler, Koşullar, Döngüler, Fonksiyonlar")
print("=" * 70)

isim = " Çağan"
yas = 25
boy = 1.78
ogrenci = True

print(f"\n[DEĞİŞKENLER]")
print(f"  İsim     : {isim}")
print(f"  Yaş      : {yas}")
print(f"  Boy      : {boy} m")
print(f"  Öğrenci  : {ogrenci}")
print(f"  Tip      : {type(isim)}")

print(f"\n[VERİ TİPLERİ]")
print(f"  int   : {yas} → {type(yas)}")
print(f"  float : {boy} → {type(boy)}")
print(f"  str   : '{isim}' → {type(isim)}")
print(f"  bool  : {ogrenci} → {type(ogrenci)}")

print(f"\n[DÖNÜŞTÜRME]")
print(f"  str(25)         = {str(25)}")
print(f"  int('42')       = {int('42')}")
print(f"  float(10)       = {float(10)}")
print(f"  bool(1)         = {bool(1)}")
print(f"  bool(0)         = {bool(0)}")

print(f"\n[ARİTMETİK İŞLEMLER]")
a, b = 15, 4
print(f"  {a} + {b}  = {a + b}")
print(f"  {a} - {b}  = {a - b}")
print(f"  {a} * {b}  = {a * b}")
print(f"  {a} / {b}  = {a / b:.2f}")
print(f"  {a} // {b} = {a // b}")
print(f"  {a} % {b}  = {a % b}")
print(f"  {a} ** {b} = {a ** b}")

print(f"\n[KARŞILAŞTIRMA İŞLETLERİ]")
print(f"  {a} > {b}   : {a > b}")
print(f"  {a} == {b}  : {a == b}")
print(f"  {a} != {b}  : {a != b}")

print(f"\n[KOŞULLAR (if/elif/else)]")
not_ort = 75
if not_ort >= 90:
    harf = "A"
elif not_ort >= 80:
    harf = "B"
elif not_ort >= 70:
    harf = "C"
elif not_ort >= 60:
    harf = "D"
else:
    harf = "F"
print(f"  Not Ortalaması: {not_ort} → Harf Notu: {harf}")

print(f"\n[FOR DÖNGÜSÜ]")
print(f"  1'den 5'e kadar:", end=" ")
for i in range(1, 6):
    print(i, end=" ")
print()

print(f"\n  Çift sayılar:", end=" ")
for i in range(1, 11):
    if i % 2 == 0:
        print(i, end=" ")
print()

print(f"\n[WHILE DÖNGÜSÜ]")
sayac = 5
while sayac > 0:
    print(f"  Geri sayım: {sayac}")
    sayac -= 1
print("  ¡Patlama!")

print(f"\n[LİSTELER]")
meyveler = ["elma", "armut", "muz", "çilek", "kivi"]
print(f"  Meyveler        : {meyveler}")
print(f"  Uzunluk         : {len(meyveler)}")
print(f"  İlk eleman      : {meyveler[0]}")
print(f"  Son eleman      : {meyveler[-1]}")
print(f"  Dilimleme [1:3] : {meyveler[1:3]}")
meyveler.append("portakal")
print(f"  Append sonrası  : {meyveler}")
meyveler.sort()
print(f"  Sıralanmış      : {meyveler}")

print(f"\n[LİST COMPREHENSION]")
kareler = [x**2 for x in range(1, 11)]
print(f"  Kareler         : {kareler}")
ciftler = [x for x in range(1, 21) if x % 2 == 0]
print(f"  Çiftler         : {ciftler}")

print(f"\n[FONKSİYONLAR]")
def selamla(isim, dil="tr"):
    if dil == "tr":
        return f"Merhaba, {isim}!"
    elif dil == "en":
        return f"Hello, {isim}!"
    else:
        return f"Salut, {isim}!"

print(f"  {selamla('Ali')}")
print(f"  {selamla('John', 'en')}")
print(f"  {selamla('Pierre', 'fr')}")

def faktoriyel(n):
    if n <= 1:
        return 1
    return n * faktoriyel(n - 1)

print(f"\n  5! = {faktoriyel(5)}")
print(f"  10! = {faktoriyel(10)}")

print(f"\n[*args VE **kwargs]")
def topla(*sayilar):
    return sum(sayilar)

def bilgi(**kwargs):
    for k, v in kwargs.items():
        print(f"    {k}: {v}")

print(f"  topla(1,2,3,4,5) = {topla(1, 2, 3, 4, 5)}")
print(f"  Bilgi:")
bilgi(isim="Ahmet", yas=30, sehir="İstanbul")

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — Temel Python kavramları.")
print(f"{'='*70}")
