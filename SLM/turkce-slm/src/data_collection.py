import urllib.request
import os

def download_wikipedia_dumps(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    url = "https://dumps.wikimedia.org/trwiki/latest/trwiki-latest-pages-articles1.xml-p1p41242.bz2"
    output_path = os.path.join(output_dir, "trwiki-dump.xml.bz2")

    if os.path.exists(output_path):
        print(f"    Dump zaten mevcut: {output_path}")
        return output_path

    print(f"    Indiriliyor: {url}")
    print(f"    Hedef: {output_path}")
    urllib.request.urlretrieve(url, output_path)
    print(f"    Indirme tamamlandi: {os.path.getsize(output_path) / 1e6:.1f} MB")
    return output_path

if __name__ == "__main__":
    download_wikipedia_dumps()
