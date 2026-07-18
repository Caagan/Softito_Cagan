# SLM — Türkçe Küçük Dil Modeli

Türkçe Wikipedia verisi üzerinde eğitilen MiniGPT tabanlı küçük dil modeli.

## Yapi

```
turkce-slm/
├── configs/config.yaml    # Model ve eğitim parametreleri
├── src/
│   ├── data_collection.py # Veri toplama
│   ├── dataset.py         # Dataset ve DataLoader
│   ├── preprocessing.py   # Metin ön işleme
│   ├── train.py           # Eğitim döngüsü
│   ├── generate.py        # Metin üretim ve tokenizer
│   └── models/
│       └── transformer.py # MiniGPT modeli
├── demo/app.py            # Demo uygulaması
├── tests/test_tokenizer.py # Unit testler
├── checkpoints/           # Model checkpoint'leri
└── figures/               # Eğitim grafikleri
```

## Kurulum

```bash
pip install torch matplotlib numpy
python src/train.py
python demo/app.py
```
