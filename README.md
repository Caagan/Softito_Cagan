# Yapay Zeka Projeleri

Makine öğrenmesi, derin öğrenme, doğal dil işleme ve büyük dil modelleri alanlarında geliştirdiğim projelerin bulunduğu ana repo.

## Yapı

```
proje-reposu/
├── Airflow/              # ETL pipeline simülasyonları
├── BigData/              # Büyük veri / dağıtık işleme projeleri
├── DeepLearning/         # Derin öğrenme projeleri
├── Docker/               # Docker & konteynerizasyon projeleri
├── EDA/                  # Keşifsel veri analizi projeleri
├── LLM/                  # Büyük dil modeli projeleri
├── MachineLearning/      # Makine öğrenmesi projeleri
├── NLP/                  # Doğal Dil İşleme projeleri
├── Python/               # Python projeleri
└── SLM/                  # Küçük dil modeli projeleri
```

> **Not:** Tüm projeler gerçek veri setleriyle çalışır. Sentetik veri üretimi yoktur.

---

## Airflow

ETL pipeline yönetimi ve Apache Airflow simülasyonu.

| # | Proje | Teknoloji |
|---|-------|-----------|
| 01 | [etl-pipeline-simulasyon](Airflow/etl-pipeline-simulasyon) | Python |

---

## BigData

Büyük veri işleme ve dağıtık hesaplama projeleri.

| # | Proje | Veri | Teknoloji |
|---|-------|------|-----------|
| 01 | [big-data-log-analytics](BigData/big-data-log-analytics) | SF 311 Vatandaş Şikayet Dataseti | PySpark |

---

## DeepLearning

| # | Proje | Veri | Teknoloji |
|---|-------|------|-----------|
| 01 | [fashion-mnist-cnn](DeepLearning/CNN/01-fashion-mnist-cnn) | Fashion MNIST (torchvision) | PyTorch |
| 02 | [goruntu-on-isleme](DeepLearning/CNN/02-goruntu-on-isleme) | Sklearn Digits | Scikit-learn |
| 03 | [opencv-plaka](DeepLearning/CNN/03-opencv-plaka) | Gerçek plaka görselleri | OpenCV |
| 04 | [cnn-siniflandirma](DeepLearning/CNN/04-cnn-siniflandirma) | torchvision verileri | PyTorch |
| 05 | [GRU](DeepLearning/GRU) | Gerçek zaman serisi verisi | PyTorch |

---

## Docker

| # | Proje | Teknoloji |
|---|-------|-----------|
| 01 | [1-single-container-xgboost](Docker/1-single-container-xgboost) | Docker, XGBoost |
| 02 | [2-docker-compose-3ml](Docker/2-docker-compose-3ml) | Docker Compose, XGBoost |
| 03 | [3-microservices-ml-gateway](Docker/3-microservices-ml-gateway) | Flask, Docker Compose |

---

## EDA

Keşifsel veri analizi projeleri — Teen Mental Health (Kaggle) veri seti üzerinde.

| # | Proje | Konu | Teknoloji |
|---|-------|------|-----------|
| 01 | [veri-yukleme-genel-bakis](EDA/01_veri_yukleme_genel_bakis) | Veri yükleme, boyut, tipler | Pandas |
| 02 | [veri-temizleme](EDA/02_veri_temizleme) | Eksik veri, aykırı değerler | Pandas, NumPy |
| 03 | [tek-degiskenli-analiz](EDA/03_tek_degiskenli_analiz) | Tek değişkenli analiz | Matplotlib, Seaborn |
| 04 | [cift-degiskenli-analiz](EDA/04_cift_degiskenli_analiz) | İkili analiz, korelasyon | Seaborn |
| 05 | [feature-engineering](EDA/05_feature_engineering) | Özellik türetme | Pandas, Scikit-learn |

---

## LLM

| # | Proje | Konu | Teknoloji |
|---|-------|------|-----------|
| 01 | [llm-karsilastirma](LLM/01-llm-karsilastirma) | LLM model karşılaştırma | distilgpt2 |
| 02 | [fine-tuning-lora](LLM/02-fine-tuning-lora) | LoRA ile ince ayar | PyTorch, LoRA |
| 03 | [langchain](LLM/03-langchain) | LangChain pipeline | LangChain, Gemini |
| 04 | [rag](LLM/04-rag) | RAG sistemi | FAISS, Gemini |
| 05 | [prompt-engineering](LLM/05-prompt-engineering) | Prompt mühendisliği | distilgpt2 |
| 06 | [rlhf-dpo](LLM/06-rlhf-dpo) | DPO eğitimi | PyTorch |
| 07 | [quantization](LLM/07-quantization) | Kuantizasyon | PyTorch |

---

## MachineLearning

### Supervised (Denetimli Öğrenme)

| # | Proje | Veri | Teknoloji |
|---|-------|------|-----------|
| 01 | [linear-regresyon](MachineLearning/Supervised/01-linear-regresyon) | California Housing (sklearn) | Scikit-learn |
| ↳ | [pay-equity-analysis](MachineLearning/Supervised/01-linear-regresyon/pay-equity-analysis) | California Housing + Cinsiyet | Scikit-learn |
| ↳ | [superlig-goal-prediction](MachineLearning/Supervised/01-linear-regresyon/superlig-goal-prediction) | Futbol istatistikleri | Scikit-learn |
| 02 | [logistic-regresyon](MachineLearning/Supervised/02-logistic-regresyon) | Credit Card Fraud (Kaggle) | Scikit-learn |
| ↳ | [churn-prediction](MachineLearning/Supervised/02-logistic-regresyon/churn-prediction) | Telco Churn | Scikit-learn |
| ↳ | [credit-scoring](MachineLearning/Supervised/02-logistic-regresyon/credit-scoring) | Credit Card Fraud (Kaggle) | Scikit-learn |
| 03 | [decision-tree](MachineLearning/Supervised/03-decision-tree) | Breast Cancer (sklearn) | Scikit-learn |
| ↳ | [decision_tree_clinical](MachineLearning/Supervised/03-decision-tree/decision_tree_clinical) | Breast Cancer (sklearn) | Scikit-learn |
| ↳ | [mobile-price-decision-tree](MachineLearning/Supervised/03-decision-tree/mobile-price-decision-tree) | Cihaz özellikleri | Scikit-learn |
| 04 | [fraud-detection-rf](MachineLearning/Supervised/04-random-forest) | Credit Card Fraud (Kaggle) | Scikit-learn |
| 05 | [bank-campaign-lightgbm](MachineLearning/Supervised/05-lightgbm) | Bank Marketing (UCI) | LightGBM |
| 06 | [svm-tumor-diagnosis](MachineLearning/Supervised/06-svm) | Breast Cancer (sklearn) | Scikit-learn |
| 07 | [knn-recommender](MachineLearning/Supervised/07-knn) | Iris (sklearn) | Scikit-learn |
| 08 | [sentiment-naive-bayes](MachineLearning/Supervised/08-naive-bayes) | 20 Newsgroups (sklearn) | Scikit-learn |
| — | [ml-karsilastirma](MachineLearning/Supervised/ml-karsilastirma) | Breast Cancer (sklearn) | XGBoost, LightGBM |
| ↳ | [logreg-vs-randomforest](MachineLearning/Supervised/ml-karsilastirma/logreg-vs-randomforest-diabetes) | Pima Diabetes (UCI) | Scikit-learn |
| ↳ | [xgboost-vs-lightgbm](MachineLearning/Supervised/ml-karsilastirma/xgboost-vs-lightgbm) | make_classification | XGBoost, LightGBM |

### Unsupervised (Denetimsiz Öğrenme)

| # | Proje | Veri | Teknoloji |
|---|-------|------|-----------|
| 01 | [ecommerce-segmentation](MachineLearning/Unsupervised/01-kmeans) | Wine (sklearn) | Scikit-learn, PCA |
| 02 | [insurance-segmentation](MachineLearning/Unsupervised/02-clustering-comparison) | Wine (sklearn) | Scikit-learn |
| 03 | [isolation-forest-fraud](MachineLearning/Unsupervised/03-isolation-forest) | Credit Card Fraud (Kaggle) | Scikit-learn |
| 04 | [ocsvm-intrusion-detection](MachineLearning/Unsupervised/04-one-class-svm) | make_blobs (sklearn) | Scikit-learn |

---

## NLP

| # | Proje | Konu | Teknoloji |
|---|-------|------|-----------|
| 01 | [tf-idf](NLP/01-tf-idf) | TF-IDF temsili | Scikit-learn |
| 02 | [word-embeddings](NLP/02-word-embeddings) | Word2Vec, GloVe, FastText | PyTorch |
| 03 | [rnn](NLP/03-rnn) | RNN | PyTorch |
| 04 | [lstm](NLP/04-lstm) | LSTM | PyTorch |
| 05 | [attention](NLP/05-attention) | Attention mekanizması | PyTorch |
| 06 | [transformer](NLP/06-transformer) | Transformer mimarisi | PyTorch |

---

## Python

| Dosya | Konu | Seviye |
|-------|------|--------|
| `python_baslangic.py` | Temel Python | Başlangıç |
| `python_1_ders.py` | Sınıflar, kalıtım | Temel |
| `temel_python.py` | Operatörler, hata yönetimi | Temel |
| `temel_python_2.py` | Tuple, set, dosya işlemleri | Orta |
| `python_class_sorular.py` | @property, iterator, Mixin | İleri |

---

## SLM

Küçük dil modelleri — Türkçe Wikipedia üzerinde MiniGPT.

| # | Proje | Veri | Teknoloji |
|---|-------|------|-----------|
| 01 | [turkce-slm](SLM/turkce-slm) | Türkçe Wikipedia (HuggingFace) | PyTorch |

---

## Teknolojiler

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB5B25?style=flat)
![LightGBM](https://img.shields.io/badge/LightGBM-9ACD32?style=flat)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=flat&logo=apachespark&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat)

## Lisans

MIT
