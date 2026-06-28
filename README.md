# Derin Öğrenme ile Doğal Dil İşleme (NLP with DL)

Duygu analizi, adlandırılmış varlık tanıma ve metin sınıflandırması gibi temel NLP görevlerini derin öğrenme ve transformer modelleriyle ele alan Jupyter notebook koleksiyonu.

---

## Notebook'lar

### 💬 NLPwithDL.ipynb
Yelp restoran yorumları üzerinde derin öğrenme tabanlı ikili duygu analizi.

**Veri Ön İşleme:**
- Küçük harfe çevirme
- Regex ile noktalama ve sayı temizleme
- Boşluk normalizasyonu

**Özellik Mühendisliği:**
- CountVectorizer ile Bag-of-Words vektörleştirme
- N-gram çıkarımı (unigram + bigram)
- TextBlob ile lemmatizasyon
- Stopword temizleme

**Model:**
- Keras Sequential API ile yapay sinir ağı
- Dense katmanlar + Dropout regularizasyon
- İkili sınıflandırma: 1 yıldız (olumsuz) ve 5 yıldız (olumlu)

**Değerlendirme:** Doğruluk, sınıflandırma raporu, karmaşıklık matrisi

- **Veri Seti:** `yelp.csv` — Yelp restoran yorumları (1–5 yıldız)
- **Kütüphaneler:** TensorFlow / Keras, scikit-learn, TextBlob, pandas, numpy, matplotlib, seaborn

---

### 🏷️ NER.ipynb
Transformer tabanlı model ile Adlandırılmış Varlık Tanıma (Named Entity Recognition).

- Hugging Face Hub'dan önceden eğitilmiş model yükleme (~1.63 GB)
- spaCy entegrasyonu ile varlık çıkarımı ve sınıflandırma
- Tokenizasyon: vocab.json, merges.txt yapılandırmaları
- 515 parametreli transformer ağırlıklarının materialization süreci

- **Kütüphaneler:** spaCy, Transformers (Hugging Face), tokenizers

---

### 📩 HamorSpam.ipynb
Önceden eğitilmiş sınıflandırıcı modeli ile metin sınıflandırması.

- SafeTensors formatında ~442 MB model yükleme
- Hugging Face transformer mimarisi
- Tokenizer konfigürasyonu ve 201 parametre ağırlığı
- İkili sınıflandırma çıkışı (classifier.weight)

- **Kütüphaneler:** Transformers (Hugging Face), tokenizers

---

### 📖 QuranDataSet.ipynb
Kuran metni üzerinde doğal dil işleme analizi.

- Kuran veri seti üzerinde metin işleme ve analiz
- Dil modellemesi ve metin araştırması

---

## Kullanılan Teknolojiler

- Python 3
- Jupyter Notebook / Google Colab
- TensorFlow / Keras
- Hugging Face Transformers
- spaCy
- scikit-learn
- TextBlob
- pandas, numpy
- matplotlib, seaborn

---

## Başlarken

```bash
git clone https://github.com/rabiayel/NLPwithDL.git
cd NLPwithDL
pip install tensorflow transformers spacy textblob scikit-learn pandas seaborn
```

```python
# spaCy dil modelini indirmek için:
python -m spacy download en_core_web_sm

# NLTK paketleri için:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

> Transformer tabanlı notebook'lar (NER.ipynb, HamorSpam.ipynb) büyük model dosyaları indirdiğinden **Google Colab** ortamında çalıştırılması önerilir.
