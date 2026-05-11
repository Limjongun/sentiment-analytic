# TikTok Review Sentiment Analysis using Naive Bayes

Project ini merupakan project sederhana untuk melakukan **analisis sentimen review aplikasi TikTok** dari Google Play Store menggunakan Python, NLP preprocessing, TF-IDF, dan algoritma **Bernoulli Naive Bayes**.

Dataset diambil langsung dari Google Play Store menggunakan library `google-play-scraper`, kemudian review diberi label sentimen berdasarkan rating pengguna.

---

## Tujuan Project

Tujuan utama project ini adalah membangun pipeline sederhana untuk analisis sentimen teks berbahasa Indonesia.

Project ini mencakup beberapa tahap penting dalam Natural Language Processing dan Machine Learning, yaitu:

1. Scraping review aplikasi dari Google Play Store.
2. Menyimpan data mentah ke file JSON.
3. Memberikan label sentimen berdasarkan rating.
4. Membersihkan teks review.
5. Melakukan tokenisasi, stopword removal, dan stemming.
6. Mengubah teks menjadi fitur numerik menggunakan TF-IDF.
7. Melatih model Bernoulli Naive Bayes.
8. Mengevaluasi performa model menggunakan akurasi dan classification report.

---

## Dataset

Dataset diperoleh dari review aplikasi TikTok di Google Play Store.

App ID yang digunakan:

```python
com.zhiliaoapp.musically
