## Teknik Özellikler

### Model Mimarisi
* **Çok Katmanlı Perseptron (MLP) - İleri Beslemeli Sinir Ağı**

### Varsayılan Parametreler (Koddaki Örnek Corpus İle)
*Not: Bu değerler, sağlanan örnek `corpus`'a göre hesaplanmıştır ve farklı veri setlerinde değişiklik gösterecektir.*

* **Pencere Boyutu (`windowSize`):** 2
* **Kelime Dağarcığı Boyutu (`vocabularySize`):** 40
* **MLP Giriş Boyutu:** 80
* **Gizli Katman Yapısı:** 1 katman, 32 nöron
* **Çıkış Katmanı Nöron Sayısı:** 40
* **Toplam Öğrenilebilir Parametre Sayısı:** ~3,912

### Eğitim
* **Öğrenme Oranı (`learningRate`):** 0.02 (Varsayılan)
* **Epoch Sayısı (`epochs`):** 1000 (Varsayılan)
* **Optimizasyon Algoritması:** Stokastik Gradyan İnişi (SGD) ile Geri Yayılım
* **Kayıp Fonksiyonu:** Cross-Entropy Loss
* **JSON formatında kaydetme
* **JSON formatından yükleme
