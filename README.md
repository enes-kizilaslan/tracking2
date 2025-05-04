# Nörogelişimsel Bozukluk Tarama Sistemi

Bu proje, çocuklarda nörogelişimsel bozuklukların erken tespiti için geliştirilmiş bir tarama sistemidir.

## 🚀 Özellikler

- 145 farklı makine öğrenmesi modeli
- Çoklu nörogelişimsel bozukluk taraması
- Kullanıcı dostu arayüz
- Anlık risk değerlendirmesi
- Detaylı sonuç raporlama

## 📦 Gerekli Dosyalar

Uygulamanın çalışması için aşağıdaki dosyaların proje dizininde bulunması gerekir:

1. `models.zip` - Eğitilmiş modelleri içeren zip dosyası
2. `cevaplar600.csv` - Soru havuzu
3. `model_performance.xlsx` - Model performans metrikleri
4. `selected_features.xlsx` - Her model için seçilmiş özellikler

## 🛠️ Kurulum

1. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

2. Streamlit uygulamasını başlatın:
```bash
streamlit run app.py
```

## 📝 Kullanım

1. Tüm soruları dikkatlice okuyun
2. Her soru için en uygun cevabı seçin
3. Tüm soruları yanıtladıktan sonra "Değerlendir" butonuna tıklayın
4. Sistem size risk değerlendirmesini gösterecektir

## ⚠️ Önemli Not

Bu sistem bir ön değerlendirme aracıdır ve kesin tanı koyamaz. Mutlaka bir uzmana danışınız.

## 🔧 Teknik Detaylar

- Python 3.8+
- Streamlit 1.22.0
- scikit-learn 0.24.2
- numpy 1.21.6
- pandas 1.5.3

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 