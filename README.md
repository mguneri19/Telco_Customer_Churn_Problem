# Telco Customer Churn Prediction

## 📊 Proje Hakkında

Bu proje, telekomünikasyon şirketinin müşteri kaybını (churn) tahmin etmek için makine öğrenmesi modelleri geliştirmeyi amaçlamaktadır. Proje kapsamlı bir veri analizi, özellik mühendisliği ve model optimizasyonu sürecini içermektedir.

## 🎯 Problem Tanımı

**Problem**: Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirmek.

**Veri Seti**: Telco müşteri churn verileri, üçüncü çeyrekte Kaliforniya'daki 7,043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir.

## 📈 Veri Seti Özellikleri

- **Gözlem Sayısı**: 7,043 müşteri
- **Değişken Sayısı**: 21 özellik
- **Hedef Değişken**: Churn (Yes/No)

### Önemli Değişkenler:
- `tenure`: Müşterinin şirkette kaldığı ay sayısı
- `Contract`: Sözleşme türü (Month-to-month, One year, Two year)
- `MonthlyCharges`: Aylık ücret
- `InternetService`: İnternet servis türü
- `PaymentMethod`: Ödeme yöntemi

## 🔍 Keşifsel Veri Analizi Bulguları

### Kritik İçgörüler:
1. **Tenure**: Ayrılan müşteriler ortalama 17.98 ay, kalan müşteriler 37.57 ay
2. **Kontrat Türü**: Month-to-month müşterilerde %42.7 churn oranı
3. **Ödeme Yöntemi**: Electronic check kullananlarda %45.3 churn oranı
4. **İnternet Servisi**: Fiber optic kullanıcılarında %41.9 churn oranı

## 🛠️ Özellik Mühendisliği

### Oluşturulan Yeni Özellikler:
- `NEW_TENURE_YEAR`: Tenure'a göre yıllık kategoriler
- `NEW_Engaged`: Uzun süreli kontrat müşterileri
- `NEW_TotalServices`: Toplam servis sayısı
- `NEW_AVG_Charges`: Ortalama aylık ücret
- `NEW_FLAG_AutoPayment`: Otomatik ödeme durumu
- `NEW_FLAG_Family`: Aile müşterisi durumu

## 🤖 Model Performansları

### En İyi Modeller:

| Model | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|---------|
| **Logistic Regression** | 0.8046 | 0.5905 | 0.8493 |
| **CatBoost** | 0.8019 | 0.5810 | 0.8480 |

### En Önemli Özellikler:
1. **Tenure** (Müşteri süresi)
2. **Contract Type** (Sözleşme türü)
3. **Engagement Status** (Bağlılık durumu)
4. **Internet Service** (İnternet servisi)
5. **Payment Method** (Ödeme yöntemi)

## 📁 Proje Yapısı

```
TelcoChurn/
├── telco_churn.py          # Ana analiz dosyası
├── data/
│   └── Telco-Customer-Churn.csv # Veri seti
├── README.md               # Bu dosya
└── requirements.txt        # Gerekli kütüphaneler
```

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler:
```bash
pip install -r requirements.txt
```

### Çalıştırma:
```bash
python telco_churn.py
```

## 📊 Sonuçlar ve Öneriler

### İş Önerileri:
1. **Month-to-month müşteriler** için özel kampanyalar
2. **Fiber optic kullanıcıları** için teknik destek artırımı
3. **Otomatik ödeme** teşvikleri
4. **İlk 24 ay** kritik dönem - özel ilgi gerekli

### Model Kullanımı:
- Logistic Regression: Hızlı tahmin ve yorumlanabilirlik için
- CatBoost: Yüksek performans gerektiren durumlar için

