"""
A Telecom Company - Customer Churn Prediction
================================

Bu proje, telekomünikasyon şirketinin müşteri kaybını (churn) tahmin etmek için 
makine öğrenmesi modelleri geliştirmeyi amaçlamaktadır.

Author: Muhammet Güneri
Date: 2025

Proje İçeriği:
- Keşifsel Veri Analizi (EDA)
- Özellik Mühendisliği
- Model Geliştirme ve Optimizasyon
- Performans Değerlendirmesi

"""

"""
Problem : Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirmek.
Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımları yapılacak.

Telecom şirketi müşteri churn verileri, üçüncü çeyrekte 7043 müşteriye
ev telefonu ve İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir.

Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu içermektedir.

"""

# Gerekli Kütüphane ve Fonksiyonlar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


print("Telco Customer Churn Analizi Başlıyor...")
print("=" * 50)

# Veri setini yükle
print("1. Veri seti yükleniyor...")
df = pd.read_csv("data/Telco-Customer-Churn.csv")
print(f"Veri seti başarıyla yüklendi!")
print(df.shape) # 7043 gözlem, 21 değişken var
df.head()
df.info()
df["customerID"].nunique()  #7043 yani her bir gözlem eşsiz




"""
CustomerId : Müşteri İd’si
Gender : Cinsiyet
SeniorCitizen : Müşterinin yaşlı olup olmadığı (1, 0)
Partner : Müşterinin bir ortağı olup olmadığı (Evet, Hayır) ? Evli olup olmama
Dependents : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır) (Çocuk, anne, baba, büyükanne)
tenure : Müşterinin şirkette kaldığı ay sayısı
PhoneService : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
MultipleLines : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
InternetService : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
OnlineSecurity : Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
OnlineBackup : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
TechSupport : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
StreamingTV : Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin, bir üçüncü taraf sağlayıcıdan televizyon programları yayınlamak için İnternet hizmetini kullanıp kullanmadığını gösterir
StreamingMovies : Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok) Müşterinin bir üçüncü taraf sağlayıcıdan film akışı yapmak için İnternet hizmetini kullanıp kullanmadığını gösterir
Contract : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
PaymentMethod : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar
TotalCharges : Müşteriden tahsil edilen toplam tutar
Churn : Müşterinin kullanıp kullanmadığı (Evet veya Hayır) - Geçen ay veya çeyreklik içerisinde ayrılan müşteriler


"""



# TotalCharges sayısal bir değişken olmalı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.info()
df.head() # Yes/No olan kategorik değişkenler var. Eğer sadece 2 tane sınıf varsa, bunları 1/0 atayabilirim

# Yes/No değerlerini 1/0'a dönüştürme fonksiyonu
def convert_yes_no_to_binary(df):
    """
    Yes/No değerlerini 1/0'a dönüştüren fonksiyon
    """
    print("Yes/No değerlerini 1/0'a dönüştürülüyor...")

    # Yes/No içeren sütunları bul
    yes_no_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_values = df[col].unique()
            if len(unique_values) == 2 and 'Yes' in unique_values and 'No' in unique_values:
                yes_no_columns.append(col)

    print(f"Bulunan Yes/No sütunları: {yes_no_columns}")

    # Her sütunu dönüştür
    for col in yes_no_columns:
        print(f"Dönüştürülüyor: {col}")
        df[col] = df[col].map({'Yes': 1, 'No': 0})
        print(f"  - {col}: Yes -> 1, No -> 0")

    return df

df = convert_yes_no_to_binary(df)

#Bulunan Yes/No sütunları: ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

print("\nDönüştürme sonrası veri tipleri:")
print(df.dtypes)

df.head()

##################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# GENEL RESİM
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.select_dtypes(include=['number']).quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

"""
Observations: 7043
Variables: 21
cat_cols: 17
num_cols: 3
cat_but_car: 1 -- kategorik ama sınıf sayısı yüksek
num_but_cat: 6 -- numerik ama sınıf sayısı düşük, kategorik gibi davranıyor.
"""

##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
    }))
    print("##########################################")
    if plot:
        sns.countplot(x=col_name, data=dataframe)
        plt.title(col_name)
        plt.xticks(rotation=45)  # okunabilirlik için
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# Tenure'e bakıldığında 1 aylık müşterilerin çok fazla olduğunu ardından da 70 aylık müşterilerin geldiğini görüyoruz.
# Bu durum farklı kontratlardan dolayı olmuş olabilir.
# Aylık sözleşmesi olan kişilerin tenure'u ile 2 yıllık sözleşmesi olan kişilerin tenure'na bakalım.
df[df["Contract"] == "Month-to-month"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Month-to-month")
plt.show(block=True)

df[df["Contract"] == "Two year"]["tenure"].hist(bins=20)
plt.xlabel("tenure")
plt.title("Two year")
plt.show(block=True)
#Düşündüğümüz gibi - Aylık kontrat sahiplerinin tenure 1 aylık, 2 yıllık kontrat sahiplerinin tenure 70 aylık

# MonthyChargers'a bakıldığında aylık sözleşmesi olan müşterilerin aylık ortalama ödemeleri daha fazla olabilir.
df[df["Contract"] == "Month-to-month"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Month-to-month")
plt.show(block=True)

df[df["Contract"] == "Two year"]["MonthlyCharges"].hist(bins=20)
plt.xlabel("MonthlyCharges")
plt.title("Two year")
plt.show(block=True)
#aynen düşündüğümüz gibi aylık ödeme ortalamaları daha fazla

##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

"""

Tenure: 37.57 [Churn=0], 17.98 [Churn=1] -- Ortalama 20 ay fark var. Ayrılanlar çok daha kısa süre kalıyor. İlk 24 ay kritik eşik
MonthlyCharges: 61.27 [Churn=0], 74.44 [Churn=1]  -- Ortalama 13 dolar daha fazla. Ayrılanlar daha yüksek paketlerde, fiyat hassasiyeti yüksek
TotalCharges: 2555 [Churn=0], 1531 [Churn=1] -- Yaklaşık 1000'lık fark var. Kalan müşteriler çok daha yüksek yaşam boyu değer bırakıyor. (LTV)

"""

##################################
# KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################


def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

"""
Kontrat süresi churn’ün en güçlü belirleyicilerinden -- Month-to-month: %42.7 churn → en riskli grup.
Otomatik ödeme → sadakat artırıyor, otomatik ödemede churn: %15, Electronic check: %45.3 churn → çok yüksek risk.
Fiber optic: %41.9 churn (çok riskli) -- Fiber optik kullanıcılarının churn yüksek
OnlineSecurity = No: %41.8 churn. TechSupport = No: %41.6 churn. Backup/DeviceProtection = No: %39–40 churn.
(bu ek hizmetlere sahip olmayanların churn oranları yüksek -- Paketlere bu servisleri bundle edip cazip fiyatla sunmak churn’u düşürür)
SeniorCitizen: %41.7 churn → yaşlı müşteriler daha riskli.

"""

##################################
# KORELASYON
##################################

df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

"""
Müşteri şirkette ne kadar uzun kalırsa, toplam ödemesi o kadar artıyor diyebiliriz.
Aylık ücret arttıkça toplam ödemeler de artıyor ama tenure süresine göre değişiyor.
tenure + TotalCharges çok yüksek korelasyonlu → modelde ikisini birden kullanmak multicollinearity oluşturabilir.
Biri çıkarılabilir veya tenure_bin gibi kategorik dönüşüm yapılabilir.

"""
#Churn ile numerik değerlerin korelasyonuna bakalım.
df[num_cols + ["Churn"]].corr()["Churn"].sort_values(ascending=False)

"""
tenure ile total charges -- 0.83 korelasyon var, çok yüksek -- multicollinearity riski var. Birinden birini modelden atabiliriz
tenure ile churn ters korelasyonlu -0.35 -- kesinlikle tenure modelimizde kalmalı.tenure arttıkça churn ihtimali azalıyor.
Modelde tenure kalıyor, multicollinearity nedeniyle total charge çıkarılıp, yerine onun türevi kullanılabilir.
(örn. avg_monthly_charge = TotalCharges / tenure).

"""

##################################
# EKSİK DEĞER ANALİZİ
##################################

df.isnull().sum() # Sadece total charges da 11 tane değer boş
len(df) #7043
#7043te 11 değer çok az -- ihmal edilebilir.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True) #Total Charges -- 11 değer, %0.16

"""
TotalCharges = tenure * MonthlyCharges aslında. Tenure 0 ise, totalcharges 0 olabilir, 
ama tenure 0 değilse, tenure * monthly charges atayabilirim.

"""
#Eğer totalcharges boş ve tenure da 0 ise Total Charges = 0 ata
df.loc[(df["TotalCharges"].isnull()) & (df["tenure"] == 0), "TotalCharges"] = 0

#TotalCharges taki boş değerlere tenure * montlycharges işleminin sonucuyla doldur
df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])

#Boş değer var mı
df.isnull().sum() #yok artık


##################################
# AYKIRI DEĞER ANALİZİ
##################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Calculates the lower and upper thresholds for outlier detection in a numerical column
    using the Interquartile Range(IQR) method between the specified quantiles.
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    """
       Checks whether a numerical column contains any outliers based on the
       thresholds calculated with the Interquartile Range (IQR) method
       between the 5th and 95th percentiles.

       Returns:
           True if at least one outlier exists, otherwise False.
       """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    """
        Replaces outlier values in a numerical column with the calculated lower and upper thresholds
        using the Interquartile Range (IQR) method between the specified quantiles

        Any values below the lower threshold are set to the lower threshold,
        and values above the upper threshold are set to the upper threshold.
        """
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Aykırı Değer Analizi
for col in num_cols:
    print(col, check_outlier(df, col))
"""
tenure False
MonthlyCharges False
TotalCharges False

aykırı değer yok
"""

##################################
# ÖZELLİK ÇIKARIMI
##################################

# Tenure değişkeninden yıllık kategorik değişken oluşturma
df["tenure"].max() #max. 72 ay bu demek 6 yıl
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"

# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)

# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# ortalama aylık ödeme
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)

#Paperless & pahalı (median üstü)
mc_median = df["MonthlyCharges"].median()
# PaperlessBilling pahalı mı değil mi
df["NEW_FLAG_Paperless_Expensive"] = ((df["PaperlessBilling"] == 1) & (df["MonthlyCharges"] > mc_median)).astype(int)

# Senior + TechSupport yok
df["NEW_Senior_NoTechSupport"] = ((df["SeniorCitizen"] == 1) & (df["TechSupport"] == "No")).astype(int)

# Sadece telefon / sadece internet
df["NEW_FLAG_OnlyPhone"]    = ((df["PhoneService"] == 1) & (df["InternetService"] == "No")).astype(int)
df["NEW_FLAG_OnlyInternet"] = ((df["PhoneService"] == 0) & (df["InternetService"] != "No")).astype(int)

# Mevcut NEW_TotalServices sayımın üstünden "yüksek paket" işareti:
df["NEW_FLAG_FullBundle"]  = (df["NEW_TotalServices"] >= 5).astype(int)

# Aile sinyali (partner veya dependents var)
df["NEW_FLAG_Family"] = ((df["Partner"] == 1) | (df["Dependents"] == 1)).astype(int)

# Yüksek gelir riski: ücret üst %20 + aylık kontrat
q80 = df["MonthlyCharges"].quantile(0.80)
df["NEW_FLAG_HighRevenueRisk"] = ((df["MonthlyCharges"] >= q80) & (df["Contract"] == "Month-to-month")).astype(int)

# Fiber + streaming kombinasyonu
df["NEW_Fiber_Streaming"] = ((df["InternetService"] == "Fiber optic") &
                             ((df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes"))).astype(int)

# Uzun süre bayrağı (>=24 ay)
df["NEW_FLAG_LongTenure"] = (df["tenure"] >= 24).astype(int)

# Hizmet yoğunluğu (tenure'a göre)
df["NEW_Service_Density"] = df["NEW_TotalServices"] / (df["tenure"] + 1)

df.head()
print(df.shape) #7031 gözlem, 41 değişken


##################################
# ENCODING
##################################

# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_names(df)

"""
Observations: 7043
Variables: 41
cat_cols: 33
num_cols: 7
cat_but_car: 1
num_but_cat: 21

"""
print(cat_cols)
print(num_cols)
print(cat_but_car)

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    """
     Encodes a binary categorical column into numeric labels (0 and 1).
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
print(binary_cols)

for col in binary_cols:
    df = label_encoder(df, col)

# ONE-HOT ENCONDING

# cat_cols listesinin güncelleme işlemi
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
print(cat_cols)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
    Applies one-hot encoding to given categorical columns.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True) #enconding false-true diye döndürdü

#booleanları integere çeviriyoruz
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

df.head()
print(num_cols)
"""
7 tane numerik değerlerim var.
['tenure', 'MonthlyCharges', 'TotalCharges', 'NEW_AVG_Charges', 'NEW_Increase', 'NEW_AVG_Service_Fee', 'NEW_Service_Density']
"""

#Eğer mesafeye dayalı algoritmalar kullanacaksam standartlaşma yapmam lazım
#Eğer ağaca dayalı algoritmalarda standartlaşmaya gerek yok
#Bu yüzden hybrid bir yöntem kullansam


##################################
# MODELLEME
##################################

y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Mesafeye dayalı modeller için numerik değerleri scaler yapıyorum
####################################
# Logistic Regression pipeline
logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("log_reg", LogisticRegression(max_iter=1000))
])

# KNN pipeline
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# SVM pipeline
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])
############################################

# Ağaç tabanlı modellerde scaler yok
rf_model = RandomForestClassifier()
lgbm_model = LGBMClassifier()
xgb_model = XGBClassifier()
cat_model = CatBoostClassifier(verbose=0)

###########################################

models = {
    "Logistic Regression": logistic_pipeline,
    "KNN": knn_pipeline,
    "SVM": svm_pipeline,
    "RandomForest": rf_model,
    "LightGBM": lgbm_model,
    "XGBoost": xgb_model,
    "CatBoost": cat_model
}

from sklearn.model_selection import cross_val_score

for name, model in models.items():
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

df.head()
"""
Logistic Regression ve CatBoost öne çıkıyor.

########## Logistic Regression ##########
Accuracy: 0.8043
Auc: 0.8491
Recall: 0.5308
Precision: 0.6655
F1: 0.5899

########## CatBoost ##########
Accuracy: 0.7965
Auc: 0.8404
Recall: 0.519
Precision: 0.6454
F1: 0.5751
"""
##################################
# LOGISTIC REGRESSION OPTIMIZATION
##################################

print("\n" + "="*50)
print("LOGISTIC REGRESSION OPTIMIZATION")
print("="*50)

logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("log_reg", LogisticRegression(max_iter=5000))
])

params_logistic = {
    "log_reg__C": [0.01, 0.1, 1, 10],
    "log_reg__penalty": ["l1", "l2"],
    "log_reg__solver": ["liblinear", "saga"]
}

lr_best_grid= GridSearchCV(logistic_pipeline, params_logistic, cv=10,
                        scoring="roc_auc", n_jobs=-1) #churn probleminde ayırma önemli olduğu için roc_auc
lr_best_grid.fit(X, y)

print("Best Logistic Params:", lr_best_grid.best_params_)
print("Best Logistic AUC:", round((lr_best_grid.best_score_),4))

lr_final = logistic_pipeline.set_params(**lr_best_grid.best_params_).fit(X, y)


cv_results = cross_validate(lr_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
print("Logistic Regression Final Results:")
print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
print(f"F1 Score: {round(cv_results['test_f1'].mean(), 4)}")
print(f"ROC AUC: {round(cv_results['test_roc_auc'].mean(), 4)}")

"""
Logistic Regression Final Results:
Accuracy: 0.8046
F1 Score: 0.5905
ROC AUC: 0.8493
"""

##################################
# CATBOOST OPTIMIZATION
##################################

print("\n" + "="*50)
print("CATBOOST OPTIMIZATION")
print("="*50)

# CatBoost için hyperparameter optimization
cat_params = {
    'iterations': [100, 200, 300],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.3],
    'l2_leaf_reg': [1, 3, 5, 7]
}

cat_model = CatBoostClassifier(verbose=0, random_state=42)
cat_grid = GridSearchCV(cat_model, cat_params, cv=10, scoring="roc_auc", n_jobs=-1)
cat_grid.fit(X, y)

print("Best CatBoost Params:", cat_grid.best_params_)
print("Best CatBoost AUC:", round(cat_grid.best_score_, 4))

# Final CatBoost model
catboost_final = CatBoostClassifier(**cat_grid.best_params_, verbose=0, random_state=42)
catboost_final.fit(X, y)

cat_cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
print("\nCatBoost Final Results:")
print(f"Accuracy: {round(cat_cv_results['test_accuracy'].mean(), 4)}")
print(f"F1 Score: {round(cat_cv_results['test_f1'].mean(), 4)}")
print(f"ROC AUC: {round(cat_cv_results['test_roc_auc'].mean(), 4)}")

"""
CatBoost Final Results:
Accuracy: 0.8019
F1 Score: 0.581
ROC AUC: 0.848

"""
##################################
# MODEL COMPARISON
##################################

print("\n" + "="*50)
print("FINAL MODEL COMPARISON")
print("="*50)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'CatBoost'],
    'Accuracy': [
        round(cv_results['test_accuracy'].mean(), 4),
        round(cat_cv_results['test_accuracy'].mean(), 4)
    ],
    'F1 Score': [
        round(cv_results['test_f1'].mean(), 4),
        round(cat_cv_results['test_f1'].mean(), 4)
    ],
    'ROC AUC': [
        round(cv_results['test_roc_auc'].mean(), 4),
        round(cat_cv_results['test_roc_auc'].mean(), 4)
    ]
})

print(comparison_df)

"""
==================================================
FINAL MODEL COMPARISON
==================================================
                 Model  Accuracy  F1 Score  ROC AUC
0  Logistic Regression     0.805     0.591    0.849
1             CatBoost     0.802     0.581    0.848

Logistic Regression aslında bir tık daha iyi performans gösterdi
Ama çok fark olmadığı için her iki modeli kullanalım
"""

##################################
# FEATURE IMPORTANCE (CatBoost için)
##################################
print("\n" + "="*50)
print("CATBOOST FEATURE IMPORTANCE")
print("="*50)
    
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': catboost_final.feature_importances_
}).sort_values('importance', ascending=False)
    
print("Top 10 Önemli Özellikler:")
print(feature_importance.head(10))
    
# Feature importance plot
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
plt.title('CatBoost - Feature Importance (Top 15)')
plt.tight_layout()
plt.show(block=True)

"""
==================================================
CATBOOST FEATURE IMPORTANCE
==================================================
Top 10 Önemli Özellikler:
                        feature  importance
1                        tenure      13.342
40                NEW_Engaged_1      10.638
26            Contract_Two year       8.154
6                  NEW_Increase       7.375
11  InternetService_Fiber optic       7.029
2                MonthlyCharges       4.783
7           NEW_AVG_Service_Fee       4.526
3                  TotalCharges       4.390
5               NEW_AVG_Charges       3.876
53        NEW_FLAG_LongTenure_1       3.685

"""

##################################
# LOGISTIC REGRESSION IMPORTANCE
##################################

print("\n" + "="*50)
print("LOGISTIC REGRESSION FEATURE IMPORTANCE")
print("="*50)

coeffs = lr_final.named_steps['log_reg'].coef_[0]
importance = np.abs(coeffs) / np.abs(coeffs).sum()

lr_importance = pd.DataFrame({
    "feature": X.columns,
    "coefficient": coeffs,
    "importance": importance
}).sort_values("importance", ascending=False)

print("Top 10 Önemli Özellikler (Normalize Importance):")
print(lr_importance.head(10))

# Görselleştirme
plt.figure(figsize=(12, 8))
top_features = lr_importance.head(15)
colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
plt.barh(top_features['feature'], top_features['importance'], color=colors)
plt.xlabel("Normalized Importance")
plt.title("Logistic Regression - Feature Importance (Top 15)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show(block=True)


"""
==================================================
LOGISTIC REGRESSION FEATURE IMPORTANCE
==================================================
Top 10 Önemli Özellikler (Normalize Importance):
                        feature  coefficient  importance
1                        tenure       -0.980       0.164
40                NEW_Engaged_1       -0.453       0.076
26            Contract_Two year       -0.381       0.064
11  InternetService_Fiber optic        0.376       0.063
34     NEW_TENURE_YEAR_5-6 Year        0.368       0.061
6                  NEW_Increase       -0.350       0.058
33     NEW_TENURE_YEAR_4-5 Year        0.237       0.040
24          StreamingMovies_Yes        0.227       0.038
22              StreamingTV_Yes        0.213       0.036
43     NEW_FLAG_ANY_STREAMING_1       -0.207       0.034

"""


"""
Ortak Noktalar: (tenure, kontrat tipi, engagement, fiber optic) 
                - hem Logistic hem de CatBoost bunların kritik olduğunu söylüyor.
Farklı Noktalar: 
Logistic streaming etkisini daha fazla öne çıkarırken,
CatBoost finansal ve türetilmiş değişkenlere (NEW_Increase, NEW_AVG_Service_Fee) daha çok önem veriyor.
"""

##################################
# MODEL KAYDETME
##################################

# Model kaydetme
import joblib

print("\n💾 Modeller kaydediliyor...")
joblib.dump(lr_final, 'logistic_regression_model.pkl')
joblib.dump(catboost_final, 'catboost_model.pkl')
print("✅ Modeller 'logistic_regression_model.pkl' ve 'catboost_model.pkl' olarak kaydedildi")
