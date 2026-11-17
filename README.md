# Fire Tahmini: Plastik Enjeksiyonda Hatalı Parçayı Önceden Yakalamak

*Predicting scrap in plastic injection molding from machine process data. English summary below.*

Plastik enjeksiyon makinesinin her çevrimde kaydettiği proses verisinden
(basınç, sıcaklık, süre) o çevrimde çıkan parçanın hatalı olma riskini tahmin eden
bir makine öğrenmesi modeli. Amaç, hatalı parçayı kalite kontrolde değil,
oluşurken yakalamak ve kontrolü en riskli çevrimlere yoğunlaştırmak.

![Gerçek veri sonuçları](images/gercek_veri_sonuc.png)

## Sonuç (gerçek fabrika verisi)

Otomotiv plastik parçaları basan 650 tonluk bir makineden **7.992 çevrim**,
bunların **71'i hatalı (%0,9)**, 26 sensör değişkeni.

| Kontrol edilen çevrim | Yakalanan hatalı parça | Rastgele kontrolde |
|---|---|---|
| En riskli %2 | **%51** (71'in 36'sı) | %2 |
| En riskli %5 | **%61** (71'in 43'ü) | %5 |

- Model, rastgele kontrole göre yaklaşık **12 kat daha isabetli**.
- Zamana göre testte (ilk %70 ile eğitim, son %30'da deneme) 9 hatalı parçanın
  6'sı en riskli %5'in içinde çıktı.
- Fireyi en çok etkileyen değişkenler: ortalama geri basınç, kovan sıcaklıkları
  (bölge 3 ve 5), çevrim süresi.

## Sınırlar

- Yalnızca 71 hatalı örnek var; zaman testindeki 9 hata kesin sonuç için az.
  Sonuç umut verici ama bir fabrikada pilotla doğrulanmalı.
- Çapraz doğrulama rakamları biraz iyimser olabilir, çünkü art arda gelen
  çevrimler birbirine çok benziyor.
- Veride hurda maliyeti yok, bu yüzden gerçek veride TL tasarruf hesabı yapılmadı.
  TL hesabı sentetik veri örneğinde gösteriliyor.

## Çalıştırma

```bash
pip install -r requirements.txt

# Gerçek veri
python download_data.py      # veriyi veri/ klasörüne indirir
python train_real.py         # sonuçları yazar, images/gercek_veri_sonuc.png üretir

# Sentetik veri + TL tasarruf hesabı
python generate_data.py      # uretim_verisi.csv
python train.py              # model.joblib, metrics.json, images/sentetik_veri_sonuc.png
python predict.py uretim_verisi.csv tahminler.csv
```

Mac/Linux'ta `python` yerine `python3` kullanman gerekebilir.

## Dosyalar

| Dosya | Ne yapar |
|---|---|
| `download_data.py` | Gerçek veriyi indirir |
| `train_real.py` | Gerçek veride modeli eğitir ve değerlendirir |
| `generate_data.py` | Proses kaymaları içeren sentetik enjeksiyon verisi üretir |
| `train.py` | Sentetik veride modeli eğitir, en kârlı alarm eşiğini seçer, yıllık TL tasarrufu hesaplar |
| `predict.py` | Yeni çevrimlere risk skoru ve alarm verir |

## Yöntem

- **Model:** Gradient boosting (scikit-learn `HistGradientBoostingClassifier`),
  lojistik regresyon ile karşılaştırıldı.
- **Değerlendirme:** Zamana göre bölme (sahadaki kullanıma en yakın senaryo) ve
  5 katlı tabakalı çapraz doğrulama. Hatalı oranı çok düşük olduğu için doğruluk
  (accuracy) yerine PR-AUC ve "en riskli %k'da yakalama oranı" kullanıldı.
- **Değişken önemi:** Permutation importance.
- **Alarm eşiği (sentetik veri):** Hurda maliyeti, müdahale maliyeti ve önleme
  oranı varsayımlarıyla en yüksek net tasarrufu veren eşik seçildi.

## Sentetik veri örneği

![Sentetik veri sonuçları](images/sentetik_veri_sonuc.png)

`generate_data.py`, ısıtıcı/ayar kaynaklı sıcaklık ve basınç düşüşleri, nemli
hammadde partisi ve kısa soğutma gibi gerçekçi fire sebepleri içeren 20.000
çevrimlik veri üretir. Varsayımlarla (hurda 45 TL, müdahale 4 TL, önleme %70)
model hatalıların %73'ünü yakalıyor ve yılda yaklaşık 160 bin TL net tasarruf
gösteriyor. **Bu rakamlar sentetik veriye aittir**; yöntemin nasıl paraya
çevrildiğini göstermek içindir.

## Kendi fabrika verinle kullanmak

1. Çevrim bazında proses parametreleri ve kalite sonucunu (hatalı/sağlam) tek CSV'de birleştir.
2. `train.py` içindeki `NUM`, `CAT`, `TARGET` listelerini kendi kolonlarınla değiştir.
3. Maliyet varsayımlarını firmayla birlikte belirle.
4. Hata oluştuktan sonra ölçülen bir değeri (ör. son kontrol ölçüsü) modele verme; sızıntı olur.

## Veri kaynakları

- Gerçek veri: Korea AI Manufacturing Platform (KAMP), enjeksiyon kalıplama veri seti.
  İndirme için [johnwslee/injection_molding_analysis](https://github.com/johnwslee/injection_molding_analysis)
  reposundaki kopyası kullanılır. Veri bu repoda yer almaz.
- Ek açık veri: iGuzzini yol lensi verisi, 1.451 parça, 4 kalite sınıfı:
  [airtlab/machine-learning-for-quality-prediction-in-plastic-injection-molding](https://github.com/airtlab/machine-learning-for-quality-prediction-in-plastic-injection-molding)
  ([makale](https://doi.org/10.3390/info13060272))

---

## English summary

A machine learning model that predicts the scrap risk of each injection molding
cycle from machine process data (pressures, temperatures, times), so inspection
can focus on the riskiest cycles.

On real factory data (KAMP dataset, 7,992 cycles, 71 defects / 0.9%), a gradient
boosting model catches **51% of defects in the top 2% riskiest cycles** and
**61% in the top 5%**, roughly 12x better than random inspection. With only 71
defects the results are promising but need a pilot to confirm. A synthetic
example (`train.py`) shows how to choose an alarm threshold and convert the
model into yearly cost savings.

Run: `pip install -r requirements.txt`, then `python download_data.py` and
`python train_real.py`.
