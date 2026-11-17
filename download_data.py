"""Gercek fabrika verisini (KAMP enjeksiyon kaliplama) indirir.

Veri bu repoda yer almaz; asil kaynagi Kore AI Manufacturing Platform (KAMP)'tir.
Burada John W.S. Lee'nin acik reposundaki kopyasi kullanilir:
https://github.com/johnwslee/injection_molding_analysis
"""
import os
import urllib.request

URL = ("https://raw.githubusercontent.com/johnwslee/injection_molding_analysis/"
       "main/data/raw/labeled_data.csv")
DST = os.path.join("veri", "kamp_labeled_data.csv")

os.makedirs("veri", exist_ok=True)
if os.path.exists(DST):
    print(f"Zaten var: {DST}")
else:
    print("Indiriliyor...")
    urllib.request.urlretrieve(URL, DST)
    print(f"Kaydedildi: {DST} ({os.path.getsize(DST) / 1e6:.1f} MB)")
