"""Yeni cevrimler icin fire riski hesaplar.

Kullanim: python predict.py yeni_veri.csv tahminler.csv
"""
import sys

import joblib
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "uretim_verisi.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "tahminler.csv"

pack = joblib.load("model.joblib")
df = pd.read_csv(src)
df["fire_riski"] = pack["model"].predict_proba(df[pack["kolonlar"]])[:, 1].round(3)
df["alarm"] = (df["fire_riski"] >= pack["esik"]).astype(int)
df.to_csv(dst, index=False)
print(f"{len(df)} cevrim skorlandi, {int(df.alarm.sum())} alarm. Cikti: {dst}")
