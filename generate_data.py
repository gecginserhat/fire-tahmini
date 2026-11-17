"""Plastik enjeksiyon icin ornek (sentetik) uretim verisi uretir.

Gercek fabrika verin oldugunda bu dosyaya gerek kalmaz:
ayni kolon adlariyla bir CSV hazirlaman yeterli.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 20_000

ts = pd.date_range("2026-01-05 08:00", periods=N, freq="3min")
hour = ts.hour
vardiya = np.where((hour >= 8) & (hour < 16), 1, np.where(hour >= 16, 2, 3))
makine = rng.integers(1, 5, N)

# Hammadde partisi yaklasik 500 cevrimde bir degisir
n_lot = N // 500 + 1
lot_seq = rng.choice(list("ABCDE"), size=n_lot, p=[0.25, 0.25, 0.2, 0.15, 0.15])
parti = np.repeat(lot_seq, 500)[:N]

eriyik_sicaklik = rng.normal(240, 6, N) + np.where(makine == 3, -3, 0)
# Proses kaymasi: rezistans/ayar sorunu olan donemlerde sicaklik ve basinc duser
for start in rng.choice(N - 200, size=25, replace=False):
    sl = slice(start, start + int(rng.integers(80, 200)))
    eriyik_sicaklik[sl] -= rng.uniform(6, 12)
kalip_sicaklik = rng.normal(55, 5, N)
enjeksiyon_basinc = rng.normal(1000, 70, N)
for start in rng.choice(N - 200, size=20, replace=False):
    enjeksiyon_basinc[start:start + int(rng.integers(60, 150))] -= rng.uniform(60, 120)
tutma_basinc = rng.normal(600, 50, N)
sogutma_suresi = rng.normal(14, 2, N).clip(8, 20)
nem = (0.05 + rng.gamma(2.0, 0.015, N) + np.where(parti == "D", 0.05, 0)).clip(0.02, 0.3)
ortam_sicaklik = 22 + 6 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1.5, N)

relu = lambda x: np.maximum(x, 0)

logit = (
    -4.7
    + 1.2 * relu((235 - eriyik_sicaklik) / 5) * relu((1000 - enjeksiyon_basinc) / 70)  # eksik baski
    + 24 * relu(nem - 0.10)                                                             # nem izi
    + 0.8 * relu(12 - sogutma_suresi) * (1 + 0.5 * relu((kalip_sicaklik - 55) / 5))     # carpilma
    + 0.6 * (parti == "D")
    + 0.25 * (vardiya == 3)
    + 0.3 * (makine == 3)
    + 0.02 * (ortam_sicaklik - 22)
)
p = 1 / (1 + np.exp(-logit))
hatali = (rng.random(N) < p).astype(int)

df = pd.DataFrame({
    "zaman": ts,
    "makine": makine,
    "vardiya": vardiya,
    "hammadde_parti": parti,
    "eriyik_sicaklik": eriyik_sicaklik.round(1),
    "kalip_sicaklik": kalip_sicaklik.round(1),
    "enjeksiyon_basinc": enjeksiyon_basinc.round(0),
    "tutma_basinc": tutma_basinc.round(0),
    "sogutma_suresi": sogutma_suresi.round(2),
    "hammadde_nem": nem.round(3),
    "ortam_sicaklik": ortam_sicaklik.round(1),
    "hatali": hatali,
})
df.to_csv("uretim_verisi.csv", index=False)
print(f"{len(df)} satir yazildi. Fire orani: %{100 * df.hatali.mean():.1f}")
