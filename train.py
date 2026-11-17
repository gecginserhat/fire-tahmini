"""Fire tahmin modelini egitir, degerlendirir ve TL bazinda tasarruf hesaplar."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---- Is varsayimlari: kendi fabrikana gore degistir ----
HURDA_MALIYETI = 45.0     # TL, bir hatali parcanin maliyeti
MUDAHALE_MALIYETI = 4.0   # TL, alarm verilen cevrimde kontrol/ayar maliyeti
ONLEME_ORANI = 0.7        # alarm dogruysa hatanin onlenebilme orani

NUM = ["eriyik_sicaklik", "kalip_sicaklik", "enjeksiyon_basinc", "tutma_basinc",
       "sogutma_suresi", "hammadde_nem", "ortam_sicaklik"]
CAT = ["makine", "vardiya", "hammadde_parti"]
TARGET = "hatali"

df = pd.read_csv("uretim_verisi.csv", parse_dates=["zaman"]).sort_values("zaman")

# Zamana gore bolme: model gecmisle egitilir, gelecekte test edilir (rastgele bolme yaniltir)
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]
X_tr, y_tr = train[NUM + CAT], train[TARGET]
X_te, y_te = test[NUM + CAT], test[TARGET]


def make(model, scale):
    num = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer([
        ("num", num, NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    return Pipeline([("pre", pre), ("model", model)])


models = {
    "Lojistik regresyon (baz)": make(LogisticRegression(max_iter=1000), True),
    "Gradient boosting": make(
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                       max_leaf_nodes=15, l2_regularization=1.0,
                                       random_state=0), False),
}

results = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    results[name] = {"roc_auc": roc_auc_score(y_te, p),
                     "pr_auc": average_precision_score(y_te, p), "proba": p}
    print(f"{name:28s} ROC-AUC {results[name]['roc_auc']:.3f}  PR-AUC {results[name]['pr_auc']:.3f}")

best_name = max(results, key=lambda k: results[k]["pr_auc"])
best, proba = models[best_name], results[best_name]["proba"]

# ---- Esik secimi: en cok para kazandiran alarm seviyesi ----
ths = np.linspace(0.02, 0.9, 89)
y = y_te.to_numpy()
savings = []
for t in ths:
    alarm = proba >= t
    tp = (alarm & (y == 1)).sum()
    savings.append(tp * ONLEME_ORANI * HURDA_MALIYETI - alarm.sum() * MUDAHALE_MALIYETI)
savings = np.array(savings)
i = savings.argmax()
th = float(ths[i])
alarm = proba >= th
tp = int((alarm & (y == 1)).sum())
fp = int((alarm & (y == 0)).sum())
fn = int((~alarm & (y == 1)).sum())

test_days = (test.zaman.max() - test.zaman.min()).total_seconds() / 86400
yearly = 365 / test_days
metrics = {
    "model": best_name,
    "roc_auc": round(results[best_name]["roc_auc"], 3),
    "pr_auc": round(results[best_name]["pr_auc"], 3),
    "fire_orani_test": round(float(y.mean()), 4),
    "esik": round(th, 3),
    "alarm_orani": round(float(alarm.mean()), 4),
    "yakalanan_hata_orani": round(tp / (tp + fn), 3),
    "alarm_isabet_orani": round(tp / max(tp + fp, 1), 3),
    "test_gun": round(test_days, 1),
    "test_donemi_tasarruf_tl": round(float(savings[i])),
    "yillik_tasarruf_tl": round(float(savings[i] * yearly)),
    "yillik_fire_maliyeti_tl": round(float(y.sum() * HURDA_MALIYETI * yearly)),
    "varsayimlar": {"hurda_maliyeti": HURDA_MALIYETI,
                    "mudahale_maliyeti": MUDAHALE_MALIYETI,
                    "onleme_orani": ONLEME_ORANI},
}
print(json.dumps(metrics, indent=2, ensure_ascii=False))

# ---- Hangi degisken fireyi en cok etkiliyor ----
imp = permutation_importance(best, X_te, y_te, scoring="average_precision",
                             n_repeats=5, random_state=0)
order = imp.importances_mean.argsort()
cols = np.array(NUM + CAT)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].barh(cols[order], imp.importances_mean[order], color="#2a6f97")
ax[0].set_title("Fireyi en çok etkileyen değişkenler")
ax[0].set_xlabel("Önem (PR-AUC düşüşü)")

ax[1].plot(ths, savings * yearly / 1000, color="#2a6f97")
ax[1].axvline(th, color="#c1121f", ls="--", label=f"Seçilen eşik {th:.2f}")
ax[1].axhline(0, color="gray", lw=0.8)
ax[1].set_title("Alarm eşiğine göre yıllık tasarruf")
ax[1].set_xlabel("Risk eşiği"); ax[1].set_ylabel("Bin TL / yıl"); ax[1].legend()

dec = pd.qcut(proba, 10, labels=False, duplicates="drop")
rate = pd.Series(y).groupby(dec).mean() * 100
ax[2].bar(rate.index + 1, rate.values, color="#2a6f97")
ax[2].set_title("Risk grubuna göre gerçek fire oranı")
ax[2].set_xlabel("Risk grubu (1 = en düşük, 10 = en yüksek)"); ax[2].set_ylabel("Fire %")
plt.tight_layout()
plt.savefig("images/sentetik_veri_sonuc.png", dpi=130)

joblib.dump({"model": best, "esik": th, "kolonlar": NUM + CAT}, "model.joblib")
json.dump(metrics, open("metrics.json", "w"), indent=2, ensure_ascii=False)
