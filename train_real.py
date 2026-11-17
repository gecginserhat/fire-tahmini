"""Gercek fabrika verisiyle fire tahmini.

Veri: Kore AI Manufacturing Platform (KAMP) enjeksiyon kaliplama veri seti,
github.com/johnwslee/injection_molding_analysis uzerinden.
7.996 cevrim, 71 hatali (%0,9). Otomotiv plastik parcalari, 650 tonluk makine.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("veri/kamp_labeled_data.csv", parse_dates=["TimeStamp"])
df = df[df.PART_NAME.str.startswith(("CN7", "RG3"))].sort_values("TimeStamp").reset_index(drop=True)
df["hatali"] = (df.PassOrFail == "N").astype(int)
df["parca_RG3"] = df.PART_NAME.str.startswith("RG3").astype(int)

num = df.select_dtypes("number").drop(columns=["hatali", "PART_FACT_SERIAL"])
num = num.loc[:, num.nunique() > 1]          # hep ayni deger olan (bos) sensorleri at
FEATS = list(num.columns)
X, y = df[FEATS], df["hatali"]
print(f"{len(df)} cevrim, {y.sum()} hatali (%{100 * y.mean():.2f}), {len(FEATS)} degisken")

models = {
    "Lojistik regresyon (baz)": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Gradient boosting": HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=10,
        l2_regularization=1.0, class_weight="balanced", random_state=0),
}

out = {"satir": int(len(df)), "hatali": int(y.sum()), "degisken": len(FEATS), "modeller": {}}

# 1) Zamana gore test: ilk %70 ile egit, son %30'da dene (sahadaki gercek kullanim)
cut = int(len(df) * 0.7)
Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
print(f"Zaman testi: egitimde {ytr.sum()} hatali, testte {yte.sum()} hatali")

# 2) 5 katli capraz dogrulama: az hatali ornek oldugu icin daha kararli bir olcum
cv = StratifiedKFold(5, shuffle=True, random_state=0)

probs = {}
for name, m in models.items():
    p_time = m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    p_cv = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
    probs[name] = (p_time, p_cv)
    r = {
        "zaman_roc_auc": round(roc_auc_score(yte, p_time), 3),
        "zaman_pr_auc": round(average_precision_score(yte, p_time), 3),
        "cv_roc_auc": round(roc_auc_score(y, p_cv), 3),
        "cv_pr_auc": round(average_precision_score(y, p_cv), 3),
    }
    out["modeller"][name] = r
    print(f"{name:26s} {r}")

best = max(out["modeller"], key=lambda k: out["modeller"][k]["cv_pr_auc"])
p_time, p_cv = probs[best]
out["secilen_model"] = best
out["rastgele_tahmin_pr_auc"] = round(float(y.mean()), 4)


def top_k(p, yy, frac):
    k = max(int(len(p) * frac), 1)
    idx = np.argsort(-p)[:k]
    hit = int(np.asarray(yy)[idx].sum())
    return {"kontrol_edilen": k, "yakalanan": hit, "toplam_hatali": int(np.sum(yy)),
            "yakalama_orani": round(hit / max(int(np.sum(yy)), 1), 3),
            "isabet_orani": round(hit / k, 3)}


out["en_riskli_yuzde_2_cv"] = top_k(p_cv, y, 0.02)
out["en_riskli_yuzde_5_cv"] = top_k(p_cv, y, 0.05)
out["en_riskli_yuzde_5_zaman"] = top_k(p_time, yte, 0.05)
print(json.dumps(out, indent=2, ensure_ascii=False))

# Grafikler
m = models[best].fit(Xtr, ytr)
imp = permutation_importance(m, Xte, yte, scoring="average_precision", n_repeats=10, random_state=0)
order = imp.importances_mean.argsort()[-10:]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].barh(np.array(FEATS)[order], imp.importances_mean[order], color="#2a6f97")
ax[0].set_title("Gerçek veride fireyi en çok etkileyen 10 değişken")
ax[0].set_xlabel("Önem (PR-AUC düşüşü)")

fr = np.linspace(0.005, 0.3, 60)
ax[1].plot(fr * 100, [top_k(p_cv, y, f)["yakalama_orani"] * 100 for f in fr],
           color="#2a6f97", label="Model (çapraz doğrulama)")
ax[1].plot(fr * 100, [top_k(p_time, yte, f)["yakalama_orani"] * 100 for f in fr],
           color="#e09f3e", label="Model (zaman testi)")
ax[1].plot(fr * 100, fr * 100, color="gray", ls="--", label="Rastgele kontrol")
ax[1].set_title("En riskli çevrimler kontrol edilirse yakalanan hata")
ax[1].set_xlabel("Kontrol edilen çevrim %"); ax[1].set_ylabel("Yakalanan hatalı parça %")
ax[1].legend()
plt.tight_layout()
plt.savefig("images/gercek_veri_sonuc.png", dpi=130)
json.dump(out, open("metrics_gercek_veri.json", "w"), indent=2, ensure_ascii=False)
