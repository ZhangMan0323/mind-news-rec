import numpy as np, pandas as pd, lightgbm as lgb
from collections import Counter
from sklearn.metrics import roc_auc_score

# pop：这条新闻有多热门（取 log）    cat：新闻属于哪个类别      in_hist：这条新闻在不在用户历史里
# hist_len：用户历史有多长          cat_pref：用户历史里这个类别占多少

C = ["imp_id", "user_id", "time", "history", "impressions"]
B = r"D:\MiND\MIND_small"

news = pd.read_csv(f"{B}/MINDsmall_dev/news.tsv", sep="\t", header=None)
cat = dict(zip(news[0], news[1]))
cid = {c: i for i, c in enumerate(news[1].unique())}

dv = pd.read_csv(f"{B}/MINDsmall_dev/behaviors.tsv", sep="\t", names=C)
cut = int(len(dv) * 0.7)
tr, te = dv.iloc[:cut], dv.iloc[cut:]

pop = Counter()
for c in tr.impressions:
    for it in c.split():
        if it.endswith("-1"):
            pop[it[:-2]] += 1

def build(df):
    n = sum(len(x.split()) for x in df.impressions)
    out = np.zeros((n, 6), dtype=np.float32)
    i = 0
    for h, imps in zip(df.history.fillna(""), df.impressions):
        hs = set(h.split())
        pc = Counter(cat.get(x) for x in hs)
        hl = max(len(hs), 1)
        for it in imps.split():
            nid, lab = it.rsplit("-", 1)
            nc = cat.get(nid)
            out[i] = (np.log1p(pop.get(nid, 0)), cid.get(nc, -1),
                      1 if nid in hs else 0, hl, pc.get(nc, 0) / hl, int(lab))
            i += 1
    return out

A, T = build(tr), build(te)
m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31)
m.fit(A[:, :5], A[:, 5])
print("训练样本:", len(A))
print("AUC     :", round(roc_auc_score(T[:, 5], m.predict_proba(T[:, :5])[:, 1]), 4))

for f, g in sorted(zip(["pop","cat","in_hist","hist_len","cat_pref"],
                       m.feature_importances_), key=lambda x: -x[1]):
    print(f"{f:10s} {g}")

st = np.array([len(x.split()) for x in te.impressions])
pred = m.predict_proba(T[:, :5])[:, 1]
idx, vals, wts = 0, [], []
for n in st:
    y = T[idx:idx+n, 5]
    p = pred[idx:idx+n]
    if 0 < y.sum() < n:
        vals.append(roc_auc_score(y, p))
        wts.append(y.sum())
    idx += n
print("GAUC    :", round(np.average(vals, weights=wts), 4))