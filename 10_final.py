import numpy as np, pandas as pd, lightgbm as lgb, pickle
from collections import Counter
from sklearn.metrics import roc_auc_score

C = ["imp_id", "user_id", "time", "history", "impressions"]
B = r"D:\MiND\MIND_small"
UV = np.load(r"D:\MiND\uv.npy")
IV = np.load(r"D:\MiND\iv.npy")

news = pd.read_csv(f"{B}/MINDsmall_dev/news.tsv", sep="\t", header=None,
                   usecols=[0, 1, 2, 3], names=["nid", "cat", "subcat", "title"],
                   on_bad_lines="skip")
news["title"] = news["title"].fillna("").str.lower()
cat = dict(zip(news.nid, news.cat))
sub = dict(zip(news.nid, news.subcat))
cid = {c: i for i, c in enumerate(news.cat.unique())}
sid = {s: i for i, s in enumerate(news.subcat.unique())}

def words(t):
    return [w for w in t.replace("|", " ").split() if len(w) > 2]
tw = {n: set(words(t)) for n, t in zip(news.nid, news.title)}

dv = pd.read_csv(f"{B}/MINDsmall_dev/behaviors.tsv", sep="\t", names=C)
cut = int(len(dv) * 0.7)
tr, te = dv.iloc[:cut], dv.iloc[cut:]

items = set()
for c in dv.impressions:
    for it in c.split(): items.add(it[:-2])
for h in dv.history.fillna(""): items.update(h.split())
items = sorted(items)
n2i = {n: i + 1 for i, n in enumerate(items)}
u2i = {u: i for i, u in enumerate(sorted(set(dv.user_id)))}

pop, popr = Counter(), Counter()
for c in tr.impressions:
    for it in c.split():
        if it.endswith("-1"): pop[it[:-2]] += 1
for c in tr.impressions.iloc[int(len(tr) * 0.7):]:
    for it in c.split():
        if it.endswith("-1"): popr[it[:-2]] += 1

prof = {}
for u, h in zip(dv.user_id, dv.history.fillna("")):
    if u in prof: continue
    ws, sc = set(), Counter()
    for n in h.split():
        sc[sub.get(n)] += 1
        ws |= tw.get(n, set())
    prof[u] = (ws, sc)

def build(df):
    n = sum(len(x.split()) for x in df.impressions)
    out = np.zeros((n, 11), dtype=np.float32)
    i = 0
    for u, h, imps in zip(df.user_id, df.history.fillna(""), df.impressions):
        hs = set(h.split()); ws, sc = prof[u]
        hl = max(len(hs), 1)
        pc = Counter(cat.get(x) for x in hs)
        uv = UV[u2i[u]]
        for it in imps.split():
            nid, lab = it.rsplit("-", 1)
            nc, ns = cat.get(nid), sub.get(nid)
            t = tw.get(nid, set())
            tm = len(t & ws) / max(len(t), 1)
            ts = float(uv @ IV[n2i[nid]])
            out[i] = (np.log1p(pop.get(nid, 0)), np.log1p(popr.get(nid, 0)),
                      cid.get(nc, -1), sid.get(ns, -1), 1 if nid in hs else 0,
                      hl, pc.get(nc, 0) / hl, np.log1p(sc.get(ns, 0)), tm,
                      ts, int(lab))
            i += 1
    return out

F = ["pop", "pop_recent", "cat", "sub", "in_hist", "hist_len",
     "cat_pref", "sub_cnt", "title_match", "tower"]
A, T = build(tr), build(te)
m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31)
m.fit(A[:, :10], A[:, 10])

pred = m.predict_proba(T[:, :10])[:, 1]
st = np.array([len(x.split()) for x in te.impressions])
idx, vals, wts = 0, [], []
for n in st:
    y, p = T[idx:idx+n, 10], pred[idx:idx+n]
    if 0 < y.sum() < n:
        vals.append(roc_auc_score(y, p)); wts.append(y.sum())
    idx += n

print("AUC :", round(roc_auc_score(T[:, 10], pred), 4))
print("GAUC:", round(np.average(vals, weights=wts), 4))
for f, g in sorted(zip(F, m.feature_importances_), key=lambda x: -x[1]):
    print(f"{f:12s} {g}")

m.booster_.save_model(r"D:\MiND\gbdt_final.txt")
np.save(r"D:\MiND\pred_te.npy", pred)
np.save(r"D:\MiND\y_te.npy", T[:, 10])

recs, idx = {}, 0
for u, imps in zip(te.user_id, te.impressions):
    n = len(imps.split())
    cand = [x.rsplit("-", 1)[0] for x in imps.split()]
    sc = pred[idx:idx+n]
    top = np.argsort(-sc)[:10]
    recs[u] = [(cand[j], float(sc[j])) for j in top]
    idx += n

pickle.dump(recs, open(r"D:\MiND\recs.pkl", "wb"))
print("推荐用户数:", len(recs))
print("示例用户:", list(recs)[0])