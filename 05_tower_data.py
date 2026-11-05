# 构造id映射和训练样本：
# 双塔是“把用户历史平均一下”，DIN 是“看着候选新闻，回头在历史里找相关的几条，按相关程度加权”

import numpy as np, pandas as pd

C = ["imp_id", "user_id", "time", "history", "impressions"]
B = r"D:\MiND\MIND_small"
dv = pd.read_csv(f"{B}/MINDsmall_dev/behaviors.tsv", sep="\t", names=C)
cut = int(len(dv) * 0.7)
tr, te = dv.iloc[:cut], dv.iloc[cut:]

items = set()
for c in dv.impressions:
    for it in c.split():
        items.add(it[:-2])
for h in dv.history.fillna(""):
    items.update(h.split())
n2i = {n: i + 1 for i, n in enumerate(sorted(items))}   # 0 留给 padding
users = sorted(set(dv.user_id))
u2i = {u: i for i, u in enumerate(users)}

MAXLEN = 20
seq = np.zeros((len(users), MAXLEN), dtype=np.int64)
for u, h in zip(dv.user_id, dv.history.fillna("")):
    ids = [n2i[n] for n in h.split() if n in n2i][-MAXLEN:]
    if ids:
        seq[u2i[u], -len(ids):] = ids

pos, neg = [], []
for u, imps in zip(tr.user_id, tr.impressions):
    P, N = [], []
    for it in imps.split():
        n, l = it.rsplit("-", 1)
        (P if l == "1" else N).append(n2i[n])
    for p in P:
        pos.append((u2i[u], p))
        neg.append((u2i[u], np.random.choice(N)))

print("物品数:", len(n2i) - 1)
print("用户数:", len(users))
print("正样本对:", len(pos))
print("历史非空用户:", int((seq.sum(1) > 0).sum()))
print("样例:", pos[0], neg[0], seq[pos[0][0]][-5:])