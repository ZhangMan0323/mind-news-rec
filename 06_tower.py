# 训练脚本：
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score

torch.manual_seed(0); np.random.seed(0)
C = ["imp_id", "user_id", "time", "history", "impressions"]
B = r"D:\MiND\MIND_small"

dv = pd.read_csv(f"{B}/MINDsmall_dev/behaviors.tsv", sep="\t", names=C)
cut = int(len(dv) * 0.7); tr, te = dv.iloc[:cut], dv.iloc[cut:]

items = set()
for c in dv.impressions:
    for it in c.split(): items.add(it[:-2])
for h in dv.history.fillna(""): items.update(h.split())
items = sorted(items)
n2i = {n: i + 1 for i, n in enumerate(items)}
u2i = {u: i for i, u in enumerate(sorted(set(dv.user_id)))}

MAXLEN = 20
seq = np.zeros((len(u2i), MAXLEN), dtype=np.int64)
for u, h in zip(dv.user_id, dv.history.fillna("")):
    ids = [n2i[n] for n in h.split() if n in n2i][-MAXLEN:]
    if ids: seq[u2i[u], -len(ids):] = ids

pos, neg = [], []
for u, imps in zip(tr.user_id, tr.impressions):
    P, N = [], []
    for it in imps.split():
        n, l = it.rsplit("-", 1)
        (P if l == "1" else N).append(n2i[n])
    if not N: continue
    for p in P:
        pos.append((u2i[u], p)); neg.append((u2i[u], np.random.choice(N)))

news = pd.read_csv(f"{B}/MINDsmall_dev/news.tsv", sep="\t", header=None,
                   usecols=[0, 3], names=["nid", "title"], on_bad_lines="skip")
news["title"] = news["title"].fillna("")
tmap = dict(zip(news.nid, news.title))
tf = TfidfVectorizer(max_features=20000, stop_words="english")
X = tf.fit_transform([tmap.get(n, "") for n in items])
E = np.zeros((len(n2i) + 1, 64), dtype=np.float32)
E[1:] = TruncatedSVD(64, random_state=0).fit_transform(X)
E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8

class TwoTower(nn.Module):
    def __init__(self, n, dim=64):
        super().__init__()
        self.emb = nn.Embedding(n, dim, padding_idx=0)
        with torch.no_grad():
            self.emb.weight.copy_(torch.tensor(E))
        self.lin = nn.Linear(dim, dim)
    def uv(self, s):
        m = (s != 0).float().unsqueeze(-1)
        v = (self.emb(s) * m).sum(1) / m.sum(1).clamp(min=1)
        return torch.relu(self.lin(v))
    def iv(self, i):
        return torch.relu(self.lin(self.emb(i)))
    def score(self, u, i):
        return (u * i).sum(-1)

model = TwoTower(len(n2i) + 1)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
seq_t = torch.tensor(seq); P = np.array(pos); N = np.array(neg)
bs = 4096
for ep in range(5):
    perm = np.random.permutation(len(P)); tot = 0.0
    for k in range(0, len(P), bs):
        b = perm[k:k+bs]
        u = model.uv(seq_t[P[b, 0]])
        lp = model.score(u, model.iv(torch.tensor(P[b, 1])))
        ln = model.score(u, model.iv(torch.tensor(N[b, 1])))
        loss = -nn.functional.logsigmoid(lp - ln).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(b)
    print(f"epoch {ep+1}: loss {tot/len(P):.4f}")

us, its, ys, sizes = [], [], [], []
for u, imps in zip(te.user_id, te.impressions):
    parts = [it.rsplit("-", 1) for it in imps.split()]
    us += [u2i[u]] * len(parts)
    its += [n2i[n] for n, _ in parts]
    ys += [int(l) for _, l in parts]
    sizes.append(len(parts))
U, I = torch.tensor(us), torch.tensor(its)
with torch.no_grad():
    out = []
    for k in range(0, len(U), 100000):
        u = model.uv(seq_t[U[k:k+100000]])
        out.append(model.score(u, model.iv(I[k:k+100000])).numpy())
    sc = np.concatenate(out)
y = np.array(ys)
print("AUC :", round(roc_auc_score(y, sc), 4))
idx, vals, wts = 0, [], []
for n in sizes:
    yy, pp = y[idx:idx+n], sc[idx:idx+n]
    if 0 < yy.sum() < n:
        vals.append(roc_auc_score(yy, pp)); wts.append(yy.sum())
    idx += n
print("GAUC:", round(np.average(vals, weights=wts), 4))

# 让双塔把向量存下来：
np.save(r"D:\MiND\uv.npy", model.uv(seq_t).detach().numpy())
np.save(r"D:\MiND\iv.npy", model.iv(torch.arange(len(n2i) + 1)).detach().numpy())