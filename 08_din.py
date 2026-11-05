import numpy as np, pandas as pd, torch, torch.nn as nn
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
seq_t = torch.tensor(seq)

def flat(df):
    N = sum(len(x.split()) for x in df.impressions)
    uu = np.empty(N, np.int64); ii = np.empty(N, np.int64); yy = np.empty(N, np.float32)
    k = 0
    for u, imps in zip(df.user_id, df.impressions):
        ui = u2i[u]
        for it in imps.split():
            n, l = it.rsplit("-", 1)
            uu[k] = ui; ii[k] = n2i[n]; yy[k] = int(l); k += 1
    return torch.tensor(uu), torch.tensor(ii), torch.tensor(yy)

U, I, Y = flat(tr)
Ut, It, Yt = flat(te)

E = np.load(r"D:\MiND\iv.npy"); E[0] = 0

class DIN(nn.Module):
    def __init__(self, n, dim=64):
        super().__init__()
        self.emb = nn.Embedding(n, dim, padding_idx=0)
        with torch.no_grad(): self.emb.weight.copy_(torch.tensor(E))
        self.att = nn.Sequential(nn.Linear(dim*4, 64), nn.Sigmoid(), nn.Linear(64, 1))
        self.mlp = nn.Sequential(nn.Linear(dim*3, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, s, c):
        h = self.emb(s)
        ce = self.emb(c).unsqueeze(1).expand_as(h)
        a = self.att(torch.cat([h, ce, h - ce, h * ce], -1)).squeeze(-1)
        a = a.masked_fill(s == 0, -1e9)
        u = (h * torch.softmax(a, 1).unsqueeze(-1)).sum(1)
        e = self.emb(c)
        return self.mlp(torch.cat([u, e, u * e], -1)).squeeze(-1)

model = DIN(len(n2i) + 1)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.BCEWithLogitsLoss()
bs = 2048

for ep in range(3):
    perm = torch.randperm(len(U)); tot = 0.0
    for k in range(0, len(U), bs):
        b = perm[k:k+bs]
        loss = crit(model(seq_t[U[b]], I[b]), Y[b])
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(b)
    print(f"epoch {ep+1}: loss {tot/len(U):.4f}", flush=True)

with torch.no_grad():
    out = []
    for k in range(0, len(Ut), 50000):
        out.append(model(seq_t[Ut[k:k+50000]], It[k:k+50000]).numpy())
    sc = np.concatenate(out)

st = np.array([len(x.split()) for x in te.impressions])
idx, vals, wts = 0, [], []
for n in st:
    y, p = Yt[idx:idx+n].numpy(), sc[idx:idx+n]
    if 0 < y.sum() < n:
        vals.append(roc_auc_score(y, p)); wts.append(y.sum())
    idx += n

print("AUC :", round(roc_auc_score(Yt.numpy(), sc), 4))
print("GAUC:", round(np.average(vals, weights=wts), 4))

with torch.no_grad():
    ptr = []
    for k in range(0, len(U), 50000):
        ptr.append(model(seq_t[U[k:k+50000]], I[k:k+50000]).numpy())
np.save(r"D:\MiND\din_tr.npy", np.concatenate(ptr))
np.save(r"D:\MiND\din_te.npy", sc)
torch.save(model.state_dict(), r"D:\MiND\din.pt")