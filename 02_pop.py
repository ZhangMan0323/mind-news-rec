import pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score

path = r"D:\MiND\MIND_small\MINDsmall_train\behaviors.tsv"
cols = ["imp_id", "user_id", "time", "history", "impressions"]
df = pd.read_csv(path, sep="\t", names=cols).head(5000)

cnt = Counter()
for h in df["history"].fillna(""):
    cnt.update(h.split())

s = df["impressions"].str.split().explode()
label = s.str[-1].astype(int)
news = s.str[:-2]
score = news.map(cnt).fillna(0)

print("候选条数:", len(s))
print("AUC     :", round(roc_auc_score(label, score), 4))