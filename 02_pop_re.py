import pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score

cols = ["imp_id", "user_id", "time", "history", "impressions"]
B = r"D:\MiND\MIND_small"

# 1. 用训练集的“点击记录”统计热门度
pop = Counter()
tr = pd.read_csv(f"{B}/MINDsmall_train/behaviors.tsv", sep="\t", names=cols)
for c in tr["impressions"]:
    for it in c.split():
        if it.endswith("-1"):
            pop[it[:-2]] += 1

# 2. 在验证集上评估（前 5000 条够快）
dv = pd.read_csv(f"{B}/MINDsmall_dev/behaviors.tsv", sep="\t", names=cols).head(5000)
s = dv["impressions"].str.split().explode()
label = s.str[-1].astype(int)
score = s.str[:-2].map(pop).fillna(0)

print("候选条数:", len(s))
print("AUC     :", round(roc_auc_score(label, score), 4))