import pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score

cols = ["imp_id", "user_id", "time", "history", "impressions"]
B = r"D:\MiND\MIND_small"
dv = pd.read_csv(f"{B}/MINDsmall_dev/behaviors.tsv", sep="\t", names=cols)

cut = int(len(dv) * 0.7)
train_part, test_part = dv.iloc[:cut], dv.iloc[cut:]

pop = Counter()
for c in train_part["impressions"]:
    for it in c.split():
        if it.endswith("-1"):
            pop[it[:-2]] += 1

s = test_part["impressions"].str.split().explode()
label = s.str[-1].astype(int)
score = s.str[:-2].map(pop).fillna(0)

print("候选条数    :", len(s))
print("能查到热度比例:", round((score > 0).mean() * 100, 1), "%")
print("AUC         :", round(roc_auc_score(label, score), 4))