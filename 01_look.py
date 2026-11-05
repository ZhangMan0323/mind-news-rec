import pandas as pd

path = r"D:\MIND\MIND_small\MINDsmall_train\behaviors.tsv"
cols = ["imp_id", "user_id", "time", "history", "impressions"]
df = pd.read_csv(path, sep="\t", names=cols)

print("曝光总数  :", len(df))
print("用户数    :", df.user_id.nunique())
print("候选总数  :", df.impressions.str.split().str.len().sum())
print("点击总数  :", df.impressions.str.count("-1").sum())