import pandas as pd

df = pd.read_csv("../08-cleaning-empty-cells/data.csv")

print("before dropping\n" + df.to_string());

for x in df.index:
    if df.loc[x,"Duration"] > 120:
        df.drop(x, inplace=True)

print("after dropping\n" + df.to_string());