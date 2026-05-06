import pandas as pd

df = pd.read_csv("../08-cleaning-empty-cells/data.csv")

dup_mask = df.duplicated()

print("duplicate mask:\n" + dup_mask.to_string())

offending_rows = df[dup_mask]
print("\noffending rows:\n" + offending_rows.to_string())

print("\noffending row indexes:", offending_rows.index.tolist())