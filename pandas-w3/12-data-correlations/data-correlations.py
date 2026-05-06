import pandas as pd

df = pd.read_csv("data.json")

c = df.corr()

print(c.to_string())