import pandas as pd

df = pd.read_csv("data.csv")

print("printing info: ")
print(df.info())
print("printing head: " + df.head(3).to_string())