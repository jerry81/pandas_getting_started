import pandas as pd

df = pd.read_csv('data.csv')

x = df["Calories"].mode()[0]

print("df before fill \n" + df.to_string())

df.fillna({"Calories": x}, inplace=True)

print("df after fill calories with mode  \n" + df.to_string())