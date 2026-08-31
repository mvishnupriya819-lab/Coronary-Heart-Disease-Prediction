import pandas as pd

df = pd.read_csv("framingham.csv")

print("DATASET SIZE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nFIRST 5 ROWS:")
print(df.head())