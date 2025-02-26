import torch
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


# df = pd.read_csv('./data/2013_P90_Scores.csv')
# print(df.head())

file_list = ['./data/2013_P90_Scores.csv', './data/2014_P90_Scores.csv', './data/2015_P90_Scores.csv', './data/2016_P90_Scores.csv', './data/2017_P90_Scores.csv', './data/2018_P90_Scores.csv', './data/2019_P90_Scores.csv', './data/2020_P90_Scores.csv', './data/2021_P90_Scores.csv', './data/2022_P90_Scores.csv', './data/2023_P90_Scores.csv']
drop_columns = ['X', 'Y', 'x', 'y', 'GlobalID', 'OBJECTID', 'Class', 'Appd_Std', 'Restr_Std', 'Min_Date', 'Lat_DD', 'Long_DD','Grow_Area']

dataframes = [pd.read_csv(file) for file in file_list]
df = pd.concat(dataframes, ignore_index=True)

print(f"df shape: {df.shape}")

X = df.drop(columns=drop_columns)
y = df['P90']


print(X.head())
print("-------")
print(y.head())
