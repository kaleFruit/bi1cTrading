import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import itertools
import pandas as pd
from tqdm import tqdm
from time import perf_counter
import string

totalData = pd.read_csv("totalData.csv")
sectors = totalData["GICS Sector"].unique()
tickers = totalData["Ticker"].unique()
tickers_per_sector = {
    sector: totalData["Ticker"]
    .loc[totalData["GICS Sector"] == sector]
    .unique()
    .tolist()
    for sector in sectors
}

LARGEST_WINDOW_SIZE = 100
MAX_DATE = 500

data_lengths = []
for ticker in tickers:
    data = totalData.loc[totalData["Ticker"] == ticker][["Close"]].values
    data_lengths.append(data.shape[0])
ser = pd.Series(data=data_lengths, index=tickers)

# the majority of samples have length 3652
# 30 sampels have length 3651 cause they are missing the last day
second_largest_length = ser.value_counts().iloc[1]
filtered_tickers = ser.loc[ser.values >= second_largest_length].index.values


def find_sharpe_ratio(ref: pd.DataFrame) -> float:
    return ref.mean() / ref.std()
