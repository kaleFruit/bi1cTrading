import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import itertools
import pandas as pd
from tqdm import tqdm
from time import perf_counter

totalData = pd.read_csv("filteredData.csv")
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
MAX_DATE = 300
TRAINING_LENGTH = totalData.loc[totalData["Ticker"] == "AAPL"].shape[0] - MAX_DATE
TEST_LENGTH = MAX_DATE + LARGEST_WINDOW_SIZE


def find_sharpe_ratio(ref: pd.DataFrame) -> float:
    return ref.mean() / ref.std()
