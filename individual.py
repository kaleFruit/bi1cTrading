import numpy as np
from numpy.random import choice
from pandas.core import window
from pandas.core.arrays import BooleanArray
import process_stocks
from tqdm import tqdm


class Individual:
    def __init__(self, num_sectors, sector_weights, root):
        self.sector_weights = sector_weights
        self.num_sectors = num_sectors
        # self.chosen_sectors = choice(
        #     process_stocks.sectors.tolist(),
        #     num_sectors,
        #     p=sector_weights,
        #     replace=False,
        # )
        self.profit = 0
        self.root = root

    def pick_stocks(self):
        pass

    def evaluate_return(self, ticker):
        daily_returns = (
            np.log(
                process_stocks.totalData.loc[
                    process_stocks.totalData["Ticker"] == ticker
                ][["Close"]]
            )
            .diff()
            .shift(-1)
            .values
        )[process_stocks.LARGEST_WINDOW_SIZE : process_stocks.MAX_DATE]
        signals = self.get_signal(ticker)[
            process_stocks.LARGEST_WINDOW_SIZE : process_stocks.MAX_DATE
        ]
        strategy_returns = np.multiply(signals, daily_returns)
        return (
            strategy_returns[strategy_returns > 0].sum()
            / np.absolute(strategy_returns[strategy_returns < 0]).sum()
        )  # pf: wins/losses

    def get_signal(self, ticker):
        signals = self.root.evaluate(ticker)
        return np.array(signals)


class CompareNode:
    LESS_THAN = 0
    GREATER_THAN = 1
    AND = 3
    OR = 4
    ADD = 5
    SCALE = 6

    def __init__(self, operation, left, right):
        self.operation = operation
        self.left = left
        self.right = right

    def evaluate(self, ticker: str):
        left = self.left.evaluate(ticker)
        right = self.right.evaluate(ticker)

        if self.operation == CompareNode.LESS_THAN:
            return np.where(left < right, 1, 0)
        elif self.operation == CompareNode.GREATER_THAN:
            return np.where(left > right, 1, 0)
        elif self.operation == CompareNode.AND:
            return np.logical_and(left, right)
        elif self.operation == CompareNode.OR:
            return np.logical_or(left, right)
        elif self.operation == CompareNode.ADD:
            return left + right
        elif self.operation == CompareNode.SCALE:
            return left * right


class EndNode:
    CLOSE = "Close"
    OPEN = "Open"
    HIGH = "High"
    LOW = "Low"

    def __init__(self, window_size, type):
        self.window_size = window_size
        self.type = type

    def evaluate(self, ticker: str):
        if self.window_size != 1:
            return process_stocks.totalData.loc[
                process_stocks.totalData["Ticker"] == ticker
            ][[self.type]].rolling(self.window_size)
        return process_stocks.totalData.loc[
            process_stocks.totalData["Ticker"] == ticker
        ][[self.type]].values


class EvaluationNode:
    AVERAGE = 0
    MAX = 1
    MIN = 2
    NOT = 3

    def __init__(self, operation, child):
        self.operation = operation
        self.child = child

    def evaluate(self, ticker: str):
        evaluated = self.child.evaluate(ticker)
        if self.operation == EvaluationNode.MAX:
            return evaluated.max()
        elif self.operation == EvaluationNode.MIN:
            return evaluated.min()
        elif self.operation == EvaluationNode.AVERAGE:
            return evaluated.mean()
        elif self.operation == EvaluationNode.NOT:
            return np.logical_not(evaluated)


if __name__ == "__main__":
    node1 = EndNode(10, EndNode.CLOSE)
    node2 = EndNode(50, EndNode.CLOSE)
    nodeComp1 = EvaluationNode(EvaluationNode.AVERAGE, node1)
    nodeComp2 = EvaluationNode(EvaluationNode.AVERAGE, node2)
    root = CompareNode(CompareNode.GREATER_THAN, nodeComp2, nodeComp1)
    ind = Individual(2, [], root)
    print(ind.evaluate_return("AAPL"))
