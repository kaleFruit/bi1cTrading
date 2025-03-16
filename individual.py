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
        self.chosen_sectors = choice(
            process_stocks.sectors.tolist(),
            num_sectors,
            # p=sector_weights,
            replace=False,
        )
        self.profit = 0
        self.root = root

    def pick_stocks(self):
        for sector in self.chosen_sectors:
            # just pick stock with best performance
            best_stock = None
            best_stock_perf = 0
            for stock in process_stocks.tickers_per_sector[sector]:
                stock_perf = self.evaluate_return(stock)
                if best_stock is None or stock_perf > best_stock_perf:
                    best_stock = stock
                    best_stock_perf = stock_perf

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


BOOLEAN_NODE = 0
END_NODE = 1
EVALUATION_NODE = 2


class Node:
    def __init__(self, operation, left, right=None):
        self.operation = operation
        self.left = left
        self.right = right

    def has_two_children(self):
        if self.right != None:
            return True
        return False


class BooleanNode(Node):
    AND = 3
    OR = 4
    NOT = 5
    types = [AND, OR, NOT]
    types_num_children = {AND: 2, OR: 2, NOT: 1}

    def evaluate(self, ticker: str):
        left = self.left.evaluate(ticker)

        if self.operation == BooleanNode.NOT:
            return np.logical_not(left)

        if self.right != None:
            right = self.right.evaluate(ticker)

            if self.operation == BooleanNode.AND:
                return np.logical_and(left, right)
            elif self.operation == BooleanNode.OR:
                return np.logical_or(left, right)


class EndNode:
    CLOSE = "Close"
    OPEN = "Open"
    HIGH = "High"
    LOW = "Low"
    types = [CLOSE, OPEN, HIGH, LOW]

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
    SCALE = 3
    ADD = 4
    LESS_THAN = 5
    GREATER_THAN = 6
    types = [AVERAGE, MAX, MIN, SCALE, ADD, LESS_THAN, GREATER_THAN]
    types_num_children = {
        AVERAGE: 1,
        MAX: 1,
        MIN: 1,
        SCALE: 2,
        ADD: 2,
        LESS_THAN: 2,
        GREATER_THAN: 2,
    }

    def __init__(self, operation, left, right=None):
        self.operation = operation
        self.left = left
        self.right = right

    def evaluate(self, ticker: str):
        left = self.left.evaluate(ticker)

        if self.operation == EvaluationNode.MAX:
            return left.max()
        elif self.operation == EvaluationNode.MIN:
            return left.min()
        elif self.operation == EvaluationNode.AVERAGE:
            return left.mean()

        if self.right != None:
            right = self.right.evaluate(ticker)
            if self.operation == EvaluationNode.ADD:
                return left + right
            elif self.operation == EvaluationNode.SCALE:
                return left * right
            elif self.operation == EvaluationNode.LESS_THAN:
                return np.where(left < right, 1, 0)
            elif self.operation == EvaluationNode.GREATER_THAN:
                return np.where(left > right, 1, 0)


if __name__ == "__main__":
    node1 = EndNode(10, EndNode.CLOSE)
    node2 = EndNode(50, EndNode.CLOSE)
    nodeComp1 = EvaluationNode(EvaluationNode.AVERAGE, node1)
    nodeComp2 = EvaluationNode(EvaluationNode.AVERAGE, node2)
    root = BooleanNode(BooleanNode.GREATER_THAN, nodeComp2, nodeComp1)
    ind = Individual(2, [], root)
    print(ind.evaluate_return("AAPL"))
