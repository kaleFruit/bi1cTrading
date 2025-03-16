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
        strategy_returns = np.multiply(
            self.get_signal(ticker), process_stocks.daily_returns
        )
        return (
            strategy_returns[strategy_returns > 0].sum()
            / np.absolute(strategy_returns[strategy_returns < 0]).sum()
        )  # pf: wins/losses

    def get_signal(self, ticker):
        data = process_stocks.totalData.loc[
            process_stocks.totalData["Ticker"] == ticker
        ].iloc[process_stocks.LARGEST_WINDOW_SIZE : -process_stocks.MAX_DATE]
        signals = []
        for idx in tqdm(range(process_stocks.LARGEST_WINDOW_SIZE, len(data))):
            if self.root.evaluate(ticker, idx):
                signals.append(1)
            else:
                signals.append(0)
        print(signals)
        return np.array(signals)


class CompareNode:
    LESS_THAN = 0
    GREATER_THAN = 1
    NOT = 2
    AND = 3
    OR = 4
    ADD = 5
    SCALE = 6

    def __init__(self, operation, left, right):
        self.operation = operation
        self.left = left
        self.right = right

    def evaluate(self, ticker: str, idx: int):
        left = self.left.evaluate(ticker, idx)
        right = self.right.evaluate(ticker, idx)

        if self.operation == CompareNode.LESS_THAN:
            return left < right
        elif self.operation == CompareNode.GREATER_THAN:
            return left > right
        elif self.operation == CompareNode.NOT:  # assumes one is none
            return not left and not right
        elif self.operation == CompareNode.AND:
            return left and right
        elif self.operation == CompareNode.OR:
            return left or right
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

    def evaluate(self, ticker: str, idx: int):
        return (
            process_stocks.totalData.loc[process_stocks.totalData["Ticker"] == ticker]
            .iloc[idx - self.window_size : idx][[self.type]]
            .values
        )


class EvaluationNode:
    AVERAGE = 0
    MAX = 1
    MIN = 2

    def __init__(self, operation, child):
        self.operation = operation
        self.child = child

    def evaluate(self, ticker: str, idx: int):
        evaluated = self.child.evaluate(ticker, idx)
        if self.operation == EvaluationNode.MAX:
            return evaluated.max()
        elif self.operation == EvaluationNode.MIN:
            return evaluated.min()
        elif self.operation == EvaluationNode.AVERAGE:
            return evaluated.mean()


if __name__ == "__main__":
    node1 = EndNode(30, EndNode.CLOSE)
    node2 = EndNode(50, EndNode.CLOSE)
    nodeComp1 = EvaluationNode(EvaluationNode.AVERAGE, node1)
    nodeComp2 = EvaluationNode(EvaluationNode.AVERAGE, node2)
    root = CompareNode(CompareNode.GREATER_THAN, nodeComp2, nodeComp1)
    ind = Individual(2, [], root)
    ind.evaluate_return("AAPL")
