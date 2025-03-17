import numpy as np
from numpy.random import choice
from pandas.core.frame import find_common_type
import process_stocks
from tqdm import tqdm
import pandas as pd
import random


class Individual:
    def __init__(self, num_sectors, sector_weights, root, stock_weights: dict):
        self.sector_weights = sector_weights
        self.num_sectors = num_sectors
        self.chosen_sectors = choice(
            process_stocks.sectors.tolist(),
            num_sectors,
            p=sector_weights,
            replace=False,
        )
        self.stock_weights = stock_weights
        self.profit = 0
        self.root = root
        self.fitness = 0
        self.tree_size = 0
        self.find_tree_size(self.root)

    def find_tree_size(self, curr_node):
        self.tree_size += 1
        if curr_node == None or isinstance(curr_node, float):
            self.tree_size -= 1
            return
        elif isinstance(curr_node, EndNode):
            return
        else:
            self.find_tree_size(curr_node.left)
            self.find_tree_size(curr_node.right)

    def pick_stocks(self):
        stocks_list = []
        for sector in self.chosen_sectors:
            available_stocks = process_stocks.tickers_per_sector[sector]
            stock_weights = np.array(
                [self.stock_weights[stock] for stock in available_stocks]
            )
            stock_weights = stock_weights / (stock_weights).sum()
            stocks = choice(
                available_stocks,
                3,
                p=stock_weights,
                replace=True,
            )
            stocks_list.extend(stocks)
        return stocks_list

    def evaluate_fitness(self):
        stocks = self.pick_stocks()
        stock_perfs = []
        for stock in stocks:
            stock_perf = self.evaluate_return(stock)
            if stock_perf == 100:
                self.stock_weights[stock] /= 1.5
                stock_perfs.append(0.1)
            else:
                stock_perfs.append(stock_perf)
                self.stock_weights[stock] *= 2
        fitness = np.array(stock_perfs).mean()
        self.fitness = fitness
        return fitness

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
        if np.absolute(strategy_returns[strategy_returns < 0]).sum() == 0:
            return 100
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
    TRUE = 1
    FALSE = 2
    types = [CLOSE, OPEN, HIGH, LOW]

    def __init__(self, window_size, type_of_node):
        self.window_size = window_size
        self.type_of_node = type_of_node

    def evaluate(self, ticker: str):
        if self.type_of_node == EndNode.TRUE or self.type_of_node == EndNode.FALSE:
            if self.type_of_node == EndNode.TRUE:
                data = pd.DataFrame(np.full(process_stocks.LENGTH, True))
            else:
                data = pd.DataFrame(np.full(process_stocks.LENGTH, False))
            data.reset_index(inplace=True, drop=True)
            data.columns = ["vals"]
            return data

        if self.window_size != 1:
            data = process_stocks.totalData.loc[
                process_stocks.totalData["Ticker"] == ticker
            ][[self.type_of_node]].rolling(self.window_size)
            data.columns = ["vals"]
            return data
        data = process_stocks.totalData.loc[
            process_stocks.totalData["Ticker"] == ticker
        ][[self.type_of_node]]
        data.columns = ["vals"]
        return data


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
        self.window_size = 2

    def evaluate(self, ticker: str):
        left = self.left.evaluate(ticker)
        if (
            self.operation == EvaluationNode.MAX
            or self.operation == EvaluationNode.MIN
            or self.operation == EvaluationNode.AVERAGE
        ):
            if isinstance(left, pd.api.typing.Rolling):
                rolled_left = left
            else:
                rolled_left = left.rolling(self.window_size)
            if self.operation == EvaluationNode.MAX:
                data = rolled_left.max()
            elif self.operation == EvaluationNode.MIN:
                data = rolled_left.min()
            else:
                data = rolled_left.mean()
            data.reset_index(drop=True, inplace=True)
            data.columns = ["vals"]
            return data

        if self.right != None and self.operation != EvaluationNode.SCALE:
            right = self.right.evaluate(ticker)
            left.reset_index(drop=True, inplace=True)
            right.reset_index(drop=True, inplace=True)
            if self.operation == EvaluationNode.ADD:
                return left + right
            elif self.operation == EvaluationNode.LESS_THAN:
                data = pd.DataFrame(np.where(left < right, 1, 0).flatten())
                data.columns = ["vals"]
                return data
            elif self.operation == EvaluationNode.GREATER_THAN:
                data = pd.DataFrame(np.where(left > right, 1, 0).flatten())
                data.columns = ["vals"]
                return data
        elif self.right != None and self.operation == EvaluationNode.SCALE:
            left.reset_index(drop=True, inplace=True)
            return left * self.right


if __name__ == "__main__":
    node1 = EndNode(10, EndNode.CLOSE)
    node2 = EndNode(50, EndNode.CLOSE)
    nodeComp1 = EvaluationNode(EvaluationNode.AVERAGE, node1)
    nodeComp2 = EvaluationNode(EvaluationNode.AVERAGE, node2)
    root = EvaluationNode(EvaluationNode.GREATER_THAN, nodeComp2, nodeComp1)
    ind = Individual(2, [], root, {})
    print(ind.evaluate_return("AAPL"))
