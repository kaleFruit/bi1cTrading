import pandas as pd
from individual import BooleanNode, EndNode, EvaluationNode, Individual
import numpy as np
import individual
import process_stocks
import random


class Population:
    POPULATION_SIZE = 1

    def __init__(self):
        self.population = []

    def gen_individual(self):
        num_sectors = random.randint(1, 10)
        ind = Individual(
            num_sectors, self.gen_random_sector_weights(), self.gen_random_tree()
        )
        return ind

    def gen_random_sector_weights(self):
        random_stock_weights = np.array(
            [random.randint(1, 20) for _ in range(len(process_stocks.sectors))]
        )
        random_stock_weights = random_stock_weights / sum(random_stock_weights)
        return random_stock_weights

    def gen_random_tree(self):
        boolean_choices = BooleanNode.types.copy()
        boolean_choices.remove(BooleanNode.NOT)
        type_of_node = random.choice(boolean_choices)
        start_level = random.randint(2, 10)
        left = self.create_node(start_level, individual.BOOLEAN_NODE, type_of_node)
        right = self.create_node(start_level, individual.BOOLEAN_NODE, type_of_node)
        root = BooleanNode(type_of_node, left, right)
        return root

    def create_node(self, level, prev_node_type, fine_grained_type):
        if level == 0:
            type_of_node = random.choice(EndNode.types)
            if prev_node_type == individual.EVALUATION_NODE:
                if (
                    fine_grained_type == EvaluationNode.ADD
                    or fine_grained_type == EvaluationNode.SCALE
                    or fine_grained_type == EvaluationNode.GREATER_THAN
                    or fine_grained_type == EvaluationNode.LESS_THAN
                ):
                    window_size = 1
                else:
                    window_size = random.randint(1, process_stocks.LARGEST_WINDOW_SIZE)
                return EndNode(window_size, type_of_node)
            elif prev_node_type == individual.BOOLEAN_NODE:
                type_of_node = random.choice([EndNode.TRUE, EndNode.FALSE])
                return EndNode(1, type_of_node)
        else:
            if prev_node_type == individual.EVALUATION_NODE:
                base_node_type = individual.EVALUATION_NODE
                available = EvaluationNode.types.copy()
                available.remove(EvaluationNode.LESS_THAN)
                available.remove(EvaluationNode.GREATER_THAN)
                type_of_node = random.choice(available)

                if EvaluationNode.types_num_children[type_of_node] == 1:
                    return EvaluationNode(
                        type_of_node,
                        self.create_node(
                            max(0, level - random.randint(0, 2)),
                            base_node_type,
                            type_of_node,
                        ),
                    )
                elif type_of_node == EvaluationNode.SCALE:
                    scale_factor = float(random.randint(1, 100)) / 100.0
                    return EvaluationNode(
                        type_of_node,
                        self.create_node(
                            max(0, level - random.randint(0, 2)),
                            base_node_type,
                            type_of_node,
                        ),
                        scale_factor,
                    )
                else:
                    left = self.create_node(
                        max(0, level - random.randint(0, 2)),
                        base_node_type,
                        type_of_node,
                    )
                    right = self.create_node(
                        max(0, level - random.randint(0, 2)),
                        base_node_type,
                        type_of_node,
                    )
                    return EvaluationNode(type_of_node, left, right)
            elif prev_node_type == individual.BOOLEAN_NODE:
                base_node_type = random.choice(
                    [individual.BOOLEAN_NODE, individual.EVALUATION_NODE]
                )
                if base_node_type == individual.BOOLEAN_NODE:
                    type_of_node = random.choice(BooleanNode.types)
                    left = self.create_node(
                        max(0, level - random.randint(0, 2)),
                        base_node_type,
                        type_of_node,
                    )
                    right = self.create_node(
                        max(0, level - random.randint(0, 2)),
                        base_node_type,
                        type_of_node,
                    )
                    return BooleanNode(type_of_node, left, right)
                else:
                    type_of_node = random.choice(
                        [EvaluationNode.GREATER_THAN, EvaluationNode.LESS_THAN]
                    )
                    left = self.create_node(
                        max(0, level - random.randint(0, 2)),
                        base_node_type,
                        type_of_node,
                    )
                    right = self.create_node(
                        max(0, level - random.randint(0, 2)),
                        base_node_type,
                        type_of_node,
                    )
                    return EvaluationNode(type_of_node, left, right)


if __name__ == "__main__":
    pop = Population()
    test = pop.gen_individual()
    test.pick_stocks()
