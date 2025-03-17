import pandas as pd
import itertools
from tqdm import tqdm
from numpy.random import choice
from individual import BooleanNode, EndNode, EvaluationNode, Individual
import numpy as np
import individual
import process_stocks
import random


class Population:
    POPULATION_SIZE = 4  # even number cuase lazy parent strategy
    NUM_GENS = 2
    NUM_SECTORS = 3

    def __init__(self):
        self.population = [
            self.gen_init_dividual() for _ in range(Population.POPULATION_SIZE)
        ]

    def gen_init_dividual(self):
        ind = Individual(
            Population.NUM_SECTORS,
            self.gen_random_sector_weights(),
            self.gen_random_tree(),
            self.gen_random_stock_weights(),
        )
        return ind

    def evolve_generations(self):
        for _ in tqdm(range(Population.NUM_GENS)):
            self.run_generation()
        fitnesses = []
        best_individual = None
        best_fitness = 0
        for individual in self.population:
            curr_fitness = individual.evaluate_fitness()
            fitnesses.append(curr_fitness)
            if best_individual == None or curr_fitness > best_fitness:
                best_individual = individual
                best_fitness = curr_fitness
        print(f"in sample perf: {best_fitness}")
        return best_individual

    def perform_test(self, indv):
        self.set_out_of_sample(indv.root)
        indv.training = False
        perf = indv.evaluate_fitness()
        print(f"out of sample perf: {perf}")

    def set_out_of_sample(self, curr_node):
        if isinstance(curr_node, EndNode):
            curr_node.training = False
        else:
            if curr_node.left != None:
                self.set_out_of_sample(curr_node.left)
            if curr_node.right != None and not isinstance(curr_node.right, float):
                self.set_out_of_sample(curr_node.right)

    def run_generation(self):
        fitnesses = []
        best_individual = None
        best_fitness = 0
        for individual in self.population:
            curr_fitness = individual.evaluate_fitness()
            fitnesses.append(curr_fitness)
            if best_individual == None or curr_fitness > best_fitness:
                best_individual = individual
                best_fitness = curr_fitness
        fitnesses = np.array(fitnesses)
        p = fitnesses / fitnesses.sum()
        p = p.tolist()
        p_dict = {i: p[i] for i in range(len(p))}
        next_population = []
        for _ in range(0, Population.POPULATION_SIZE, 2):
            index1 = choice(
                list(p_dict.keys()),
                p=list(p_dict.values()),
            )
            del p_dict[index1]
            total = sum(p_dict.values())
            p_dict = {pair[0]: pair[1] / total for pair in p_dict.items()}
            parent1 = self.population[index1]

            index2 = choice(
                list(p_dict.keys()),
                p=list(p_dict.values()),
            )
            del p_dict[index2]
            total = sum(p_dict.values())
            p_dict = {pair[0]: pair[1] / total for pair in p_dict.items()}
            parent2 = self.population[index2]

            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            next_population.append(child1)
            next_population.append(child2)
        next_population[-1] = best_individual
        self.population = next_population

    def mutate(self, child):
        sector_weights = np.array(child.sector_weights)
        random_arr = np.random.rand(len(sector_weights)) / 3
        child.sector_weights = sector_weights + random_arr

        for stock in child.stock_weights.keys():
            child.stock_weights[stock] += random.randint(1, 10) / 100

        p_sum = sum(child.stock_weights.values())
        for stock in child.stock_weights.keys():
            child.stock_weights[stock] /= p_sum
        return child

        # randomize tree stuff later

    def copy_tree(self, root_node):
        if isinstance(root_node, EndNode):
            return EndNode(root_node.window_size, root_node.type_of_node)
        elif root_node == None:
            return None
        elif isinstance(root_node, float):
            return root_node
        else:
            left = self.copy_tree(root_node.left)
            right = self.copy_tree(root_node.right)
            if isinstance(root_node, BooleanNode):
                return BooleanNode(root_node.operation, left, right)
            else:
                return EvaluationNode(root_node.operation, left, right)

    def copy_tree_until(self, curr_node, n, implant):
        if n == 0:
            return implant
        if isinstance(curr_node, EndNode):
            return EndNode(curr_node.window_size, curr_node.type_of_node)
        elif curr_node == None:
            return None
        elif isinstance(curr_node, float):
            return curr_node
        else:
            left = self.copy_tree_until(curr_node.left, n - 1, implant)
            right = self.copy_tree_until(curr_node.right, n - 2, implant)
            if isinstance(curr_node, BooleanNode):
                return BooleanNode(curr_node.operation, left, right)
            else:
                return EvaluationNode(curr_node.operation, left, right)

    def select_node(self, curr_node, n):
        if n == 0:
            return curr_node
        if (
            curr_node == None
            or isinstance(curr_node, EndNode)
            or isinstance(curr_node, float)
        ):
            return None
        else:
            temp1 = self.select_node(curr_node.left, n - 1)
            temp2 = self.select_node(curr_node.right, n - 2)
            return temp1 or temp2

    def crossover(self, parent1, parent2):
        # sector weights
        idx = random.randint(0, len(process_stocks.sectors))
        sector_weights1 = (
            parent1.sector_weights[:idx].tolist()
            + parent2.sector_weights[idx:].tolist()
        )
        sector_weights1 = np.array(sector_weights1)
        sector_weights1 /= sector_weights1.sum()

        sector_weights2 = (
            parent2.sector_weights[:idx].tolist()
            + parent1.sector_weights[idx:].tolist()
        )
        sector_weights2 = np.array(sector_weights2)
        sector_weights2 /= sector_weights2.sum()

        # stock weights
        idx = random.randint(0, len(process_stocks.tickers))
        stock_weights1 = dict(list(parent1.stock_weights.items())[:idx])
        stock_weights1.update(dict(list(parent2.stock_weights.items())[idx:]))

        stock_weights2 = dict(list(parent2.stock_weights.items())[:idx])
        stock_weights2.update(dict(list(parent1.stock_weights.items())[idx:]))

        # tree
        node1 = None
        idx1 = 0
        while not isinstance(node1, BooleanNode) or node1.operation == BooleanNode.NOT:
            idx1 = random.randint(0, parent1.tree_size)
            node1 = self.select_node(parent1.root, idx1)

        node2 = None
        idx2 = 0
        while not isinstance(node2, BooleanNode) or node2.operation == BooleanNode.NOT:
            idx2 = random.randint(0, parent2.tree_size)
            node2 = self.select_node(parent2.root, idx2)

        fragment1 = self.copy_tree(node1)
        fragment2 = self.copy_tree(node2)

        root1 = self.copy_tree_until(parent1.root, idx1, fragment1)
        root2 = self.copy_tree_until(parent2.root, idx2, fragment2)

        child1 = Individual(
            Population.NUM_SECTORS, sector_weights1, root1, stock_weights1
        )
        child2 = Individual(
            Population.NUM_SECTORS, sector_weights2, root2, stock_weights2
        )
        return child1, child2

    def gen_random_sector_weights(self):
        random_sector_weights = np.array(
            [random.randint(1, 10) for _ in range(len(process_stocks.sectors))]
        )
        random_sector_weights = random_sector_weights / sum(random_sector_weights)
        return random_sector_weights

    def gen_random_stock_weights(self):
        random_stock_weights = {
            stock: random.randint(1, 10) for stock in process_stocks.tickers
        }
        p_sum = sum(random_stock_weights.values())
        for stock in random_stock_weights.keys():
            random_stock_weights[stock] /= p_sum
        return random_stock_weights

    def gen_random_tree(self):
        boolean_choices = BooleanNode.types.copy()
        boolean_choices.remove(BooleanNode.NOT)
        type_of_node = random.choice(boolean_choices)
        start_level = random.randint(2, 8)
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
    best_individual = pop.evolve_generations()
    pop.perform_test(best_individual)
