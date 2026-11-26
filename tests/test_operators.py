import math
import random
import unittest

from symbolic_regression import (
    Binary,
    Constant,
    GeneticOperationError,
    GeneticOperators,
    Parameter,
    ParameterNode,
    Power,
    TreeGenerator,
    TreeGeneratorConfig,
    Unary,
    Variable,
    expression_from_dict,
)


def make_config(**overrides):
    arguments = {
        "variables": ("x", "y"),
        "min_depth": 1,
        "max_depth": 4,
        "max_nodes": 10,
        "unary_functions": ("sin", "cos", "neg"),
        "binary_operators": ("add", "sub", "mul"),
        "include_power": True,
        "variable_probability": 0.6,
        "parameter_probability": 0.5,
        "leaf_probability": 0.3,
    }
    arguments.update(overrides)
    return TreeGeneratorConfig(**arguments)


def sample_expression():
    return Binary(
        "add",
        Unary("sin", Variable("x")),
        ParameterNode(Parameter("a", 1)),
    )


class OperatorReproducibilityTests(unittest.TestCase):
    def test_crossover_is_reproducible(self):
        config = make_config()
        parents = TreeGenerator(config, seed=100).generate_many(2)
        first = GeneticOperators(config, seed=7)
        second = GeneticOperators(config, seed=7)

        self.assertEqual(
            first.crossover(*parents),
            second.crossover(*parents),
        )

    def test_mutations_are_reproducible(self):
        config = make_config()
        expression = sample_expression()

        for method_name in (
            "point_mutation",
            "subtree_mutation",
            "parameter_mutation",
        ):
            with self.subTest(method=method_name):
                first = GeneticOperators(config, seed=17)
                second = GeneticOperators(config, seed=17)
                self.assertEqual(
                    getattr(first, method_name)(expression),
                    getattr(second, method_name)(expression),
                )

    def test_global_random_state_is_not_changed(self):
        random.seed(2026)
        state = random.getstate()
        operators = GeneticOperators(make_config(), seed=5)
        expression = sample_expression()

        operators.point_mutation(expression)
        operators.parameter_mutation(expression)
        operators.subtree_mutation(expression)

        self.assertEqual(random.getstate(), state)

    def test_operator_configuration_is_validated(self):
        for scale in (0, -1, math.nan, math.inf):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    GeneticOperators(
                        make_config(),
                        seed=1,
                        parameter_mutation_scale=scale,
                    )
        with self.assertRaises(ValueError):
            GeneticOperators(make_config(), seed=1, max_attempts=0)


class CrossoverTests(unittest.TestCase):
    def test_crossover_at_disambiguates_parameter_names(self):
        first_parameter = Parameter("p", 1)
        second_parameter = Parameter("p", 10)
        first = Binary(
            "add",
            ParameterNode(first_parameter),
            Binary(
                "mul",
                ParameterNode(first_parameter),
                Variable("x"),
            ),
        )
        second = Binary(
            "add",
            ParameterNode(second_parameter),
            Binary(
                "mul",
                ParameterNode(second_parameter),
                Variable("y"),
            ),
        )
        operators = GeneticOperators(make_config(), seed=1)

        first_child, second_child = operators.crossover_at(
            first,
            second,
            (1,),
            (1,),
        )

        self.assertEqual(len(first_child.parameters()), 2)
        self.assertEqual(len(second_child.parameters()), 2)
        self.assertIn("cross_second_p", first_child.parameters())
        self.assertIn("cross_first_p", second_child.parameters())
        self.assertEqual(first_child.evaluate({"y": 2}), 21)
        self.assertEqual(second_child.evaluate({"x": 3}), 13)
        self.assertEqual(first.parameters()["p"].value, 1)
        self.assertEqual(second.parameters()["p"].value, 10)

    def test_crossover_at_root_does_not_rename_unneeded_parameters(self):
        first = Variable("x")
        second = ParameterNode(Parameter("p", 4))
        operators = GeneticOperators(make_config(), seed=1)

        first_child, second_child = operators.crossover_at(
            first,
            second,
            (),
            (),
        )

        self.assertEqual(first_child, second)
        self.assertEqual(second_child, first)

    def test_random_crossovers_respect_constraints(self):
        config = make_config()
        parents = TreeGenerator(config, seed=10).generate_many(2)
        operators = GeneticOperators(config, seed=20)

        for _ in range(30):
            children = operators.crossover(*parents)
            for child in children:
                complexity = child.complexity()
                self.assertLessEqual(complexity.depth, config.max_depth)
                self.assertLessEqual(complexity.node_count, config.max_nodes)

    def test_crossover_rejects_oversized_parent(self):
        config = make_config(max_depth=2, max_nodes=3)
        oversized = Binary(
            "add",
            Binary("add", Variable("x"), Variable("x")),
            Variable("y"),
        )
        operators = GeneticOperators(config, seed=1)

        with self.assertRaisesRegex(ValueError, "parent expression exceeds"):
            operators.crossover(oversized, Variable("x"))

    def test_crossover_results_survive_serialization(self):
        config = make_config()
        parents = TreeGenerator(config, seed=10).generate_many(2)
        children = GeneticOperators(config, seed=20).crossover(*parents)

        for child in children:
            self.assertEqual(expression_from_dict(child.to_dict()), child)


class SubtreeMutationTests(unittest.TestCase):
    def test_subtree_mutation_at_disambiguates_parameters(self):
        expression = Binary(
            "add",
            ParameterNode(Parameter("p", 1)),
            Variable("x"),
        )
        replacement = ParameterNode(Parameter("p", 10))
        operators = GeneticOperators(make_config(), seed=1)

        result = operators.subtree_mutation_at(
            expression,
            (1,),
            replacement,
        )

        self.assertEqual(len(result.parameters()), 2)
        self.assertIn("mutation_p", result.parameters())
        self.assertEqual(result.evaluate({"x": 0}), 11)
        self.assertEqual(expression.parameters()["p"].value, 1)

    def test_subtree_mutation_at_root_preserves_replacement(self):
        replacement = ParameterNode(Parameter("p", 10))
        operators = GeneticOperators(make_config(), seed=1)

        result = operators.subtree_mutation_at(
            sample_expression(),
            (),
            replacement,
        )

        self.assertEqual(result, replacement)

    def test_random_subtree_mutations_respect_constraints(self):
        config = make_config()
        expression = TreeGenerator(config, seed=10).generate()
        operators = GeneticOperators(config, seed=20)

        for _ in range(30):
            result = operators.subtree_mutation(expression)
            complexity = result.complexity()
            self.assertLessEqual(complexity.depth, config.max_depth)
            self.assertLessEqual(complexity.node_count, config.max_nodes)


class PointMutationTests(unittest.TestCase):
    def test_unary_point_mutation_preserves_operand(self):
        operators = GeneticOperators(
            make_config(unary_functions=("sin", "cos")),
            seed=1,
        )
        expression = Unary("sin", Variable("x"))

        self.assertEqual(
            operators.point_mutation(expression),
            Unary("cos", Variable("x")),
        )

    def test_binary_point_mutation_preserves_children(self):
        operators = GeneticOperators(
            make_config(binary_operators=("add", "sub")),
            seed=1,
        )
        expression = Binary("add", Variable("x"), Variable("y"))

        self.assertEqual(
            operators.point_mutation(expression),
            Binary("sub", Variable("x"), Variable("y")),
        )

    def test_point_mutation_requires_replaceable_node(self):
        operators = GeneticOperators(make_config(), seed=1)

        for expression in (Variable("x"), Power(Variable("x"), Constant(2))):
            with self.subTest(expression=str(expression)):
                with self.assertRaises(GeneticOperationError):
                    operators.point_mutation(expression)


class ParameterMutationTests(unittest.TestCase):
    def test_shared_parameter_is_updated_in_all_occurrences(self):
        parameter = Parameter("a", 1)
        expression = Binary(
            "add",
            Binary("mul", ParameterNode(parameter), Variable("x")),
            ParameterNode(parameter),
        )
        operators = GeneticOperators(make_config(), seed=1)

        result = operators.parameter_mutation(expression)

        self.assertEqual(len(result.parameters()), 1)
        self.assertNotEqual(result.parameters()["a"].value, 1)
        values = {
            node.parameter.value
            for node in result.walk()
            if isinstance(node, ParameterNode)
        }
        self.assertEqual(len(values), 1)
        self.assertEqual(expression.parameters()["a"].value, 1)

    def test_parameter_mutation_respects_bounds(self):
        expression = ParameterNode(Parameter("a", 0, lower=-1, upper=1))
        operators = GeneticOperators(
            make_config(),
            seed=1,
            parameter_mutation_scale=10,
        )

        result = operators.parameter_mutation(expression)

        value = result.parameters()["a"].value
        self.assertGreaterEqual(value, -1)
        self.assertLessEqual(value, 1)
        self.assertNotEqual(value, 0)

    def test_fixed_constant_can_be_mutated(self):
        operators = GeneticOperators(make_config(), seed=1)

        result = operators.parameter_mutation(Constant(2))

        self.assertIsInstance(result, Constant)
        self.assertNotEqual(result.value, 2)
        self.assertGreaterEqual(result.value, -5)
        self.assertLessEqual(result.value, 5)

    def test_constant_exponent_uses_exponent_bounds(self):
        config = make_config(exponent_lower=1, exponent_upper=1.5)
        operators = GeneticOperators(config, seed=1)
        expression = Power(Variable("x"), Constant(2))

        result = operators.parameter_mutation(expression)

        self.assertIsInstance(result, Power)
        self.assertGreaterEqual(result.exponent.value, 1)
        self.assertLessEqual(result.exponent.value, 1.5)
        self.assertNotEqual(result.exponent.value, 2)

    def test_parameter_mutation_requires_mutable_numeric_leaf(self):
        operators = GeneticOperators(make_config(), seed=1)
        expressions = [
            Variable("x"),
            ParameterNode(Parameter("a", 1, optimizable=False)),
            ParameterNode(Parameter("a", 1, lower=1, upper=1)),
        ]

        for expression in expressions:
            with self.subTest(expression=str(expression)):
                with self.assertRaises(GeneticOperationError):
                    operators.parameter_mutation(expression)


if __name__ == "__main__":
    unittest.main()
