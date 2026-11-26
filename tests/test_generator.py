import math
import random
import unittest

from symbolic_regression import (
    Binary,
    Constant,
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
        "max_nodes": 12,
        "unary_functions": ("sin", "cos", "log", "neg"),
        "binary_operators": ("add", "sub", "mul", "div"),
        "include_power": True,
        "variable_probability": 0.6,
        "parameter_probability": 0.5,
        "leaf_probability": 0.3,
    }
    arguments.update(overrides)
    return TreeGeneratorConfig(**arguments)


class GeneratorReproducibilityTests(unittest.TestCase):
    def test_same_seed_produces_same_tree_sequence(self):
        config = make_config()
        first = TreeGenerator(config, seed=2026)
        second = TreeGenerator(config, seed=2026)

        self.assertEqual(first.generate_many(20), second.generate_many(20))

    def test_generate_many_matches_consecutive_generate_calls(self):
        config = make_config()
        consecutive = TreeGenerator(config, seed=17)
        batch = TreeGenerator(config, seed=17)

        expected = [consecutive.generate(), consecutive.generate()]
        self.assertEqual(batch.generate_many(2), expected)

    def test_different_seeds_produce_different_sequences(self):
        config = make_config()
        first = TreeGenerator(config, seed=1).generate_many(10)
        second = TreeGenerator(config, seed=2).generate_many(10)

        self.assertNotEqual(first, second)

    def test_private_rng_does_not_change_global_random_state(self):
        random.seed(12345)
        state = random.getstate()

        TreeGenerator(make_config(), seed=9).generate()

        self.assertEqual(random.getstate(), state)

    def test_full_generation_is_reproducible(self):
        config = make_config(max_depth=3, max_nodes=7)
        first = TreeGenerator(config, seed=99).generate_many(5, "full")
        second = TreeGenerator(config, seed=99).generate_many(5, "full")

        self.assertEqual(first, second)


class GeneratorConstraintTests(unittest.TestCase):
    def test_grow_respects_depth_and_node_limits(self):
        config = make_config(min_depth=2, max_depth=5, max_nodes=11)
        generator = TreeGenerator(config, seed=123)

        for expression in generator.generate_many(100):
            with self.subTest(expression=str(expression)):
                complexity = expression.complexity()
                self.assertGreaterEqual(complexity.depth, 2)
                self.assertLessEqual(complexity.depth, 5)
                self.assertLessEqual(complexity.node_count, 11)

    def test_binary_only_generator_can_enforce_min_depth(self):
        config = make_config(
            min_depth=2,
            max_depth=4,
            max_nodes=15,
            unary_functions=(),
            include_power=False,
        )
        generator = TreeGenerator(config, seed=321)

        for expression in generator.generate_many(50):
            self.assertGreaterEqual(expression.depth(), 2)
            self.assertLessEqual(expression.complexity().node_count, 15)

    def test_full_tree_has_exact_depth_and_size(self):
        config = make_config(max_depth=3, max_nodes=7)

        expression = TreeGenerator(config, seed=5).generate("full")

        self.assertEqual(expression.depth(), 3)
        self.assertEqual(expression.complexity().node_count, 7)
        for node in expression.walk():
            if node.children():
                self.assertIsInstance(node, Binary)

    def test_full_tree_rejects_insufficient_node_budget(self):
        generator = TreeGenerator(
            make_config(max_depth=4, max_nodes=14),
            seed=5,
        )

        with self.assertRaisesRegex(ValueError, "full tree requires"):
            generator.generate("full")

    def test_full_tree_requires_binary_operator_above_depth_one(self):
        generator = TreeGenerator(
            make_config(
                max_depth=2,
                max_nodes=3,
                binary_operators=(),
            ),
            seed=5,
        )

        with self.assertRaisesRegex(ValueError, "requires at least one"):
            generator.generate("full")

    def test_single_node_budget_always_returns_leaf(self):
        config = make_config(max_depth=5, max_nodes=1)
        generator = TreeGenerator(config, seed=5)

        for expression in generator.generate_many(20):
            self.assertEqual(expression.complexity().node_count, 1)
            self.assertIsInstance(
                expression,
                (Variable, Constant, ParameterNode),
            )

    def test_power_exponents_are_numeric_and_nonlinear_parameters(self):
        config = make_config(
            max_depth=3,
            max_nodes=7,
            unary_functions=(),
            binary_operators=(),
            include_power=True,
            variable_probability=0,
            parameter_probability=1,
            leaf_probability=0,
        )

        expression = TreeGenerator(config, seed=5).generate()

        self.assertIsInstance(expression, Power)
        for node in expression.walk():
            if isinstance(node, Power):
                self.assertIsInstance(node.exponent, ParameterNode)
                self.assertTrue(node.exponent.parameter.nonlinear)

    def test_parameter_names_are_unique_and_values_bounded(self):
        config = make_config(
            max_nodes=15,
            variable_probability=0,
            parameter_probability=1,
        )
        generator = TreeGenerator(config, seed=77)

        for expression in generator.generate_many(30):
            parameter_nodes = [
                node for node in expression.walk()
                if isinstance(node, ParameterNode)
            ]
            parameters = expression.parameters()
            self.assertEqual(len(parameter_nodes), len(parameters))
            for node in parameter_nodes:
                parameter = node.parameter
                self.assertGreaterEqual(parameter.value, parameter.lower)
                self.assertLessEqual(parameter.value, parameter.upper)

    def test_leaf_only_catalog_generates_leaf(self):
        config = make_config(
            unary_functions=(),
            binary_operators=(),
            include_power=False,
        )

        expression = TreeGenerator(config, seed=1).generate()

        self.assertEqual(expression.children(), ())

    def test_impossible_minimum_depth_is_rejected(self):
        config = make_config(
            min_depth=2,
            unary_functions=(),
            binary_operators=(),
            include_power=False,
        )

        with self.assertRaisesRegex(ValueError, "cannot satisfy min_depth"):
            TreeGenerator(config, seed=1).generate()

    def test_generated_trees_survive_serialization(self):
        generator = TreeGenerator(make_config(), seed=42)

        for expression in generator.generate_many(30):
            with self.subTest(expression=str(expression)):
                restored = expression_from_dict(expression.to_dict())
                self.assertEqual(restored, expression)

    def test_invalid_method_and_count_are_rejected(self):
        generator = TreeGenerator(make_config(), seed=1)

        with self.assertRaises(ValueError):
            generator.generate("unknown")
        with self.assertRaises(ValueError):
            generator.generate_many(-1)
        self.assertEqual(generator.generate_many(0), [])


class GeneratorConfigValidationTests(unittest.TestCase):
    def test_invalid_configurations_are_rejected(self):
        cases = [
            {"variables": ()},
            {"variables": ("x", "x")},
            {"variables": ("x", "")},
            {"min_depth": 0},
            {"max_depth": 0},
            {"min_depth": 5, "max_depth": 4},
            {"max_nodes": 0},
            {"variable_probability": -0.1},
            {"parameter_probability": 1.1},
            {"leaf_probability": math.nan},
            {"constant_lower": 2, "constant_upper": 1},
            {"exponent_lower": 2, "exponent_upper": 1},
            {"unary_functions": ("unknown",)},
            {"binary_operators": ("unknown",)},
            {"unary_functions": ("sin", "sin")},
            {"binary_operators": ("add", "add")},
        ]

        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    make_config(**overrides)


if __name__ == "__main__":
    unittest.main()
