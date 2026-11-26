import unittest

from symbolic_regression import (
    Binary,
    Constant,
    ExpressionEvaluationError,
    Parameter,
    ParameterNode,
    Power,
    Unary,
    Variable,
    canonicalize,
    iter_subtrees,
    prune_subtree,
    replace_subtree,
    subtree_at,
)


class CanonicalizationTests(unittest.TestCase):
    def test_fixed_arithmetic_is_folded(self):
        expression = Binary(
            "add",
            Constant(2),
            Binary("mul", Constant(3), Constant(4)),
        )

        self.assertEqual(canonicalize(expression), Constant(14))

    def test_invalid_fixed_subtree_is_not_hidden(self):
        expression = Unary("log", Constant(0))

        with self.assertRaises(ExpressionEvaluationError):
            canonicalize(expression)

    def test_neutral_operations_are_removed(self):
        variable = Variable("x")
        cases = [
            Binary("add", variable, Constant(0)),
            Binary("add", Constant(0), variable),
            Binary("sub", variable, Constant(0)),
            Binary("mul", variable, Constant(1)),
            Binary("mul", Constant(1), variable),
            Binary("div", variable, Constant(1)),
            Power(variable, Constant(1)),
        ]

        for expression in cases:
            with self.subTest(expression=str(expression)):
                self.assertEqual(canonicalize(expression), variable)

    def test_free_parameter_is_not_folded_into_constant(self):
        expression = Binary(
            "add",
            ParameterNode(Parameter("a", 2)),
            Constant(1),
        )

        result = canonicalize(expression)

        self.assertIsInstance(result, Binary)
        self.assertEqual(set(result.parameters()), {"a"})
        self.assertEqual(result.evaluate({}), 3)

    def test_zero_multiplication_preserves_unknown_domain(self):
        expression = Binary(
            "mul",
            Constant(0),
            Unary("log", Variable("x")),
        )

        result = canonicalize(expression)

        self.assertIsInstance(result, Binary)
        with self.assertRaises(ExpressionEvaluationError):
            result.evaluate({"x": 0})

    def test_zero_multiplication_of_valid_fixed_terms_is_folded(self):
        expression = Binary(
            "mul",
            Constant(0),
            Binary("add", Constant(3), Constant(4)),
        )

        self.assertEqual(canonicalize(expression), Constant(0))

    def test_division_by_itself_is_not_cancelled(self):
        expression = Binary("div", Variable("x"), Variable("x"))

        result = canonicalize(expression)

        self.assertEqual(result, expression)
        with self.assertRaises(ExpressionEvaluationError):
            result.evaluate({"x": 0})

    def test_exp_log_is_not_cancelled(self):
        expression = Unary("exp", Unary("log", Variable("x")))

        self.assertEqual(canonicalize(expression), expression)

    def test_commutative_arguments_receive_stable_order(self):
        first = Binary("add", Variable("y"), Variable("x"))
        second = Binary("add", Variable("x"), Variable("y"))

        self.assertEqual(canonicalize(first), canonicalize(second))
        self.assertEqual(canonicalize(first).evaluate({"x": 2, "y": 3}), 5)

    def test_canonicalization_is_idempotent(self):
        expression = Binary(
            "mul",
            Binary("add", Variable("y"), Constant(0)),
            Power(Variable("x"), Constant(1)),
        )

        once = canonicalize(expression)

        self.assertEqual(canonicalize(once), once)


class SubtreeOperationTests(unittest.TestCase):
    def setUp(self):
        self.expression = Binary(
            "add",
            Binary(
                "mul",
                ParameterNode(Parameter("a", 2)),
                Variable("x"),
            ),
            Unary("sin", Variable("y")),
        )

    def test_iter_subtrees_returns_parent_first_paths(self):
        paths = [path for path, _node in iter_subtrees(self.expression)]

        self.assertEqual(
            paths,
            [(), (0,), (0, 0), (0, 1), (1,), (1, 0)],
        )

    def test_subtree_at_returns_expected_node(self):
        self.assertEqual(
            subtree_at(self.expression, (0, 1)),
            Variable("x"),
        )
        self.assertEqual(subtree_at(self.expression, ()), self.expression)

    def test_replace_subtree_returns_new_tree(self):
        result = replace_subtree(self.expression, (0, 1), Constant(5))

        self.assertEqual(result.evaluate({"y": 0}), 10)
        self.assertEqual(self.expression.evaluate({"x": 3, "y": 0}), 6)
        self.assertNotEqual(result, self.expression)

    def test_replace_root(self):
        replacement = Variable("replacement")

        self.assertIs(
            replace_subtree(self.expression, (), replacement),
            replacement,
        )

    def test_invalid_paths_are_rejected(self):
        invalid_paths = [(2,), (-1,), (0, 2), (0, 0, 0)]

        for path in invalid_paths:
            with self.subTest(path=path):
                with self.assertRaises(IndexError):
                    subtree_at(self.expression, path)
                with self.assertRaises(IndexError):
                    replace_subtree(self.expression, path, Constant(0))

    def test_prune_subtree_reduces_size_without_mutating_input(self):
        original_size = self.expression.complexity().node_count

        result = prune_subtree(self.expression, (0,), Variable("z"))

        self.assertEqual(
            result.complexity().node_count,
            original_size - 2,
        )
        self.assertEqual(result.evaluate({"z": 3, "y": 0}), 3)
        self.assertEqual(
            self.expression.complexity().node_count,
            original_size,
        )

    def test_prune_subtree_rejects_equal_or_larger_replacement(self):
        replacements = [
            self.expression,
            subtree_at(self.expression, (0,)),
        ]

        for replacement in replacements:
            with self.subTest(replacement=str(replacement)):
                with self.assertRaises(ValueError):
                    prune_subtree(self.expression, (0,), replacement)

    def test_prune_root_to_leaf(self):
        result = prune_subtree(self.expression, (), Constant(0))

        self.assertEqual(result, Constant(0))


if __name__ == "__main__":
    unittest.main()
