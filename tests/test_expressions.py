import json
import math
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
    expression_from_dict,
)


class ExpressionEvaluationTests(unittest.TestCase):
    def test_linear_expression_with_free_parameter(self):
        expression = Binary(
            "add",
            Binary(
                "mul",
                ParameterNode(Parameter("a", 2.5)),
                Variable("x"),
            ),
            Constant(1),
        )

        self.assertEqual(expression.evaluate({"x": 4}), 11)
        self.assertEqual(str(expression), "((a · x) + 1)")
        self.assertEqual(
            expression.format(resolve_parameters=True),
            "((2.5 · x) + 1)",
        )

    def test_periodic_expression_uses_period_and_shift(self):
        expression = Unary(
            "sin",
            Binary(
                "div",
                Binary(
                    "mul",
                    Constant(2 * math.pi),
                    Binary(
                        "add",
                        Variable("t"),
                        ParameterNode(Parameter("s", 0.5)),
                    ),
                ),
                ParameterNode(
                    Parameter("T", 4, lower=0.1, nonlinear=True)
                ),
            ),
        )

        self.assertAlmostEqual(expression.evaluate({"t": 1.5}), 0)

    def test_all_registered_unary_functions_evaluate(self):
        cases = [
            ("neg", -2, 2),
            ("abs", -2, 2),
            ("sqrt", 9, 3),
            ("exp", 0, 1),
            ("log", math.e, 1),
            ("sin", 0, 0),
            ("cos", 0, 1),
            ("tan", 0, 0),
            ("arcsin", 1, math.pi / 2),
            ("arccos", 1, 0),
            ("arctan", 1, math.pi / 4),
            ("sinh", 0, 0),
            ("cosh", 0, 1),
            ("tanh", 0, 0),
            ("asinh", 0, 0),
            ("acosh", 1, 0),
            ("atanh", 0, 0),
        ]

        for name, argument, expected in cases:
            with self.subTest(function=name):
                expression = Unary(name, Constant(argument))
                self.assertAlmostEqual(expression.evaluate({}), expected)

    def test_missing_variable_is_rejected(self):
        with self.assertRaisesRegex(
            ExpressionEvaluationError,
            "missing variable",
        ):
            Variable("x").evaluate({})

    def test_non_numeric_variable_is_rejected(self):
        with self.assertRaisesRegex(
            ExpressionEvaluationError,
            "not numeric",
        ):
            Variable("x").evaluate({"x": "four"})

    def test_non_finite_variable_is_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ExpressionEvaluationError):
                    Variable("x").evaluate({"x": value})

    def test_unary_domains_are_enforced(self):
        cases = [
            ("sqrt", -1),
            ("log", 0),
            ("arcsin", 2),
            ("arccos", -2),
            ("tan", math.pi / 2),
            ("acosh", 0.5),
            ("atanh", 1),
        ]

        for name, argument in cases:
            with self.subTest(function=name, argument=argument):
                with self.assertRaises(ExpressionEvaluationError):
                    Unary(name, Constant(argument)).evaluate({})

    def test_exp_overflow_is_rejected(self):
        with self.assertRaises(ExpressionEvaluationError):
            Unary("exp", Constant(1000)).evaluate({})

    def test_division_by_zero_is_rejected(self):
        expression = Binary("div", Constant(1), Constant(0))

        with self.assertRaisesRegex(
            ExpressionEvaluationError,
            "division by zero",
        ):
            expression.evaluate({})

    def test_non_finite_binary_result_is_rejected(self):
        expression = Binary("mul", Constant(1e308), Constant(1e308))

        with self.assertRaises(ExpressionEvaluationError):
            expression.evaluate({})

    def test_power_supports_integer_exponent_with_negative_base(self):
        expression = Power(Constant(-2), Constant(3))

        self.assertEqual(expression.evaluate({}), -8)

    def test_power_supports_fractional_exponent_with_positive_base(self):
        expression = Power(Constant(9), Constant(0.5))

        self.assertEqual(expression.evaluate({}), 3)

    def test_power_rejects_invalid_domains(self):
        cases = [(-2, 0.5), (0, -1)]
        for base, exponent in cases:
            with self.subTest(base=base, exponent=exponent):
                with self.assertRaises(ExpressionEvaluationError):
                    Power(Constant(base), Constant(exponent)).evaluate({})

    def test_power_overflow_is_rejected(self):
        with self.assertRaises(ExpressionEvaluationError):
            Power(Constant(10), Constant(10000)).evaluate({})

    def test_unknown_operations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown unary function"):
            Unary("system", Constant(1))
        with self.assertRaisesRegex(ValueError, "unknown binary operator"):
            Binary("exec", Constant(1), Constant(2))


class ExpressionStructureTests(unittest.TestCase):
    def test_complexity_counts_nodes_depth_and_parameters(self):
        expression = Binary(
            "add",
            Binary(
                "mul",
                ParameterNode(Parameter("a", 2)),
                Variable("x"),
            ),
            Constant(1),
        )

        complexity = expression.complexity()

        self.assertEqual(complexity.node_count, 5)
        self.assertEqual(complexity.depth, 3)
        self.assertEqual(complexity.free_parameters, 1)
        self.assertEqual(complexity.nonlinear_parameters, 0)

    def test_nonlinear_parameter_and_weighted_complexity(self):
        expression = Power(
            Variable("x"),
            ParameterNode(Parameter("p", 2, nonlinear=True)),
        )

        complexity = expression.complexity()

        self.assertEqual(complexity.node_count, 3)
        self.assertEqual(complexity.depth, 2)
        self.assertEqual(complexity.free_parameters, 1)
        self.assertEqual(complexity.nonlinear_parameters, 1)
        self.assertEqual(complexity.weighted(1, 2, 3), 8)

    def test_complexity_weights_must_be_valid(self):
        complexity = Constant(1).complexity()

        for weights in [(-1, 0, 0), (1, math.nan, 0), (1, 1, math.inf)]:
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    complexity.weighted(*weights)

    def test_structural_key_ignores_parameter_name_and_value(self):
        first = Binary(
            "mul",
            ParameterNode(Parameter("a", 2)),
            Variable("x"),
        )
        second = Binary(
            "mul",
            ParameterNode(Parameter("b", 50)),
            Variable("x"),
        )

        self.assertEqual(first.structural_key(), second.structural_key())

    def test_structural_key_preserves_fixed_constants_and_operators(self):
        first = Binary("mul", Constant(2), Variable("x"))
        other_constant = Binary("mul", Constant(3), Variable("x"))
        other_operator = Binary("add", Constant(2), Variable("x"))

        self.assertNotEqual(
            first.structural_key(),
            other_constant.structural_key(),
        )
        self.assertNotEqual(
            first.structural_key(),
            other_operator.structural_key(),
        )

    def test_conflicting_parameter_reuse_is_rejected(self):
        expression = Binary(
            "add",
            ParameterNode(Parameter("a", 1)),
            ParameterNode(Parameter("a", 2)),
        )

        with self.assertRaisesRegex(ValueError, "conflicting definitions"):
            expression.evaluate({})

    def test_parameter_bounds_are_validated(self):
        invalid_parameters = [
            {"name": "a", "value": 2, "lower": 0, "upper": 1},
            {"name": "a", "value": -1, "lower": 0},
            {"name": "a", "value": 1, "lower": 2, "upper": 0},
        ]

        for arguments in invalid_parameters:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    Parameter(**arguments)

    def test_serialization_round_trip(self):
        expression = Binary(
            "add",
            Power(
                Variable("x"),
                ParameterNode(Parameter("p", 2, nonlinear=True)),
            ),
            Unary("sin", Constant(0.5)),
        )

        restored = expression_from_dict(expression.to_dict())

        self.assertEqual(restored, expression)
        self.assertEqual(
            restored.structural_key(),
            expression.structural_key(),
        )
        self.assertEqual(
            restored.evaluate({"x": 3}),
            expression.evaluate({"x": 3}),
        )
        json.dumps(expression.to_dict(), allow_nan=False)

    def test_unknown_serialized_node_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "unknown expression node type",
        ):
            expression_from_dict({"type": "unknown"})


if __name__ == "__main__":
    unittest.main()
