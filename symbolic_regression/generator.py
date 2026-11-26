"""Reproducible random generation of expression trees."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from symbolic_regression.expressions import (
    Binary,
    Constant,
    Expr,
    Parameter,
    ParameterNode,
    Power,
    Unary,
    Variable,
)
from symbolic_regression.primitives import (
    BINARY_PRIMITIVES,
    UNARY_PRIMITIVES,
    get_binary_primitive,
    get_unary_primitive,
)


@dataclass(frozen=True, slots=True)
class TreeGeneratorConfig:
    """Configuration for random tree generation.

    A leaf has depth one. ``grow`` permits early leaves; ``full`` uses binary
    arithmetic nodes until ``max_depth`` is reached.
    """

    variables: tuple[str, ...]
    min_depth: int = 1
    max_depth: int = 4
    max_nodes: int = 15
    unary_functions: tuple[str, ...] = tuple(UNARY_PRIMITIVES)
    binary_operators: tuple[str, ...] = tuple(BINARY_PRIMITIVES)
    include_power: bool = True
    variable_probability: float = 0.6
    parameter_probability: float = 0.5
    leaf_probability: float = 0.35
    constant_lower: float = -5.0
    constant_upper: float = 5.0
    exponent_lower: float = -3.0
    exponent_upper: float = 3.0

    def __post_init__(self) -> None:
        for field_name in (
            "variables",
            "unary_functions",
            "binary_operators",
        ):
            value = tuple(getattr(self, field_name))
            object.__setattr__(self, field_name, value)

        if not self.variables:
            raise ValueError("at least one variable is required")
        if any(not name for name in self.variables):
            raise ValueError("variable names must not be empty")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("variable names must be unique")
        if self.min_depth < 1 or self.max_depth < 1:
            raise ValueError("tree depths must be at least one")
        if self.min_depth > self.max_depth:
            raise ValueError("min_depth cannot exceed max_depth")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be at least one")

        for name in self.unary_functions:
            get_unary_primitive(name)
        if len(set(self.unary_functions)) != len(self.unary_functions):
            raise ValueError("unary functions must be unique")
        for name in self.binary_operators:
            get_binary_primitive(name)
        if len(set(self.binary_operators)) != len(self.binary_operators):
            raise ValueError("binary operators must be unique")

        probabilities = (
            self.variable_probability,
            self.parameter_probability,
            self.leaf_probability,
        )
        if not all(0 <= value <= 1 for value in probabilities):
            raise ValueError("probabilities must be between zero and one")
        if not all(math.isfinite(value) for value in probabilities):
            raise ValueError("probabilities must be finite")

        bounds = (
            self.constant_lower,
            self.constant_upper,
            self.exponent_lower,
            self.exponent_upper,
        )
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError("numeric generation bounds must be finite")
        if self.constant_lower > self.constant_upper:
            raise ValueError("constant lower bound exceeds its upper bound")
        if self.exponent_lower > self.exponent_upper:
            raise ValueError("exponent lower bound exceeds its upper bound")


class TreeGenerator:
    """Generate a reproducible sequence of trees using a private RNG."""

    __slots__ = ("config", "seed", "_random", "_parameter_counter")

    def __init__(self, config: TreeGeneratorConfig, seed: int) -> None:
        self.config = config
        self.seed = seed
        self._random = random.Random(seed)
        self._parameter_counter = 0

    def generate(self, method: str = "grow") -> Expr:
        """Generate one tree using ``grow`` or ``full``."""

        if method not in {"grow", "full"}:
            raise ValueError("generation method must be 'grow' or 'full'")
        self._parameter_counter = 0
        if method == "full":
            return self._generate_full_tree()
        minimum = self._minimum_nodes(self.config.min_depth - 1)
        if minimum is None or minimum > self.config.max_nodes:
            raise ValueError(
                "configuration cannot satisfy min_depth within max_nodes"
            )
        return self._generate_grow(1, self.config.max_nodes)

    def generate_many(self, count: int, method: str = "grow") -> list[Expr]:
        """Generate a consecutive sequence from the same random stream."""

        if count < 0:
            raise ValueError("tree count must not be negative")
        return [self.generate(method) for _ in range(count)]

    def _generate_full_tree(self) -> Expr:
        required_nodes = 2 ** self.config.max_depth - 1
        if required_nodes > self.config.max_nodes:
            raise ValueError(
                "full tree requires at least 2**max_depth - 1 nodes"
            )
        if self.config.max_depth > 1 and not self.config.binary_operators:
            raise ValueError(
                "full generation requires at least one binary operator"
            )
        return self._generate_full(1)

    def _generate_full(self, depth: int) -> Expr:
        if depth == self.config.max_depth:
            return self._leaf()
        operator = self._random.choice(self.config.binary_operators)
        return Binary(
            operator,
            self._generate_full(depth + 1),
            self._generate_full(depth + 1),
        )

    def _generate_grow(self, depth: int, budget: int) -> Expr:
        required_levels = max(0, self.config.min_depth - depth)
        minimum = self._minimum_nodes(required_levels)
        if minimum is None or budget < minimum:
            raise ValueError("node budget cannot satisfy min_depth")

        if depth == self.config.max_depth or budget == 1:
            if required_levels:
                raise ValueError("maximum depth cannot satisfy min_depth")
            return self._leaf()

        leaf_allowed = required_levels == 0
        if leaf_allowed and self._random.random() < self._probability_leaf:
            return self._leaf()

        child_levels = max(0, required_levels - 1)
        child_minimum = self._minimum_nodes(child_levels)
        if child_minimum is None:
            raise ValueError("no internal operation can satisfy min_depth")

        choices = []
        can_descend = depth < self.config.max_depth
        if (
            can_descend
            and self.config.unary_functions
            and budget >= 1 + child_minimum
        ):
            choices.append("unary")
        if (
            can_descend
            and self.config.binary_operators
            and budget >= 1 + 2 * child_minimum
        ):
            choices.append("binary")
        if (
            can_descend
            and leaf_allowed
            and self.config.include_power
            and budget >= 3
        ):
            choices.append("power")

        if not choices:
            if leaf_allowed:
                return self._leaf()
            raise ValueError("no operation fits the configured constraints")

        choice = self._random.choice(choices)
        if choice == "unary":
            function = self._random.choice(self.config.unary_functions)
            return Unary(
                function,
                self._generate_grow(depth + 1, budget - 1),
            )
        if choice == "power":
            base = self._generate_grow(depth + 1, budget - 2)
            exponent = self._numeric_leaf(
                self.config.exponent_lower,
                self.config.exponent_upper,
                nonlinear=True,
            )
            return Power(base, exponent)

        operator = self._random.choice(self.config.binary_operators)
        remaining = budget - 1
        extra = remaining - 2 * child_minimum
        left_extra = self._random.randint(0, extra) if extra else 0
        left_budget = child_minimum + left_extra
        right_budget = child_minimum + extra - left_extra
        return Binary(
            operator,
            self._generate_grow(depth + 1, left_budget),
            self._generate_grow(depth + 1, right_budget),
        )

    @property
    def _probability_leaf(self) -> float:
        return self.config.leaf_probability

    def _minimum_nodes(self, internal_levels: int) -> int | None:
        if internal_levels <= 0:
            return 1
        if self.config.unary_functions:
            return internal_levels + 1
        if self.config.binary_operators:
            return 2 ** (internal_levels + 1) - 1
        return None

    def _leaf(self) -> Expr:
        if self._random.random() < self.config.variable_probability:
            return Variable(self._random.choice(self.config.variables))
        return self._numeric_leaf(
            self.config.constant_lower,
            self.config.constant_upper,
            nonlinear=False,
        )

    def _numeric_leaf(
        self,
        lower: float,
        upper: float,
        nonlinear: bool,
    ) -> Expr:
        value = self._random.uniform(lower, upper)
        if self._random.random() >= self.config.parameter_probability:
            return Constant(value)
        name = f"p{self._parameter_counter}"
        self._parameter_counter += 1
        return ParameterNode(
            Parameter(
                name=name,
                value=value,
                lower=lower,
                upper=upper,
                nonlinear=nonlinear,
            )
        )
