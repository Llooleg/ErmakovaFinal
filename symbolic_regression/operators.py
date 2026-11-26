"""Reproducible genetic operators for expression trees."""

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
)
from symbolic_regression.generator import TreeGenerator, TreeGeneratorConfig
from symbolic_regression.transforms import (
    NodePath,
    canonicalize,
    iter_subtrees,
    replace_subtree,
    subtree_at,
)


class GeneticOperationError(RuntimeError):
    """Raised when an operator cannot satisfy its configured constraints."""


@dataclass(frozen=True, slots=True)
class _NumericTarget:
    name: str | None
    paths: tuple[NodePath, ...]
    value: float
    lower: float | None
    upper: float | None


class GeneticOperators:
    """Apply crossover and mutations using private reproducible RNGs."""

    def __init__(
        self,
        config: TreeGeneratorConfig,
        seed: int,
        parameter_mutation_scale: float = 0.15,
        max_attempts: int = 50,
    ) -> None:
        if not math.isfinite(parameter_mutation_scale):
            raise ValueError("parameter mutation scale must be finite")
        if parameter_mutation_scale <= 0:
            raise ValueError("parameter mutation scale must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self.config = config
        self.seed = seed
        self.parameter_mutation_scale = parameter_mutation_scale
        self.max_attempts = max_attempts
        self._random = random.Random(seed)
        self.tree_generator = TreeGenerator(config, seed)

    def crossover(
        self,
        first: Expr,
        second: Expr,
    ) -> tuple[Expr, Expr]:
        """Exchange random subtrees and return two valid children."""

        self._validate_parent(first)
        self._validate_parent(second)
        first_paths = tuple(iter_subtrees(first))
        second_paths = tuple(iter_subtrees(second))
        last_error: Exception | None = None

        for _ in range(self.max_attempts):
            first_path = self._random.choice(first_paths)[0]
            second_path = self._random.choice(second_paths)[0]
            try:
                return self.crossover_at(
                    first,
                    second,
                    first_path,
                    second_path,
                )
            except (ValueError, IndexError, GeneticOperationError) as error:
                last_error = error

        raise GeneticOperationError(
            "crossover could not satisfy the configured constraints"
        ) from last_error

    def crossover_at(
        self,
        first: Expr,
        second: Expr,
        first_path: NodePath,
        second_path: NodePath,
    ) -> tuple[Expr, Expr]:
        """Exchange subtrees with deterministic parameter renaming."""

        self._validate_parent(first)
        self._validate_parent(second)
        first_insert = subtree_at(second, second_path)
        second_insert = subtree_at(first, first_path)
        first_insert = self._rename_parameters(
            first_insert,
            set() if not first_path else set(first.parameters()),
            "cross_second_",
        )
        second_insert = self._rename_parameters(
            second_insert,
            set() if not second_path else set(second.parameters()),
            "cross_first_",
        )
        first_child = replace_subtree(first, first_path, first_insert)
        second_child = replace_subtree(second, second_path, second_insert)
        return self._finalize(first_child), self._finalize(second_child)

    def subtree_mutation(self, expression: Expr) -> Expr:
        """Replace a random subtree with a newly generated tree."""

        self._validate_parent(expression)
        paths = tuple(iter_subtrees(expression))
        last_error: Exception | None = None

        for _ in range(self.max_attempts):
            path = self._random.choice(paths)[0]
            replacement = self.tree_generator.generate("grow")
            try:
                return self.subtree_mutation_at(expression, path, replacement)
            except (ValueError, IndexError, GeneticOperationError) as error:
                last_error = error

        raise GeneticOperationError(
            "subtree mutation could not satisfy the configured constraints"
        ) from last_error

    def subtree_mutation_at(
        self,
        expression: Expr,
        path: NodePath,
        replacement: Expr,
    ) -> Expr:
        """Replace a specified subtree without parameter name collisions."""

        self._validate_parent(expression)
        occupied = set() if not path else set(expression.parameters())
        replacement = self._rename_parameters(
            replacement,
            occupied,
            "mutation_",
        )
        return self._finalize(replace_subtree(expression, path, replacement))

    def point_mutation(self, expression: Expr) -> Expr:
        """Replace a unary function or binary operator, preserving arity."""

        self._validate_parent(expression)
        candidates = []
        for path, node in iter_subtrees(expression):
            if isinstance(node, Unary):
                alternatives = tuple(
                    name
                    for name in self.config.unary_functions
                    if name != node.function
                )
                if alternatives:
                    candidates.append((path, node, alternatives))
            elif isinstance(node, Binary):
                alternatives = tuple(
                    name
                    for name in self.config.binary_operators
                    if name != node.operator
                )
                if alternatives:
                    candidates.append((path, node, alternatives))

        if not candidates:
            raise GeneticOperationError(
                "point mutation requires a replaceable function or operator"
            )

        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            path, node, alternatives = self._random.choice(candidates)
            replacement_name = self._random.choice(alternatives)
            if isinstance(node, Unary):
                replacement = Unary(replacement_name, node.operand)
            else:
                replacement = Binary(
                    replacement_name,
                    node.left,
                    node.right,
                )
            try:
                mutated = replace_subtree(expression, path, replacement)
                return self._finalize(mutated)
            except (ValueError, GeneticOperationError) as error:
                last_error = error

        raise GeneticOperationError(
            "point mutation could not produce a valid expression"
        ) from last_error

    def parameter_mutation(self, expression: Expr) -> Expr:
        """Mutate a free parameter or numeric constant within its bounds."""

        self._validate_parent(expression)
        targets = self._numeric_targets(expression)
        if not targets:
            raise GeneticOperationError(
                "parameter mutation requires a mutable numeric leaf"
            )

        target = self._random.choice(targets)
        value = self._mutate_value(target)
        result = expression
        for path in target.paths:
            node = subtree_at(expression, path)
            if target.name is None:
                replacement: Expr = Constant(value)
            else:
                if not isinstance(node, ParameterNode):
                    raise TypeError("parameter target no longer matches")
                replacement = ParameterNode(
                    Parameter(
                        name=target.name,
                        value=value,
                        lower=target.lower,
                        upper=target.upper,
                        optimizable=node.parameter.optimizable,
                        nonlinear=node.parameter.nonlinear,
                    )
                )
            result = replace_subtree(result, path, replacement)
        return self._finalize(result)

    def _numeric_targets(self, expression: Expr) -> list[_NumericTarget]:
        groups: dict[str, list[NodePath]] = {}
        parameters = expression.parameters()
        constants: list[_NumericTarget] = []

        for path, node in iter_subtrees(expression):
            if isinstance(node, ParameterNode):
                parameter = node.parameter
                if not parameter.optimizable:
                    continue
                if (
                    parameter.lower is not None
                    and parameter.lower == parameter.upper
                ):
                    continue
                groups.setdefault(parameter.name, []).append(path)
            elif isinstance(node, Constant):
                lower = self.config.constant_lower
                upper = self.config.constant_upper
                if self._is_power_exponent(expression, path):
                    lower = self.config.exponent_lower
                    upper = self.config.exponent_upper
                if lower == upper:
                    continue
                constants.append(
                    _NumericTarget(
                        name=None,
                        paths=(path,),
                        value=node.value,
                        lower=lower,
                        upper=upper,
                    )
                )

        grouped = [
            _NumericTarget(
                name=name,
                paths=tuple(paths),
                value=parameters[name].value,
                lower=parameters[name].lower,
                upper=parameters[name].upper,
            )
            for name, paths in groups.items()
        ]
        return grouped + constants

    def _is_power_exponent(self, expression: Expr, path: NodePath) -> bool:
        if not path or path[-1] != 1:
            return False
        return isinstance(subtree_at(expression, path[:-1]), Power)

    def _mutate_value(self, target: _NumericTarget) -> float:
        if target.lower is not None and target.upper is not None:
            deviation = self.parameter_mutation_scale * (
                target.upper - target.lower
            )
        else:
            deviation = self.parameter_mutation_scale * max(
                1.0,
                abs(target.value),
            )

        for _ in range(self.max_attempts):
            value = self._random.gauss(target.value, deviation)
            if target.lower is not None:
                value = max(target.lower, value)
            if target.upper is not None:
                value = min(target.upper, value)
            if math.isfinite(value) and value != target.value:
                return value

        raise GeneticOperationError(
            f"numeric target {target.name!r} cannot be changed"
        )

    def _rename_parameters(
        self,
        expression: Expr,
        occupied: set[str],
        prefix: str,
    ) -> Expr:
        parameters = expression.parameters()
        mapping: dict[str, str] = {}
        used = set(occupied)
        for name in parameters:
            candidate = name
            if candidate in used:
                candidate = prefix + name
                suffix = 1
                while candidate in used:
                    candidate = f"{prefix}{name}_{suffix}"
                    suffix += 1
            mapping[name] = candidate
            used.add(candidate)

        result = expression
        for path, node in tuple(iter_subtrees(expression)):
            if not isinstance(node, ParameterNode):
                continue
            old = node.parameter
            replacement = Parameter(
                name=mapping[old.name],
                value=old.value,
                lower=old.lower,
                upper=old.upper,
                optimizable=old.optimizable,
                nonlinear=old.nonlinear,
            )
            result = replace_subtree(
                result,
                path,
                ParameterNode(replacement),
            )
        return result

    def _validate_parent(self, expression: Expr) -> None:
        if not self._within_limits(expression):
            raise ValueError(
                "parent expression exceeds depth or node constraints"
            )

    def _finalize(self, expression: Expr) -> Expr:
        result = canonicalize(expression)
        if not self._within_limits(result):
            raise GeneticOperationError(
                "operation result exceeds depth or node constraints"
            )
        return result

    def _within_limits(self, expression: Expr) -> bool:
        complexity = expression.complexity()
        return (
            complexity.depth <= self.config.max_depth
            and complexity.node_count <= self.config.max_nodes
        )
