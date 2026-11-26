"""Immutable expression trees with safe scalar evaluation."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from symbolic_regression.primitives import (
    get_binary_primitive,
    get_unary_primitive,
)


class ExpressionEvaluationError(ValueError):
    """Raised when an expression cannot produce a finite real value."""


@dataclass(frozen=True, slots=True)
class Parameter:
    """A named numeric parameter stored in an expression tree."""

    name: str
    value: float
    lower: float | None = None
    upper: float | None = None
    optimizable: bool = True
    nonlinear: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name must not be empty")
        values = [self.value]
        if self.lower is not None:
            values.append(self.lower)
        if self.upper is not None:
            values.append(self.upper)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("parameter values and bounds must be finite")
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower > self.upper
        ):
            raise ValueError("parameter lower bound exceeds its upper bound")
        if self.lower is not None and self.value < self.lower:
            raise ValueError("parameter value is below its lower bound")
        if self.upper is not None and self.value > self.upper:
            raise ValueError("parameter value is above its upper bound")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "lower": self.lower,
            "upper": self.upper,
            "optimizable": self.optimizable,
            "nonlinear": self.nonlinear,
        }


@dataclass(frozen=True, slots=True)
class Complexity:
    """Structural measurements of an expression."""

    node_count: int
    depth: int
    free_parameters: int
    nonlinear_parameters: int

    def weighted(
        self,
        node_weight: float,
        parameter_weight: float,
        nonlinear_parameter_weight: float,
    ) -> float:
        """Calculate a score using explicitly supplied experiment weights."""

        weights = (
            node_weight,
            parameter_weight,
            nonlinear_parameter_weight,
        )
        valid_weights = all(
            math.isfinite(weight) and weight >= 0 for weight in weights
        )
        if not valid_weights:
            raise ValueError(
                "complexity weights must be finite and non-negative"
            )
        return (
            node_weight * self.node_count
            + parameter_weight * self.free_parameters
            + nonlinear_parameter_weight * self.nonlinear_parameters
        )


class Expr(ABC):
    """Base class for all expression nodes."""

    def evaluate(self, variables: Mapping[str, float]) -> float:
        """Evaluate the tree without executing generated source code."""

        self.parameters()
        return self._evaluate(variables, "$")

    @abstractmethod
    def _evaluate(
        self,
        variables: Mapping[str, float],
        path: str,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def children(self) -> tuple[Expr, ...]:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def to_structural_dict(self) -> dict[str, Any]:
        """Return a representation without free parameter values or names."""

        raise NotImplementedError

    @abstractmethod
    def format(self, resolve_parameters: bool = False) -> str:
        raise NotImplementedError

    def walk(self) -> Iterator[Expr]:
        yield self
        for child in self.children():
            yield from child.walk()

    def parameters(self) -> dict[str, Parameter]:
        """Return named parameters and reject inconsistent reuse of a name."""

        result: dict[str, Parameter] = {}
        for node in self.walk():
            if not isinstance(node, ParameterNode):
                continue
            parameter = node.parameter
            previous = result.get(parameter.name)
            if previous is not None and previous != parameter:
                raise ValueError(
                    f"parameter {parameter.name!r} has conflicting definitions"
                )
            result[parameter.name] = parameter
        return result

    def complexity(self) -> Complexity:
        nodes = tuple(self.walk())
        parameters = self.parameters()
        free_parameters = [
            parameter
            for parameter in parameters.values()
            if parameter.optimizable
        ]
        return Complexity(
            node_count=len(nodes),
            depth=self.depth(),
            free_parameters=len(free_parameters),
            nonlinear_parameters=sum(
                parameter.nonlinear for parameter in free_parameters
            ),
        )

    def depth(self) -> int:
        """Return tree depth; a leaf has depth one."""

        children = self.children()
        if not children:
            return 1
        return 1 + max(child.depth() for child in children)

    def structural_key(self) -> str:
        """Return a stable key for structurally equivalent trees."""

        return json.dumps(
            self.to_structural_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def __str__(self) -> str:
        return self.format()


@dataclass(frozen=True, slots=True)
class Variable(Expr):
    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("variable name must not be empty")

    def _evaluate(
        self,
        variables: Mapping[str, float],
        path: str,
    ) -> float:
        if self.name not in variables:
            raise ExpressionEvaluationError(
                f"{path}: missing variable {self.name!r}"
            )
        try:
            value = float(variables[self.name])
        except (TypeError, ValueError) as error:
            raise ExpressionEvaluationError(
                f"{path}: variable {self.name!r} is not numeric"
            ) from error
        return _finite(value, path)

    def children(self) -> tuple[Expr, ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {"type": "variable", "name": self.name}

    def to_structural_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def format(self, resolve_parameters: bool = False) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Constant(Expr):
    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("constant value must be finite")

    def _evaluate(
        self,
        variables: Mapping[str, float],
        path: str,
    ) -> float:
        return self.value

    def children(self) -> tuple[Expr, ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {"type": "constant", "value": self.value}

    def to_structural_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def format(self, resolve_parameters: bool = False) -> str:
        return _format_number(self.value)


@dataclass(frozen=True, slots=True)
class ParameterNode(Expr):
    parameter: Parameter

    def _evaluate(
        self,
        variables: Mapping[str, float],
        path: str,
    ) -> float:
        return self.parameter.value

    def children(self) -> tuple[Expr, ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {"type": "parameter", "parameter": self.parameter.to_dict()}

    def to_structural_dict(self) -> dict[str, Any]:
        return {"type": "parameter"}

    def format(self, resolve_parameters: bool = False) -> str:
        if resolve_parameters:
            return _format_number(self.parameter.value)
        return self.parameter.name


@dataclass(frozen=True, slots=True)
class Unary(Expr):
    function: str
    operand: Expr

    def __post_init__(self) -> None:
        get_unary_primitive(self.function)

    def _evaluate(
        self,
        variables: Mapping[str, float],
        path: str,
    ) -> float:
        value = self.operand._evaluate(variables, f"{path}.operand")
        primitive = get_unary_primitive(self.function)
        if primitive.validator is not None:
            reason = primitive.validator(value)
            if reason is not None:
                raise ExpressionEvaluationError(f"{path}: {reason}")
        try:
            result = primitive.function(value)
        except (OverflowError, ValueError, ZeroDivisionError) as error:
            raise ExpressionEvaluationError(
                f"{path}: cannot evaluate {self.function}({value})"
            ) from error
        return _finite(result, path)

    def children(self) -> tuple[Expr, ...]:
        return (self.operand,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "unary",
            "function": self.function,
            "operand": self.operand.to_dict(),
        }

    def to_structural_dict(self) -> dict[str, Any]:
        return {
            "type": "unary",
            "function": self.function,
            "operand": self.operand.to_structural_dict(),
        }

    def format(self, resolve_parameters: bool = False) -> str:
        operand = self.operand.format(resolve_parameters)
        return f"{self.function}({operand})"


@dataclass(frozen=True, slots=True)
class Binary(Expr):
    operator: str
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        get_binary_primitive(self.operator)

    def _evaluate(
        self,
        variables: Mapping[str, float],
        path: str,
    ) -> float:
        left = self.left._evaluate(variables, f"{path}.left")
        right = self.right._evaluate(variables, f"{path}.right")
        primitive = get_binary_primitive(self.operator)
        if primitive.validator is not None:
            reason = primitive.validator(left, right)
            if reason is not None:
                raise ExpressionEvaluationError(f"{path}: {reason}")
        try:
            result = primitive.function(left, right)
        except (OverflowError, ValueError, ZeroDivisionError) as error:
            raise ExpressionEvaluationError(
                f"{path}: cannot evaluate {self.operator}"
            ) from error
        return _finite(result, path)

    def children(self) -> tuple[Expr, ...]:
        return (self.left, self.right)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "binary",
            "operator": self.operator,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    def to_structural_dict(self) -> dict[str, Any]:
        return {
            "type": "binary",
            "operator": self.operator,
            "left": self.left.to_structural_dict(),
            "right": self.right.to_structural_dict(),
        }

    def format(self, resolve_parameters: bool = False) -> str:
        symbols = {"add": "+", "sub": "−", "mul": "·", "div": "/"}
        left = self.left.format(resolve_parameters)
        right = self.right.format(resolve_parameters)
        return f"({left} {symbols[self.operator]} {right})"


@dataclass(frozen=True, slots=True)
class Power(Expr):
    base: Expr
    exponent: Expr

    def _evaluate(
        self,
        variables: Mapping[str, float],
        path: str,
    ) -> float:
        base = self.base._evaluate(variables, f"{path}.base")
        exponent = self.exponent._evaluate(variables, f"{path}.exponent")
        is_integer = exponent.is_integer()
        if base < 0 and not is_integer:
            raise ExpressionEvaluationError(
                f"{path}: a negative base requires an integer exponent"
            )
        if base == 0 and exponent < 0:
            raise ExpressionEvaluationError(
                f"{path}: zero cannot have a negative exponent"
            )
        try:
            result = math.pow(base, exponent)
        except (OverflowError, ValueError, ZeroDivisionError) as error:
            raise ExpressionEvaluationError(
                f"{path}: cannot evaluate the power"
            ) from error
        return _finite(result, path)

    def children(self) -> tuple[Expr, ...]:
        return (self.base, self.exponent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "power",
            "base": self.base.to_dict(),
            "exponent": self.exponent.to_dict(),
        }

    def to_structural_dict(self) -> dict[str, Any]:
        return {
            "type": "power",
            "base": self.base.to_structural_dict(),
            "exponent": self.exponent.to_structural_dict(),
        }

    def format(self, resolve_parameters: bool = False) -> str:
        base = self.base.format(resolve_parameters)
        exponent = self.exponent.format(resolve_parameters)
        return f"({base})^({exponent})"


def expression_from_dict(data: Mapping[str, Any]) -> Expr:
    """Restore a tree produced by ``Expr.to_dict``."""

    node_type = data.get("type")
    if node_type == "variable":
        return Variable(name=data["name"])
    if node_type == "constant":
        return Constant(value=data["value"])
    if node_type == "parameter":
        return ParameterNode(parameter=Parameter(**data["parameter"]))
    if node_type == "unary":
        return Unary(
            function=data["function"],
            operand=expression_from_dict(data["operand"]),
        )
    if node_type == "binary":
        return Binary(
            operator=data["operator"],
            left=expression_from_dict(data["left"]),
            right=expression_from_dict(data["right"]),
        )
    if node_type == "power":
        return Power(
            base=expression_from_dict(data["base"]),
            exponent=expression_from_dict(data["exponent"]),
        )
    raise ValueError(f"unknown expression node type: {node_type}")


def _finite(value: float, path: str) -> float:
    if not math.isfinite(value):
        raise ExpressionEvaluationError(
            f"{path}: expression produced a non-finite value"
        )
    return value


def _format_number(value: float) -> str:
    return format(value, ".15g")
