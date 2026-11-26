"""Catalogs of allowed mathematical operations.

Only operations registered here can be evaluated by an expression tree.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType


UnaryValidator = Callable[[float], str | None]
BinaryValidator = Callable[[float, float], str | None]


@dataclass(frozen=True, slots=True)
class UnaryPrimitive:
    """Definition of a unary mathematical operation."""

    function: Callable[[float], float]
    validator: UnaryValidator | None = None


@dataclass(frozen=True, slots=True)
class BinaryPrimitive:
    """Definition of a binary mathematical operation."""

    function: Callable[[float, float], float]
    validator: BinaryValidator | None = None


def _accept(value: float) -> str | None:
    return None


def _positive(value: float) -> str | None:
    if value <= 0:
        return "log requires an argument greater than zero"
    return None


def _nonnegative(value: float) -> str | None:
    if value < 0:
        return "sqrt requires a non-negative argument"
    return None


def _closed_unit_interval(value: float) -> str | None:
    if not -1 <= value <= 1:
        return "inverse trigonometric argument must be in [-1, 1]"
    return None


def _acosh_domain(value: float) -> str | None:
    if value < 1:
        return "acosh requires an argument greater than or equal to one"
    return None


def _atanh_domain(value: float) -> str | None:
    if not -1 < value < 1:
        return "atanh requires an argument strictly between -1 and 1"
    return None


def _tan_domain(value: float) -> str | None:
    if abs(math.cos(value)) <= 1e-12:
        return "tan is undefined at the argument pole"
    return None


def _division_domain(_left: float, right: float) -> str | None:
    if right == 0:
        return "division by zero"
    return None


_UNARY_PRIMITIVES = {
    "neg": UnaryPrimitive(function=lambda value: -value),
    "abs": UnaryPrimitive(function=abs),
    "sqrt": UnaryPrimitive(function=math.sqrt, validator=_nonnegative),
    "exp": UnaryPrimitive(function=math.exp),
    "log": UnaryPrimitive(function=math.log, validator=_positive),
    "sin": UnaryPrimitive(function=math.sin),
    "cos": UnaryPrimitive(function=math.cos),
    "tan": UnaryPrimitive(function=math.tan, validator=_tan_domain),
    "arcsin": UnaryPrimitive(
        function=math.asin,
        validator=_closed_unit_interval,
    ),
    "arccos": UnaryPrimitive(
        function=math.acos,
        validator=_closed_unit_interval,
    ),
    "arctan": UnaryPrimitive(function=math.atan),
    "sinh": UnaryPrimitive(function=math.sinh),
    "cosh": UnaryPrimitive(function=math.cosh),
    "tanh": UnaryPrimitive(function=math.tanh),
    "asinh": UnaryPrimitive(function=math.asinh),
    "acosh": UnaryPrimitive(function=math.acosh, validator=_acosh_domain),
    "atanh": UnaryPrimitive(function=math.atanh, validator=_atanh_domain),
}

_BINARY_PRIMITIVES = {
    "add": BinaryPrimitive(function=lambda left, right: left + right),
    "sub": BinaryPrimitive(function=lambda left, right: left - right),
    "mul": BinaryPrimitive(function=lambda left, right: left * right),
    "div": BinaryPrimitive(
        function=lambda left, right: left / right,
        validator=_division_domain,
    ),
}

UNARY_PRIMITIVES: Mapping[str, UnaryPrimitive] = MappingProxyType(
    _UNARY_PRIMITIVES
)
BINARY_PRIMITIVES: Mapping[str, BinaryPrimitive] = MappingProxyType(
    _BINARY_PRIMITIVES
)


def get_unary_primitive(name: str) -> UnaryPrimitive:
    """Return a unary operation or reject an unregistered name."""

    try:
        return UNARY_PRIMITIVES[name]
    except KeyError as error:
        raise ValueError(f"unknown unary function: {name}") from error


def get_binary_primitive(name: str) -> BinaryPrimitive:
    """Return a binary operation or reject an unregistered name."""

    try:
        return BINARY_PRIMITIVES[name]
    except KeyError as error:
        raise ValueError(f"unknown binary operator: {name}") from error
