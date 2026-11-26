"""Conservative canonicalization and immutable subtree operations."""

from __future__ import annotations

from collections.abc import Iterator

from symbolic_regression.expressions import (
    Binary,
    Constant,
    Expr,
    ParameterNode,
    Power,
    Unary,
    Variable,
)


NodePath = tuple[int, ...]


def canonicalize(expression: Expr) -> Expr:
    """Simplify an expression without intentionally changing its domain.

    Fixed subtrees are evaluated and folded. Free parameters are never folded
    into fixed constants. Rules that could remove a domain check are skipped.
    """

    if isinstance(expression, Unary):
        rebuilt = Unary(
            expression.function,
            canonicalize(expression.operand),
        )
    elif isinstance(expression, Binary):
        rebuilt = Binary(
            expression.operator,
            canonicalize(expression.left),
            canonicalize(expression.right),
        )
    elif isinstance(expression, Power):
        rebuilt = Power(
            canonicalize(expression.base),
            canonicalize(expression.exponent),
        )
    elif isinstance(expression, (Variable, Constant, ParameterNode)):
        rebuilt = expression
    else:
        raise TypeError(f"unsupported expression node: {type(expression)!r}")
    return _canonical_node(rebuilt)


def iter_subtrees(expression: Expr) -> Iterator[tuple[NodePath, Expr]]:
    """Yield every node and its stable child-index path, parents first."""

    def visit(node: Expr, path: NodePath) -> Iterator[tuple[NodePath, Expr]]:
        yield path, node
        for index, child in enumerate(node.children()):
            yield from visit(child, path + (index,))

    yield from visit(expression, ())


def subtree_at(expression: Expr, path: NodePath) -> Expr:
    """Return a subtree at a path such as ``(0, 1)``."""

    node = expression
    for depth, index in enumerate(path):
        children = node.children()
        if index < 0 or index >= len(children):
            raise IndexError(
                f"invalid subtree path {path!r} at depth {depth}"
            )
        node = children[index]
    return node


def replace_subtree(
    expression: Expr,
    path: NodePath,
    replacement: Expr,
) -> Expr:
    """Replace one subtree without mutating the input tree."""

    def replace(node: Expr, remaining: NodePath) -> Expr:
        if not remaining:
            return replacement
        index = remaining[0]
        children = list(node.children())
        if index < 0 or index >= len(children):
            raise IndexError(f"invalid subtree path {path!r}")
        children[index] = replace(children[index], remaining[1:])
        return _rebuild_with_children(node, tuple(children))

    return replace(expression, path)


def prune_subtree(
    expression: Expr,
    path: NodePath,
    replacement: Expr,
) -> Expr:
    """Replace a subtree only if the replacement is structurally smaller."""

    target = subtree_at(expression, path)
    target_size = target.complexity().node_count
    replacement_size = replacement.complexity().node_count
    if replacement_size >= target_size:
        raise ValueError(
            "pruning requires a replacement with fewer nodes than the subtree"
        )
    return replace_subtree(expression, path, replacement)


def _canonical_node(expression: Expr) -> Expr:
    if _is_fixed(expression):
        return Constant(expression.evaluate({}))

    if isinstance(expression, Power):
        if _has_constant_value(expression.exponent, 1):
            return expression.base
        return expression

    if not isinstance(expression, Binary):
        return expression

    left = expression.left
    right = expression.right
    operator = expression.operator

    if operator == "add":
        if _has_constant_value(right, 0):
            return left
        if _has_constant_value(left, 0):
            return right
    elif operator == "sub" and _has_constant_value(right, 0):
        return left
    elif operator == "mul":
        if _has_constant_value(right, 1):
            return left
        if _has_constant_value(left, 1):
            return right
    elif operator == "div" and _has_constant_value(right, 1):
        return left

    if operator in {"add", "mul"}:
        ordered = sorted(
            (left, right),
            key=lambda node: node.structural_key(),
        )
        if ordered != [left, right]:
            return Binary(operator, ordered[0], ordered[1])

    return expression


def _is_fixed(expression: Expr) -> bool:
    return all(
        not isinstance(node, (Variable, ParameterNode))
        for node in expression.walk()
    )


def _has_constant_value(expression: Expr, value: float) -> bool:
    return isinstance(expression, Constant) and expression.value == value


def _rebuild_with_children(
    expression: Expr,
    children: tuple[Expr, ...],
) -> Expr:
    if isinstance(expression, Unary) and len(children) == 1:
        return Unary(expression.function, children[0])
    if isinstance(expression, Binary) and len(children) == 2:
        return Binary(expression.operator, children[0], children[1])
    if isinstance(expression, Power) and len(children) == 2:
        return Power(children[0], children[1])
    if isinstance(expression, (Variable, Constant, ParameterNode)):
        if not children:
            return expression
    raise ValueError(
        f"invalid children for expression node: {type(expression)!r}"
    )
