
import ast
import math
import operator


class ConstantFolder(ast.NodeTransformer):
    BINARY_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    UNARY_OPERATORS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    @staticmethod
    def is_numeric_constant(node):
        return (
            isinstance(node, ast.Constant)
            and type(node.value) in (int, float)
        )

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        operation = self.UNARY_OPERATORS.get(type(node.op))

        if operation is None or not self.is_numeric_constant(node.operand):
            return node

        try:
            result = operation(node.operand.value)

            if type(result) not in (int, float):
                return node

            if isinstance(result, float) and not math.isfinite(result):
                return node

            return ast.copy_location(ast.Constant(value=result), node)

        except (ArithmeticError, TypeError, ValueError, OverflowError):
            return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)

        operation = self.BINARY_OPERATORS.get(type(node.op))

        # Simplify only identities that preserve ordinary Python semantics.
        if isinstance(node.op, ast.Add):
            if self.is_numeric_constant(node.right) and node.right.value == 0:
                return node.left
            if self.is_numeric_constant(node.left) and node.left.value == 0:
                return node.right

        elif isinstance(node.op, ast.Sub):
            if self.is_numeric_constant(node.right) and node.right.value == 0:
                return node.left

        elif isinstance(node.op, ast.Mult):
            if self.is_numeric_constant(node.right) and node.right.value == 1:
                return node.left
            if self.is_numeric_constant(node.left) and node.left.value == 1:
                return node.right

        elif isinstance(node.op, ast.Div):
            if self.is_numeric_constant(node.right) and node.right.value == 1:
                return node.left

        elif isinstance(node.op, ast.Pow):
            if self.is_numeric_constant(node.right) and node.right.value == 1:
                return node.left
            if self.is_numeric_constant(node.right) and node.right.value == 0:
                # Do not simplify 0**0 or expressions that might be complex.
                if self.is_numeric_constant(node.left):
                    try:
                        result = node.left.value ** 0
                        if type(result) in (int, float):
                            return ast.copy_location(
                                ast.Constant(value=result), node
                            )
                    except (ArithmeticError, TypeError, ValueError, OverflowError):
                        pass
                return node

        # Constant folding: evaluate operations whose operands are constants.
        if operation is None:
            return node

        if not (
            self.is_numeric_constant(node.left)
            and self.is_numeric_constant(node.right)
        ):
            return node

        try:
            result = operation(node.left.value, node.right.value)

            if type(result) not in (int, float):
                return node

            if isinstance(result, float) and not math.isfinite(result):
                return node

            return ast.copy_location(ast.Constant(value=result), node)

        except (ArithmeticError, TypeError, ValueError, OverflowError):
            return node


def optimize_code(source_code):
    """Optimize Python source using constant folding and safe identities."""
    try:
        tree = ast.parse(source_code)
        tree = ConstantFolder().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    except SyntaxError as error:
        raise ValueError(f"Invalid Python code: {error}") from error
