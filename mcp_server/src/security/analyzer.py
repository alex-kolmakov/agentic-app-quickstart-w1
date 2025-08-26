import ast
from typing import List, Tuple

class ASTSafetyAnalyzer(ast.NodeVisitor):
    def __init__(self, allowed_imports=None, allowed_calls=None):
        self.allowed_imports = allowed_imports or [
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'pathlib', 
            'os', 'datetime', 'math', 'statistics'
        ]
        self.allowed_calls = allowed_calls or [
            'print', 'len', 'str', 'int', 'float', 'bool', 'list', 'dict',
            'sum', 'min', 'max', 'abs', 'round', 'sorted', 'enumerate',
            'range', 'zip', 'map', 'filter', 'any', 'all'
        ]
        self.unsafe = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name not in self.allowed_imports:
                self.unsafe.append(f"Prohibited import: {alias.name}")

    def visit_ImportFrom(self, node):
        if node.module not in self.allowed_imports:
            self.unsafe.append(f"Prohibited import from module: {node.module}")

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in self.allowed_calls:
                self.unsafe.append(f"Prohibited function call: {func_name}")
        self.generic_visit(node)

    def analyze(self, code_string: str) -> Tuple[bool, List[str]]:
        self.unsafe = []
        try:
            tree = ast.parse(code_string)
            self.visit(tree)
            return not self.unsafe, self.unsafe
        except SyntaxError as e:
            return False, [f"Syntax error: {str(e)}"]
